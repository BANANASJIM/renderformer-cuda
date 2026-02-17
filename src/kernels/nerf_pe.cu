#include "kernels/nerf_pe.h"
#include <cmath>
#include <vector>

namespace rf {

// NeRF PE: [x, sin(f₀x), sin(f₁x), ..., cos(f₀x), cos(f₁x), ...]
// Python: sin(cat([scaled, scaled + pi/2])) = cat([sin(scaled), cos(scaled)])
__global__ void nerf_pe_kernel(const half* __restrict__ input,
                               half* __restrict__ output,
                               int seq_len, int in_dim, int num_freqs,
                               int out_dim, bool include_input,
                               const float* __restrict__ freqs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * out_dim;
    int batch_offset = blockIdx.y;

    if (idx >= total) return;

    int token = idx / out_dim;
    int feat = idx % out_dim;

    const half* in_ptr = input + (batch_offset * seq_len + token) * in_dim;
    half* out_ptr = output + (batch_offset * seq_len + token) * out_dim;

    int encoded_dim = in_dim * num_freqs * 2;
    int input_offset = include_input ? in_dim : 0;

    if (include_input && feat < in_dim) {
        // Copy input directly
        out_ptr[feat] = in_ptr[feat];
    } else {
        int enc_feat = feat - input_offset;
        if (enc_feat >= 0 && enc_feat < encoded_dim) {
            // First half: sin, second half: cos
            int half_dim = in_dim * num_freqs;
            bool is_cos = (enc_feat >= half_dim);
            int sin_idx = is_cos ? (enc_feat - half_dim) : enc_feat;

            int d = sin_idx / num_freqs;   // which input dimension
            int f = sin_idx % num_freqs;   // which frequency

            float val = __half2float(in_ptr[d]) * freqs[f];
            float result = is_cos ? cosf(val) : sinf(val);
            out_ptr[feat] = __float2half(result);
        }
    }
}

void nerf_pe(const half* input, half* output,
             int batch, int seq_len, int in_dim, int num_freqs,
             float min_freq_exp, float max_freq_exp,
             bool include_input, cudaStream_t stream) {
    int out_dim = in_dim * num_freqs * 2 + (include_input ? in_dim : 0);

    // Precompute frequencies on host then upload
    std::vector<float> h_freqs(num_freqs);
    for (int i = 0; i < num_freqs; ++i) {
        float exp = min_freq_exp + (max_freq_exp - min_freq_exp) * i / std::max(num_freqs - 1, 1);
        h_freqs[i] = std::pow(2.0f, exp);
    }

    // TODO: cache this allocation
    float* d_freqs;
    cudaMalloc(&d_freqs, num_freqs * sizeof(float));
    cudaMemcpyAsync(d_freqs, h_freqs.data(), num_freqs * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    int total = seq_len * out_dim;
    int threads = 256;
    int blocks_x = (total + threads - 1) / threads;
    dim3 grid(blocks_x, batch);

    nerf_pe_kernel<<<grid, threads, 0, stream>>>(
        input, output, seq_len, in_dim, num_freqs, out_dim,
        include_input, d_freqs);

    cudaFree(d_freqs);
}

// fp32 input kernel
__global__ void nerf_pe_f32_kernel(const float* __restrict__ input,
                                    half* __restrict__ output,
                                    int seq_len, int in_dim, int num_freqs,
                                    int out_dim, bool include_input,
                                    const float* __restrict__ freqs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * out_dim;
    int batch_offset = blockIdx.y;
    if (idx >= total) return;

    int token = idx / out_dim;
    int feat = idx % out_dim;
    const float* in_ptr = input + (batch_offset * seq_len + token) * in_dim;
    half* out_ptr = output + (batch_offset * seq_len + token) * out_dim;

    int input_offset = include_input ? in_dim : 0;
    if (include_input && feat < in_dim) {
        out_ptr[feat] = __float2half(in_ptr[feat]);
    } else {
        int enc_feat = feat - input_offset;
        int half_dim = in_dim * num_freqs;
        bool is_cos = (enc_feat >= half_dim);
        int sin_idx = is_cos ? (enc_feat - half_dim) : enc_feat;
        int d = sin_idx / num_freqs;
        int f = sin_idx % num_freqs;
        float val = in_ptr[d] * freqs[f];
        out_ptr[feat] = __float2half(is_cos ? cosf(val) : sinf(val));
    }
}

void nerf_pe_f32(const float* input, half* output,
                 int batch, int seq_len, int in_dim, int num_freqs,
                 float min_freq_exp, float max_freq_exp,
                 bool include_input, cudaStream_t stream) {
    int out_dim = in_dim * num_freqs * 2 + (include_input ? in_dim : 0);
    std::vector<float> h_freqs(num_freqs);
    for (int i = 0; i < num_freqs; ++i) {
        float exp = min_freq_exp + (max_freq_exp - min_freq_exp) * i / std::max(num_freqs - 1, 1);
        h_freqs[i] = std::pow(2.0f, exp);
    }
    float* d_freqs;
    cudaMalloc(&d_freqs, num_freqs * sizeof(float));
    cudaMemcpyAsync(d_freqs, h_freqs.data(), num_freqs * sizeof(float), cudaMemcpyHostToDevice, stream);

    int total = seq_len * out_dim;
    int threads = 256;
    dim3 grid((total + threads - 1) / threads, batch);
    nerf_pe_f32_kernel<<<grid, threads, 0, stream>>>(
        input, output, seq_len, in_dim, num_freqs, out_dim, include_input, d_freqs);
    cudaFree(d_freqs);
}

} // namespace rf
