#include "kernels/rope.h"
#include <cmath>

namespace rf {

// Compute freqs for triangle RoPE:
// 1. pos[B,S,3,3] reshaped as [B,S,9], multiply each of 9 dims by each learned freq → [B,S,9,nf]
// 2. rearrange to [B,1,S, 9*nf]
// 3. hf_format: cat([freqs, freqs]) → [B,1,S, 18*nf]
// 4. zero-pad to head_dim, then cos/sin
__global__ void triangle_rope_freqs_kernel(
    const half* __restrict__ pos,         // [B, S, 9]
    const float* __restrict__ learned_freqs, // [num_freqs]
    half* __restrict__ out_cos,           // [B, 1, S, head_dim]
    half* __restrict__ out_sin,           // [B, 1, S, head_dim]
    int seq_len, int num_freqs, int head_dim) {

    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;

    // freq_dim = 9 * num_freqs (before hf doubling)
    int freq_dim = 9 * num_freqs;
    int hf_dim = freq_dim; // half of the hf-doubled dimension

    float val = 0.0f;
    // Python layout: [freqs(freq_dim), zeros(pad), freqs(freq_dim), zeros(pad)]
    // where each half is head_dim/2 and pad = head_dim/2 - freq_dim
    int half_hd = head_dim / 2;
    int local_d = d % half_hd; // position within each half

    if (local_d < freq_dim) {
        int vert_coord = local_d / num_freqs; // 0..8
        int freq_idx = local_d % num_freqs;
        float p = __half2float(pos[(b * seq_len + s) * 9 + vert_coord]);
        val = p * learned_freqs[freq_idx];
    }
    // else: val stays 0 (zero-padding region within each half)

    int out_idx = (b * seq_len + s) * head_dim + d;
    out_cos[out_idx] = __float2half(cosf(val));
    out_sin[out_idx] = __float2half(sinf(val));
}

void triangle_rope_freqs(const half* pos, const float* learned_freqs,
                         half* out_cos, half* out_sin,
                         int batch, int seq_len, int num_freqs, int head_dim,
                         cudaStream_t stream) {
    int threads = 128;
    int blocks_x = (head_dim + threads - 1) / threads;
    dim3 grid(blocks_x, seq_len, batch);

    triangle_rope_freqs_kernel<<<grid, threads, 0, stream>>>(
        pos, learned_freqs, out_cos, out_sin,
        seq_len, num_freqs, head_dim);
}

// rotate_half_hf: [-x2, x1] where x1 = x[..., :D/2], x2 = x[..., D/2:]
// Apply: out = x * cos + rotate_half_hf(x) * sin
__global__ void apply_rope_kernel(
    half* __restrict__ q,     // [B, H, S, D]
    half* __restrict__ k,     // [B, H, S, D]
    const half* __restrict__ cos_vals, // [B, 1, S, D]
    const half* __restrict__ sin_vals, // [B, 1, S, D]
    int num_heads, int seq_len, int head_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = idx % head_dim;
    int h = idx / head_dim;
    if (h >= num_heads) return;

    int cs_idx = (b * seq_len + s) * head_dim + d;
    float c = __half2float(cos_vals[cs_idx]);
    float sn = __half2float(sin_vals[cs_idx]);

    // rotate_half_hf index
    int half_d = head_dim / 2;
    int rot_d = (d < half_d) ? (d + half_d) : (d - half_d);
    float rot_sign = (d < half_d) ? -1.0f : 1.0f;

    auto apply = [&](half* tensor) {
        int qk_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
        float x_d = __half2float(tensor[qk_idx + d]);
        float x_rot = __half2float(tensor[qk_idx + rot_d]) * rot_sign;
        tensor[qk_idx + d] = __float2half(x_d * c + x_rot * sn);
    };

    apply(q);
    apply(k);
}

void apply_rope(half* q, half* k, const half* cos_vals, const half* sin_vals,
                int batch, int num_heads, int seq_len, int head_dim,
                cudaStream_t stream) {
    int threads_per_token = num_heads * head_dim;
    int threads = 256;
    int blocks_x = (threads_per_token + threads - 1) / threads;
    dim3 grid(blocks_x, seq_len, batch);

    apply_rope_kernel<<<grid, threads, 0, stream>>>(
        q, k, cos_vals, sin_vals, num_heads, seq_len, head_dim);
}

// Single tensor version (for cross-attn Q or K separately)
__global__ void apply_rope_one_kernel(
    half* __restrict__ x,
    const half* __restrict__ cos_vals,
    const half* __restrict__ sin_vals,
    int num_heads, int seq_len, int head_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z;
    int s = blockIdx.y;
    int d = idx % head_dim;
    int h = idx / head_dim;
    if (h >= num_heads) return;

    int cs_idx = (b * seq_len + s) * head_dim + d;
    float c = __half2float(cos_vals[cs_idx]);
    float sn = __half2float(sin_vals[cs_idx]);

    int half_d = head_dim / 2;
    int rot_d = (d < half_d) ? (d + half_d) : (d - half_d);
    float rot_sign = (d < half_d) ? -1.0f : 1.0f;

    int x_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
    float x_d = __half2float(x[x_idx + d]);
    float x_rot = __half2float(x[x_idx + rot_d]) * rot_sign;
    x[x_idx + d] = __float2half(x_d * c + x_rot * sn);
}

void apply_rope_one(half* x, const half* cos_vals, const half* sin_vals,
                    int batch, int num_heads, int seq_len, int head_dim,
                    cudaStream_t stream) {
    int threads_per_token = num_heads * head_dim;
    int threads = 256;
    int blocks_x = (threads_per_token + threads - 1) / threads;
    dim3 grid(blocks_x, seq_len, batch);

    apply_rope_one_kernel<<<grid, threads, 0, stream>>>(
        x, cos_vals, sin_vals, num_heads, seq_len, head_dim);
}

// fp32 variants
__global__ void apply_rope_f32_kernel(
    float* __restrict__ q, float* __restrict__ k,
    const half* __restrict__ cos_vals, const half* __restrict__ sin_vals,
    int num_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z, s = blockIdx.y;
    int d = idx % head_dim, h = idx / head_dim;
    if (h >= num_heads) return;

    int cs_idx = (b * seq_len + s) * head_dim + d;
    float c = __half2float(cos_vals[cs_idx]);
    float sn = __half2float(sin_vals[cs_idx]);

    int half_d = head_dim / 2;
    int rot_d = (d < half_d) ? (d + half_d) : (d - half_d);
    float rot_sign = (d < half_d) ? -1.0f : 1.0f;

    auto apply = [&](float* tensor) {
        int base = ((b * num_heads + h) * seq_len + s) * head_dim;
        float x_d = tensor[base + d];
        float x_rot = tensor[base + rot_d] * rot_sign;
        tensor[base + d] = x_d * c + x_rot * sn;
    };
    apply(q);
    apply(k);
}

void apply_rope_f32(float* q, float* k, const half* cos_vals, const half* sin_vals,
                    int batch, int num_heads, int seq_len, int head_dim,
                    cudaStream_t stream) {
    int t = 256, n = num_heads * head_dim;
    dim3 grid((n + t - 1) / t, seq_len, batch);
    apply_rope_f32_kernel<<<grid, t, 0, stream>>>(q, k, cos_vals, sin_vals, num_heads, seq_len, head_dim);
}

__global__ void apply_rope_one_f32_kernel(
    float* __restrict__ x, const half* __restrict__ cos_vals, const half* __restrict__ sin_vals,
    int num_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.z, s = blockIdx.y;
    int d = idx % head_dim, h = idx / head_dim;
    if (h >= num_heads) return;

    int cs_idx = (b * seq_len + s) * head_dim + d;
    float c = __half2float(cos_vals[cs_idx]);
    float sn = __half2float(sin_vals[cs_idx]);

    int half_d = head_dim / 2;
    int rot_d = (d < half_d) ? (d + half_d) : (d - half_d);
    float rot_sign = (d < half_d) ? -1.0f : 1.0f;

    int base = ((b * num_heads + h) * seq_len + s) * head_dim;
    float x_d = x[base + d];
    float x_rot = x[base + rot_d] * rot_sign;
    x[base + d] = x_d * c + x_rot * sn;
}

void apply_rope_one_f32(float* x, const half* cos_vals, const half* sin_vals,
                        int batch, int num_heads, int seq_len, int head_dim,
                        cudaStream_t stream) {
    int t = 256, n = num_heads * head_dim;
    dim3 grid((n + t - 1) / t, seq_len, batch);
    apply_rope_one_f32_kernel<<<grid, t, 0, stream>>>(x, cos_vals, sin_vals, num_heads, seq_len, head_dim);
}

} // namespace rf
