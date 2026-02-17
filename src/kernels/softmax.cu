#include "kernels/softmax.h"
#include <cfloat>

namespace rf {

// One block per row, shared memory reduction
__global__ void softmax_kernel(float* __restrict__ data, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float* row_data = data + row * cols;

    // Find max
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_data[j]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    // Exp and sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = expf(row_data[j] - row_max);
        row_data[j] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    // Normalize
    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_data[j] *= inv_sum;
}

void softmax_inplace(float* data, int rows, int cols, cudaStream_t stream) {
    int threads = min(cols, 256);
    // Round up to power of 2 for reduction
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    softmax_kernel<<<rows, threads, threads * sizeof(float), stream>>>(data, cols);
}

__global__ void masked_softmax_kernel(float* __restrict__ data,
                                      const bool* __restrict__ mask,
                                      int cols, int mask_rows) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    float* row_data = data + row * cols;
    const bool* row_mask = mask + (row % mask_rows) * cols;

    // Apply mask and find max
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        if (!row_mask[j]) row_data[j] = -FLT_MAX;
        local_max = fmaxf(local_max, row_data[j]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        float v = expf(row_data[j] - row_max);
        row_data[j] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int j = threadIdx.x; j < cols; j += blockDim.x)
        row_data[j] *= inv_sum;
}

void masked_softmax_inplace(float* data, const bool* mask,
                            int rows, int cols, int mask_rows,
                            cudaStream_t stream) {
    int threads = min(cols, 256);
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    masked_softmax_kernel<<<rows, threads, threads * sizeof(float), stream>>>(
        data, mask, cols, mask_rows);
}

__global__ void swin_masked_softmax_kernel(float* __restrict__ data,
                                            const bool* __restrict__ mask,
                                            int num_heads, int ws2, int nW) {
    extern __shared__ float sdata[];
    // blockIdx.x = score row index within [BnW * H * ws2] total rows
    int row = blockIdx.x;
    float* row_data = data + row * ws2;

    // Decode: row = (bh * ws2 + sq), bh = b*nW*H + w*H + h
    int sq = row % ws2;
    int bh = row / ws2;
    int w = (bh / num_heads) % nW;
    const bool* row_mask = mask + (w * ws2 + sq) * ws2;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < ws2; j += blockDim.x) {
        if (!row_mask[j]) row_data[j] = -FLT_MAX;
        local_max = fmaxf(local_max, row_data[j]);
    }
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < ws2; j += blockDim.x) {
        float v = expf(row_data[j] - row_max);
        row_data[j] = v;
        local_sum += v;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int j = threadIdx.x; j < ws2; j += blockDim.x)
        row_data[j] *= inv_sum;
}

void swin_masked_softmax_inplace(float* data, const bool* mask,
                                  int BnW, int num_heads, int ws2, int nW,
                                  cudaStream_t stream) {
    int total_rows = BnW * num_heads * ws2;
    int threads = min(ws2, 256);
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    swin_masked_softmax_kernel<<<total_rows, threads, threads * sizeof(float), stream>>>(
        data, mask, num_heads, ws2, nW);
}

// Warp-level online softmax (max, sum) reduction via shuffle
__device__ __forceinline__ void warp_reduce_online(float& m, float& l) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float mb = __shfl_xor_sync(0xFFFFFFFF, m, offset);
        float lb = __shfl_xor_sync(0xFFFFFFFF, l, offset);
        float mc = fmaxf(m, mb);
        l = l * __expf(m - mc) + lb * __expf(mb - mc);
        m = mc;
    }
}

// 2-pass online softmax (fp32) → fp16 with vectorized loads/stores
// Pass 1: online max+sum via float4 loads + warp shuffle reduction
// Pass 2: normalize + cast to fp16 via float4 loads + half2 stores
__global__ void softmax_cast_fp16_kernel(const float* __restrict__ in,
                                          half* __restrict__ out,
                                          int cols, int in_stride, int out_stride) {
    extern __shared__ float sdata[];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* ri = in + (long long)row * in_stride;
    half* ro = out + (long long)row * out_stride;

    // Pass 1: online (max, sum) with float4 vectorized loads
    float m = -FLT_MAX, l = 0.0f;
    int cols4 = cols & ~3;
    for (int j = tid * 4; j < cols4; j += blockDim.x * 4) {
        float4 v = *reinterpret_cast<const float4*>(ri + j);
        #pragma unroll
        for (float x : {v.x, v.y, v.z, v.w}) {
            if (x > m) { l = l * __expf(m - x) + 1.0f; m = x; }
            else l += __expf(x - m);
        }
    }
    for (int j = cols4 + tid; j < cols; j += blockDim.x) {
        float x = ri[j];
        if (x > m) { l = l * __expf(m - x) + 1.0f; m = x; }
        else l += __expf(x - m);
    }

    // Warp shuffle reduction, then cross-warp via shared memory
    warp_reduce_online(m, l);
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) { sdata[2 * wid] = m; sdata[2 * wid + 1] = l; }
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        m = (lane < nw) ? sdata[2 * lane] : -FLT_MAX;
        l = (lane < nw) ? sdata[2 * lane + 1] : 0.0f;
        warp_reduce_online(m, l);
        if (lane == 0) { sdata[0] = m; sdata[1] = l; }
    }
    __syncthreads();
    float row_max = sdata[0], inv_sum = 1.0f / sdata[1];

    // Pass 2: normalize + cast with float4 reads + half2 writes
    for (int j = tid * 4; j < cols4; j += blockDim.x * 4) {
        float4 v = *reinterpret_cast<const float4*>(ri + j);
        __half2 h01 = __floats2half2_rn(__expf(v.x - row_max) * inv_sum,
                                         __expf(v.y - row_max) * inv_sum);
        __half2 h23 = __floats2half2_rn(__expf(v.z - row_max) * inv_sum,
                                         __expf(v.w - row_max) * inv_sum);
        *reinterpret_cast<__half2*>(ro + j) = h01;
        *reinterpret_cast<__half2*>(ro + j + 2) = h23;
    }
    for (int j = cols4 + tid; j < cols; j += blockDim.x)
        ro[j] = __float2half(__expf(ri[j] - row_max) * inv_sum);
    for (int j = cols + tid; j < out_stride; j += blockDim.x)
        ro[j] = __ushort_as_half(0);
}

void softmax_cast_fp16(const float* in, half* out,
                       int rows, int cols, int in_stride, int out_stride,
                       cudaStream_t stream) {
    int threads = min(cols, 256);
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int nwarps = threads >> 5;
    softmax_cast_fp16_kernel<<<rows, threads, 2 * nwarps * sizeof(float), stream>>>(
        in, out, cols, in_stride, out_stride);
}

__global__ void softmax_cast_fp16_masked_kernel(const float* __restrict__ in,
                                                 half* __restrict__ out,
                                                 const bool* __restrict__ mask,
                                                 int cols, int in_stride, int out_stride,
                                                 int mask_rows) {
    extern __shared__ float sdata[];
    int row = blockIdx.x, tid = threadIdx.x;
    const float* ri = in + (long long)row * in_stride;
    half* ro = out + (long long)row * out_stride;
    const bool* rm = mask + (row % mask_rows) * cols;

    float m = -FLT_MAX, l = 0.0f;
    for (int j = tid; j < cols; j += blockDim.x) {
        if (rm[j]) {
            float x = ri[j];
            if (x > m) { l = l * __expf(m - x) + 1.0f; m = x; }
            else l += __expf(x - m);
        }
    }

    warp_reduce_online(m, l);
    int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) { sdata[2 * wid] = m; sdata[2 * wid + 1] = l; }
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        m = (lane < nw) ? sdata[2 * lane] : -FLT_MAX;
        l = (lane < nw) ? sdata[2 * lane + 1] : 0.0f;
        warp_reduce_online(m, l);
        if (lane == 0) { sdata[0] = m; sdata[1] = l; }
    }
    __syncthreads();
    float row_max = sdata[0], inv_sum = 1.0f / sdata[1];

    for (int j = tid; j < cols; j += blockDim.x)
        ro[j] = __float2half(rm[j] ? __expf(ri[j] - row_max) * inv_sum : 0.0f);
    for (int j = cols + tid; j < out_stride; j += blockDim.x)
        ro[j] = __ushort_as_half(0);
}

void softmax_cast_fp16_masked(const float* in, half* out, const bool* mask,
                              int rows, int cols, int in_stride, int out_stride,
                              int mask_rows, cudaStream_t stream) {
    int threads = min(cols, 256);
    int t = 1;
    while (t < threads) t <<= 1;
    threads = t;
    int nwarps = threads >> 5;
    softmax_cast_fp16_masked_kernel<<<rows, threads, 2 * nwarps * sizeof(float), stream>>>(
        in, out, mask, cols, in_stride, out_stride, mask_rows);
}

} // namespace rf
