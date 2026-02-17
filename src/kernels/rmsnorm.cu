#include "kernels/rmsnorm.h"
#include <cuda_fp16.h>

namespace rf {

// RMSNorm: y = x * rsqrt(mean(x²) + eps) * gamma
// Block per row, warp-level reduction
__global__ void rmsnorm_kernel(const half* __restrict__ x,
                               const float* __restrict__ gamma,
                               half* __restrict__ y,
                               int dim, float eps) {
    int row = blockIdx.x;
    const half* x_row = x + row * dim;
    half* y_row = y + row * dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    // Cross-warp reduction via shared memory
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(sum_sq / dim + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x_row[i]);
        y_row[i] = __float2half(v * rms_inv * gamma[i]);
    }
}

void rmsnorm(const half* x, const float* gamma, half* y,
             int rows, int dim, float eps, cudaStream_t stream) {
    int threads = std::min(dim, 256);
    rmsnorm_kernel<<<rows, threads, 0, stream>>>(x, gamma, y, dim, eps);
}

__global__ void rmsnorm_f32_kernel(const float* __restrict__ x,
                                    const float* __restrict__ gamma,
                                    float* __restrict__ y,
                                    int dim, float eps) {
    int row = blockIdx.x;
    const float* x_row = x + row * dim;
    float* y_row = y + row * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        sum_sq += x_row[i] * x_row[i];

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) shared[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        sum_sq = (lane < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) s_rms_inv = rsqrtf(sum_sq / dim + eps);
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y_row[i] = x_row[i] * rms_inv * gamma[i];
}

void rmsnorm_f32(const float* x, const float* gamma, float* y,
                 int rows, int dim, float eps, cudaStream_t stream) {
    rmsnorm_f32_kernel<<<rows, std::min(dim, 256), 0, stream>>>(x, gamma, y, dim, eps);
}

} // namespace rf
