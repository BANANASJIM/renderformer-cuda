#include "kernels/activations.h"

namespace rf {

__global__ void fused_silu_mul_kernel(const half* __restrict__ a,
                                      const half* __restrict__ b,
                                      half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float va = __half2float(a[i]);
    float vb = __half2float(b[i]);
    float silu = va / (1.0f + expf(-va)); // SiLU = x * sigmoid(x)
    out[i] = __float2half(silu * vb);
}

void fused_silu_mul(const half* a, const half* b, half* out, int n,
                    cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_silu_mul_kernel<<<blocks, threads, 0, stream>>>(a, b, out, n);
}

__global__ void fused_silu_mul_f32_kernel(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float silu = a[i] / (1.0f + expf(-a[i]));
    out[i] = silu * b[i];
}

void fused_silu_mul_f32(const float* a, const float* b, float* out, int n,
                        cudaStream_t stream) {
    int t = 256;
    fused_silu_mul_f32_kernel<<<(n + t - 1) / t, t, 0, stream>>>(a, b, out, n);
}

__global__ void elu_kernel(const half* __restrict__ x, half* __restrict__ y,
                           int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    y[i] = __float2half(v > 0.0f ? v : alpha * (expf(v) - 1.0f));
}

void elu(const half* x, half* y, int n, float alpha, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    elu_kernel<<<blocks, threads, 0, stream>>>(x, y, n, alpha);
}

__global__ void pow10_minus1_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = powf(10.0f, x[i]) - 1.0f;
}

void pow10_minus1(const float* x, float* y, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    pow10_minus1_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

} // namespace rf
