#include "layers/linear.h"
#include "core/cuda_utils.h"
#include <cublas_v2.h>

namespace rf {

void linear(cublasHandle_t handle,
            const half* X, const half* W, const half* bias,
            half* Y, int M, int N, int K,
            cudaStream_t stream) {
    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    // Row-major: Y[M,N] = X[M,K] @ W[N,K]^T
    // cuBLAS col-major trick: C[N,M] = W_col^T[N,K] @ X_col[K,M]
    CHECK_CUBLAS(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, CUDA_R_16F, K,
        X, CUDA_R_16F, K,
        &beta,
        Y, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    if (bias) add_bias(Y, bias, M, N, stream);
}

__global__ void add_bias_kernel(half* __restrict__ Y, const half* __restrict__ bias,
                                int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int col = idx % N;
    float v = __half2float(Y[idx]) + __half2float(bias[col]);
    Y[idx] = __float2half(v);
}

void add_bias(half* Y, const half* bias, int M, int N, cudaStream_t stream) {
    int total = M * N;
    int threads = 256;
    add_bias_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
        Y, bias, M, N);
}

__global__ void cast_half_to_float_kernel(const half* __restrict__ in,
                                           float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

__global__ void add_bias_f32_kernel(float* __restrict__ Y, const half* __restrict__ bias,
                                     int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    Y[idx] += __half2float(bias[idx % N]);
}

void linear_f32(cublasHandle_t handle,
                const float* X, const half* W, const half* bias,
                float* Y, float* w_workspace,
                int M, int N, int K,
                cudaStream_t stream) {
    // Cast fp16 weight → fp32
    int wn = N * K, t = 256;
    cast_half_to_float_kernel<<<(wn + t - 1) / t, t, 0, stream>>>(W, w_workspace, wn);

    cublasSetStream(handle, stream);
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        w_workspace, CUDA_R_32F, K,
        X, CUDA_R_32F, K,
        &beta,
        Y, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    if (bias) {
        int total = M * N;
        add_bias_f32_kernel<<<(total + t - 1) / t, t, 0, stream>>>(Y, bias, M, N);
    }
}

} // namespace rf
