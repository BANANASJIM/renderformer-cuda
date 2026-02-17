#pragma once
#include <cuda_fp16.h>
#include <cublasLt.h>

namespace rf {

// Y[M,N] = X[M,K] @ W[N,K]^T + bias[N]  (bias optional)
// All pointers are fp16, accumulation in fp32
void linear(cublasHandle_t handle,
            const half* X, const half* W, const half* bias,
            half* Y, int M, int N, int K,
            cudaStream_t stream = nullptr);

// Add bias[N] to each row of Y[M,N] in-place
void add_bias(half* Y, const half* bias, int M, int N,
              cudaStream_t stream = nullptr);

// fp32 GEMM: Y[M,N] = X_f32[M,K] @ W_f16[N,K]^T + bias_f16[N]
// W is cast fp16→fp32 on-the-fly using w_workspace[N*K]
void linear_f32(cublasHandle_t handle,
                const float* X, const half* W, const half* bias,
                float* Y, float* w_workspace,
                int M, int N, int K,
                cudaStream_t stream = nullptr);

} // namespace rf
