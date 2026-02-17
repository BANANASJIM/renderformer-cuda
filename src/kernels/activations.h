#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

// SiLU(a) ⊙ b — fused element-wise after w1/w3 GEMMs
void fused_silu_mul(const half* a, const half* b, half* out, int n,
                    cudaStream_t stream = nullptr);

// fp32 variant
void fused_silu_mul_f32(const float* a, const float* b, float* out, int n,
                        cudaStream_t stream = nullptr);

// ELU(x, alpha=1e-3) element-wise
void elu(const half* x, half* y, int n, float alpha = 1e-3f,
         cudaStream_t stream = nullptr);

// 10^x - 1 (log decode for HDR output)
void pow10_minus1(const float* x, float* y, int n,
                  cudaStream_t stream = nullptr);

} // namespace rf
