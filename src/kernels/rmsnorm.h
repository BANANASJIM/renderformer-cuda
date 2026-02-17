#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace rf {

// Default eps matching PyTorch: torch.finfo(dtype).eps
constexpr float kEpsFp16 = 0.0009765625f; // 2^-10, torch.finfo(float16).eps
constexpr float kEpsFp32 = 1.1920929e-07f; // 2^-23, torch.finfo(float32).eps

// y[rows, dim] = rmsnorm(x[rows, dim], gamma[dim])
void rmsnorm(const half* x, const float* gamma, half* y,
             int rows, int dim, float eps = kEpsFp16,
             cudaStream_t stream = nullptr);

// fp32 variant: same computation, fp32 in/out
void rmsnorm_f32(const float* x, const float* gamma, float* y,
                 int rows, int dim, float eps = kEpsFp32,
                 cudaStream_t stream = nullptr);

} // namespace rf
