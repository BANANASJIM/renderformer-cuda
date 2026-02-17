#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

// NeRF positional encoding:
// out = [x, sin(f0*x), sin(f1*x), ..., cos(f0*x), cos(f1*x), ...]
// in_dim=9, num_freqs=12, include_input=true → out_dim=225
// in_dim=9, num_freqs=6, include_input=true → out_dim=117
void nerf_pe(const half* input, half* output,
             int batch, int seq_len, int in_dim, int num_freqs,
             float min_freq_exp, float max_freq_exp,
             bool include_input, cudaStream_t stream = nullptr);

// fp32 input variant (better precision for high-frequency components)
void nerf_pe_f32(const float* input, half* output,
                 int batch, int seq_len, int in_dim, int num_freqs,
                 float min_freq_exp, float max_freq_exp,
                 bool include_input, cudaStream_t stream = nullptr);

} // namespace rf
