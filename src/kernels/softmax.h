#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

// Row-wise softmax: out[i,j] = exp(x[i,j] - max_j) / sum_j(exp(x[i,j] - max_j))
// Operates on fp32 in-place. rows × cols matrix.
void softmax_inplace(float* data, int rows, int cols,
                     cudaStream_t stream = nullptr);

// Masked softmax: applies -inf to positions where mask[row % mask_rows][col] == false
void masked_softmax_inplace(float* data, const bool* mask,
                            int rows, int cols, int mask_rows,
                            cudaStream_t stream = nullptr);

// 2D masked softmax for swin attention:
// scores: [BnW * H, ws2, ws2] where BnW = batch * num_windows
// mask: [nW, ws2, ws2] — per-window per-query-key mask
// Maps score row (bh, sq) to mask[(bh / num_heads) % nW][sq][sk]
void swin_masked_softmax_inplace(float* data, const bool* mask,
                                  int BnW, int num_heads, int ws2, int nW,
                                  cudaStream_t stream = nullptr);

// Fused softmax (fp32 input) → fp16 output with strided access.
// in[row * in_stride .. +cols], out[row * out_stride .. +cols]
void softmax_cast_fp16(const float* in, half* out,
                       int rows, int cols, int in_stride, int out_stride,
                       cudaStream_t stream = nullptr);

void softmax_cast_fp16_masked(const float* in, half* out, const bool* mask,
                              int rows, int cols, int in_stride, int out_stride,
                              int mask_rows, cudaStream_t stream = nullptr);

} // namespace rf
