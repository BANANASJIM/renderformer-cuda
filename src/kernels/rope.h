#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

// Compute triangle RoPE frequencies from positions and learned freqs.
// pos: [B, S, 9] (3 vertices × 3 coords, fp16)
// learned_freqs: [num_freqs] (fp32, from model weights)
// out_cos, out_sin: [B, 1, S, head_dim] (fp16)
// hf_format: freqs → cat([freqs, freqs]) then cos/sin, with zero-padding to head_dim
void triangle_rope_freqs(const half* pos, const float* learned_freqs,
                         half* out_cos, half* out_sin,
                         int batch, int seq_len, int num_freqs, int head_dim,
                         cudaStream_t stream = nullptr);

// Apply RoPE in-place: q = q*cos + rotate_half_hf(q)*sin (same for k)
// q, k: [B, H, S, D] fp16
// cos, sin: [B, 1, S, D] fp16
void apply_rope(half* q, half* k, const half* cos_vals, const half* sin_vals,
                int batch, int num_heads, int seq_len, int head_dim,
                cudaStream_t stream = nullptr);

// Apply RoPE to single tensor (for cross-attn where Q and K have different pos)
void apply_rope_one(half* x, const half* cos_vals, const half* sin_vals,
                    int batch, int num_heads, int seq_len, int head_dim,
                    cudaStream_t stream = nullptr);

// fp32 variants: operate on fp32 Q/K with fp16 cos/sin
void apply_rope_f32(float* q, float* k, const half* cos_vals, const half* sin_vals,
                    int batch, int num_heads, int seq_len, int head_dim,
                    cudaStream_t stream = nullptr);

void apply_rope_one_f32(float* x, const half* cos_vals, const half* sin_vals,
                        int batch, int num_heads, int seq_len, int head_dim,
                        cudaStream_t stream = nullptr);

} // namespace rf
