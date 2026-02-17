#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace rf {

// Scaled dot-product attention (naive, no FlashAttention)
// Q[B*H, Sq, D], K[B*H, Sk, D], V[B*H, Sk, D] → O[B*H, Sq, D]
// mask: [B, Sk] bool (true = attend), broadcast across heads. nullptr = no mask.
// workspace: fp32 buffer of size B*H*Sq*Sk for attention scores
void sdpa(cublasHandle_t handle,
          const half* Q, const half* K, const half* V, half* O,
          float* workspace,
          const bool* mask, int B, int H, int Sq, int Sk, int D,
          cudaStream_t stream = nullptr);

// Multi-head self-attention (in_proj QKV packed)
// x[M, dim] → out[M, dim]
// Weights: in_proj_w[3*dim, dim], out_proj_w[dim, dim]
// QK norm weights: q_norm_gamma[dim], k_norm_gamma[dim] (nullptr to skip)
struct MHAWeights {
    const half* in_proj_w;     // [3*dim, dim]
    const half* in_proj_b;     // [3*dim] or nullptr
    const half* out_proj_w;    // [dim, dim]
    const half* out_proj_b;    // [dim] or nullptr
    const float* q_norm_gamma; // [dim] or nullptr
    const float* k_norm_gamma; // [dim] or nullptr
};

void self_attention(cublasHandle_t handle, const MHAWeights& w,
                    const half* x, half* out, half* qkv_buf, float* score_buf,
                    const bool* mask, int B, int S, int dim, int num_heads,
                    const half* rope_cos, const half* rope_sin,
                    cudaStream_t stream = nullptr);

// Multi-head cross-attention (separate Q, K, V projections)
struct CrossAttnWeights {
    const half* q_proj_w;      // [dim, q_dim]
    const half* q_proj_b;      // [dim] or nullptr
    const half* k_proj_w;      // [dim, kv_dim]
    const half* k_proj_b;      // [dim] or nullptr
    const half* v_proj_w;      // [dim, kv_dim]
    const half* v_proj_b;      // [dim] or nullptr
    const half* out_proj_w;    // [q_dim, dim]
    const half* out_proj_b;    // [q_dim] or nullptr
    const float* q_norm_gamma; // [dim] or nullptr
    const float* k_norm_gamma; // [dim] or nullptr
};

void cross_attention(cublasHandle_t handle, const CrossAttnWeights& w,
                     const half* q_in, const half* kv_in,
                     half* out, half* proj_buf, float* score_buf,
                     const bool* mask,
                     int B, int Sq, int Sk, int q_dim, int kv_dim, int dim, int num_heads,
                     const half* q_rope_cos, const half* q_rope_sin,
                     const half* k_rope_cos, const half* k_rope_sin,
                     cudaStream_t stream = nullptr);

// KV cache: precompute K_bhsd and V_bhsd from context (view-independent)
// ctx_normed: [B*Sk, kv_dim] — already normalized context
// K_bhsd_out, V_bhsd_out: [B, num_heads, Sk, head_dim] — preallocated output
// proj_buf: workspace [B*Sk * dim] for intermediate projection
void precompute_cross_kv(cublasHandle_t handle, const CrossAttnWeights& w,
                         const half* ctx_normed, half* K_bhsd_out, half* V_bhsd_out,
                         half* proj_buf,
                         int B, int Sk, int kv_dim, int dim, int num_heads,
                         cudaStream_t stream = nullptr);

// Cross-attention using cached K_bhsd/V_bhsd (applies RoPE to K copy per view)
// If Sk_pad > 0, K/V are pre-padded to [B,H,Sk_pad,D] and sdpa skips padding.
void cross_attention_cached(cublasHandle_t handle, const CrossAttnWeights& w,
                            const half* q_in, half* out,
                            half* proj_buf, float* score_buf,
                            const half* K_cached, const half* V_cached,
                            const bool* mask,
                            int B, int Sq, int Sk, int q_dim, int dim, int num_heads,
                            const half* q_rope_cos, const half* q_rope_sin,
                            const half* k_rope_cos, const half* k_rope_sin,
                            int Sk_pad = 0,
                            cudaStream_t stream = nullptr);

// Swin window self-attention
struct SwinAttnWeights {
    const half* in_proj_w;     // [3*dim, dim]
    const half* in_proj_b;     // [3*dim] or nullptr
    const half* out_proj_w;    // [dim, dim]
    const half* out_proj_b;    // [dim] or nullptr
    const float* q_norm_gamma; // [dim] or nullptr
    const float* k_norm_gamma; // [dim] or nullptr
};

// scratch: preallocated buffer for Q/K/V/P_half/shifted
// min size: (3*B*nW*ws^2*dim + B*nW*H_heads*ws^4) half elements
void swin_self_attention(cublasHandle_t handle, const SwinAttnWeights& w,
                         const half* x, half* out,
                         half* window_buf, half* qkv_buf, float* score_buf,
                         half* scratch,
                         int B, int H, int W, int dim, int num_heads,
                         int window_size, int shift_size,
                         const bool* shift_mask,
                         cudaStream_t stream = nullptr);

// ---- fp32 variants ----

void self_attention_f32(cublasHandle_t handle, const MHAWeights& w,
                        const float* x, float* out, float* qkv_buf, float* score_buf,
                        float* w_workspace,
                        const bool* mask, int B, int S, int dim, int num_heads,
                        const half* rope_cos, const half* rope_sin,
                        cudaStream_t stream = nullptr);

void sdpa_f32(cublasHandle_t handle,
              const float* Q, const float* K, const float* V, float* O,
              float* workspace,
              const bool* mask, int B, int H, int Sq, int Sk, int D,
              cudaStream_t stream = nullptr);

void cross_attention_f32(cublasHandle_t handle, const CrossAttnWeights& w,
                         const float* q_in, const float* kv_in,
                         float* out, float* proj_buf, float* score_buf,
                         float* w_workspace,
                         const bool* mask,
                         int B, int Sq, int Sk, int q_dim, int kv_dim, int dim, int num_heads,
                         const half* q_rope_cos, const half* q_rope_sin,
                         const half* k_rope_cos, const half* k_rope_sin,
                         cudaStream_t stream = nullptr);

void swin_self_attention_f32(cublasHandle_t handle, const SwinAttnWeights& w,
                              const float* x, float* out,
                              float* window_buf, float* qkv_buf, float* score_buf,
                              float* w_workspace,
                              int B, int H, int W, int dim, int num_heads,
                              int window_size, int shift_size,
                              const bool* shift_mask,
                              cudaStream_t stream = nullptr);

// Window partition/reverse kernels
void window_partition(const half* x, half* windows,
                      int B, int H, int W, int C, int window_size,
                      cudaStream_t stream = nullptr);

void window_reverse(const half* windows, half* x,
                    int B, int H, int W, int C, int window_size,
                    cudaStream_t stream = nullptr);

// Cyclic shift (roll) for swin attention
void cyclic_shift_2d(const half* x, half* out,
                     int B, int H, int W, int C,
                     int shift_h, int shift_w,
                     cudaStream_t stream = nullptr);

} // namespace rf
