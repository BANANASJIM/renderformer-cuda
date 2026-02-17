#pragma once
#include "layers/attention.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace rf {

// SwiGLU FFN weights: out = W2(SiLU(W1·x) ⊙ W3·x)
struct FFNWeights {
    const half* w1; // [hidden_dim, dim]
    const half* w2; // [dim, hidden_dim]
    const half* w3; // [hidden_dim, dim]
};

// Encoder attention layer weights
struct EncoderLayerWeights {
    const float* query_norm_gamma;  // [dim]
    MHAWeights attn;
    const float* ffn_norm_gamma;    // [dim]
    FFNWeights ffn;
};

// Decoder attention layer weights
struct DecoderLayerWeights {
    const float* query_norm_gamma;  // [dim]
    const float* kv_norm_gamma;     // [ctx_dim]
    CrossAttnWeights cross_attn;
    const float* self_attn_norm_gamma; // [dim]
    SwinAttnWeights self_attn;
    const float* ffn_norm_gamma;    // [dim]
    FFNWeights ffn;
};

// Forward one encoder layer: norm → self_attn → add → norm → ffn → add
// x: [B, S, dim] in/out (modified in-place via residual)
// temp: workspace buffer [B, S, dim]
// qkv_buf: [B*S, 3*dim], score_buf: [B*H*S*S] fp32
void encoder_layer_forward(cublasHandle_t handle,
                           const EncoderLayerWeights& w,
                           half* x, half* temp, half* qkv_buf, half* ffn_buf,
                           float* score_buf,
                           const bool* mask,
                           int B, int S, int dim, int num_heads, int ffn_hidden,
                           const half* rope_cos, const half* rope_sin,
                           cudaStream_t stream = nullptr);

// Mixed-precision encoder layer: fp32 hidden state, fp16 sub-layers (matches autocast(fp16))
// x is fp32, pre-norms computed in fp32, cast to fp16 for attention/FFN, residual in fp32
void encoder_layer_forward_mixed(cublasHandle_t handle,
                                  const EncoderLayerWeights& w,
                                  float* x, half* temp, half* qkv_buf, half* ffn_buf,
                                  float* score_buf,
                                  const bool* mask,
                                  int B, int S, int dim, int num_heads, int ffn_hidden,
                                  const half* rope_cos, const half* rope_sin,
                                  cudaStream_t stream = nullptr);

// fp32 encoder layer
void encoder_layer_forward_f32(cublasHandle_t handle,
                               const EncoderLayerWeights& w,
                               float* x, float* temp, float* qkv_buf, float* ffn_buf,
                               float* score_buf, float* w_workspace,
                               const bool* mask,
                               int B, int S, int dim, int num_heads, int ffn_hidden,
                               const half* rope_cos, const half* rope_sin,
                               cudaStream_t stream = nullptr);

// Forward one decoder layer: norm → cross_attn → add → norm → swin_self_attn → add → norm → ffn → add
void decoder_layer_forward(cublasHandle_t handle,
                           const DecoderLayerWeights& w,
                           half* x, half* temp,
                           half* proj_buf, half* qkv_buf, half* ffn_buf,
                           half* window_buf,
                           float* cross_score_buf, float* self_score_buf,
                           const half* ctx, const bool* ctx_mask,
                           int B, int Sq, int Sk, int dim, int ctx_dim,
                           int num_heads, int ffn_hidden,
                           int patch_h, int patch_w,
                           int window_size, int shift_size,
                           const bool* shift_mask,
                           const half* q_rope_cos, const half* q_rope_sin,
                           const half* k_rope_cos, const half* k_rope_sin,
                           cudaStream_t stream = nullptr);

// Decoder layer using cached KV (skips ctx norm + K/V projection)
void decoder_layer_forward_cached(cublasHandle_t handle,
                                   const DecoderLayerWeights& w,
                                   half* x, half* temp,
                                   half* proj_buf, half* qkv_buf, half* ffn_buf,
                                   half* window_buf,
                                   float* cross_score_buf, float* self_score_buf,
                                   const half* K_cached, const half* V_cached,
                                   const bool* ctx_mask,
                                   int B, int Sq, int Sk, int dim,
                                   int num_heads, int ffn_hidden,
                                   int patch_h, int patch_w,
                                   int window_size, int shift_size,
                                   const bool* shift_mask,
                                   const half* q_rope_cos, const half* q_rope_sin,
                                   const half* k_rope_cos, const half* k_rope_sin,
                                   cudaStream_t stream = nullptr);

// Element-wise: y = x + alpha (residual add)
void residual_add(half* y, const half* residual, int n,
                  cudaStream_t stream = nullptr);

// fp32 residual += fp32 sub-layer output
void residual_add_f32(float* y, const float* residual, int n,
                      cudaStream_t stream = nullptr);

// fp32→fp16 / fp16→fp32 cast
void cast_fp32_to_fp16(const float* in, half* out, int n, cudaStream_t stream = nullptr);
void cast_fp16_to_fp32(const half* in, float* out, int n, cudaStream_t stream = nullptr);

// Mixed residual: fp32 += fp16 (matches autocast residual promotion)
void residual_add_mixed(float* y, const half* residual, int n, cudaStream_t stream = nullptr);

// SwiGLU FFN forward
void ffn_swiglu(cublasHandle_t handle, const FFNWeights& w,
                const half* x, half* out, half* buf,
                int M, int dim, int hidden_dim,
                cudaStream_t stream = nullptr);

// fp32 SwiGLU FFN
void ffn_swiglu_f32(cublasHandle_t handle, const FFNWeights& w,
                    const float* x, float* out, float* buf, float* w_workspace,
                    int M, int dim, int hidden_dim,
                    cudaStream_t stream = nullptr);

// Mixed-precision decoder: fp32 hidden state, fp16 sub-layers (tensor cores)
void decoder_layer_forward_mixed(cublasHandle_t handle,
                                  const DecoderLayerWeights& w,
                                  float* x_f32,
                                  half* temp, half* proj_buf, half* qkv_buf, half* ffn_buf,
                                  half* window_buf,
                                  float* cross_score_buf, float* self_score_buf,
                                  const half* ctx, const bool* ctx_mask,
                                  int B, int Sq, int Sk, int dim, int ctx_dim,
                                  int num_heads, int ffn_hidden,
                                  int patch_h, int patch_w,
                                  int window_size, int shift_size,
                                  const bool* shift_mask,
                                  const half* q_rope_cos, const half* q_rope_sin,
                                  const half* k_rope_cos, const half* k_rope_sin,
                                  cudaStream_t stream = nullptr);

// Decoder layer with full fp32 computation (matches Python's autocast(fp32))
void decoder_layer_forward_f32(cublasHandle_t handle,
                               const DecoderLayerWeights& w,
                               float* x_f32,
                               float* temp_f32, float* proj_buf_f32,
                               float* qkv_buf_f32, float* ffn_buf_f32,
                               float* window_buf_f32,
                               float* cross_score_buf, float* self_score_buf,
                               float* w_workspace,
                               const float* ctx_f32, const bool* ctx_mask,
                               int B, int Sq, int Sk, int dim, int ctx_dim,
                               int num_heads, int ffn_hidden,
                               int patch_h, int patch_w,
                               int window_size, int shift_size,
                               const bool* shift_mask,
                               const half* q_rope_cos, const half* q_rope_sin,
                               const half* k_rope_cos, const half* k_rope_sin,
                               cudaStream_t stream = nullptr);

} // namespace rf
