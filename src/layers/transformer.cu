#include "layers/transformer.h"
#include "layers/linear.h"
#include "kernels/rmsnorm.h"
#include "kernels/activations.h"
#include "core/cuda_utils.h"

namespace rf {

__global__ void residual_add_kernel(half* __restrict__ y,
                                     const half* __restrict__ r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half(__half2float(y[i]) + __half2float(r[i]));
}

void residual_add(half* y, const half* residual, int n, cudaStream_t stream) {
    int threads = 256;
    residual_add_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(y, residual, n);
}

__global__ void residual_add_f32_kernel(float* __restrict__ y,
                                         const float* __restrict__ r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += r[i];
}

void residual_add_f32(float* y, const float* residual, int n, cudaStream_t stream) {
    int t = 256;
    residual_add_f32_kernel<<<(n + t - 1) / t, t, 0, stream>>>(y, residual, n);
}

__global__ void cast_f32_to_f16_kernel(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ void cast_f16_to_f32_kernel(const half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

void cast_fp32_to_fp16(const float* in, half* out, int n, cudaStream_t stream) {
    int t = 256;
    cast_f32_to_f16_kernel<<<(n + t - 1) / t, t, 0, stream>>>(in, out, n);
}

void cast_fp16_to_fp32(const half* in, float* out, int n, cudaStream_t stream) {
    int t = 256;
    cast_f16_to_f32_kernel<<<(n + t - 1) / t, t, 0, stream>>>(in, out, n);
}

__global__ void residual_add_mixed_kernel(float* __restrict__ y,
                                           const half* __restrict__ r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += __half2float(r[i]);
}

void residual_add_mixed(float* y, const half* residual, int n, cudaStream_t stream) {
    int t = 256;
    residual_add_mixed_kernel<<<(n + t - 1) / t, t, 0, stream>>>(y, residual, n);
}

void ffn_swiglu(cublasHandle_t handle, const FFNWeights& w,
                const half* x, half* out, half* buf,
                int M, int dim, int hidden_dim,
                cudaStream_t stream) {
    // buf layout: w1_out[M, hidden], w3_out[M, hidden]
    half* w1_out = buf;
    half* w3_out = buf + M * hidden_dim;

    linear(handle, x, w.w1, nullptr, w1_out, M, hidden_dim, dim, stream);
    linear(handle, x, w.w3, nullptr, w3_out, M, hidden_dim, dim, stream);

    // fused SiLU(w1_out) ⊙ w3_out
    fused_silu_mul(w1_out, w3_out, w1_out, M * hidden_dim, stream);

    // w2: [dim, hidden] → out = w1_out @ w2^T
    linear(handle, w1_out, w.w2, nullptr, out, M, dim, hidden_dim, stream);
}

void encoder_layer_forward(cublasHandle_t handle,
                           const EncoderLayerWeights& w,
                           half* x, half* temp, half* qkv_buf, half* ffn_buf,
                           float* score_buf,
                           const bool* mask,
                           int B, int S, int dim, int num_heads, int ffn_hidden,
                           const half* rope_cos, const half* rope_sin,
                           cudaStream_t stream) {
    int M = B * S;
    int n = M * dim;

    // Pre-norm → self-attention → residual
    rmsnorm(x, w.query_norm_gamma, temp, M, dim, kEpsFp16, stream);
    self_attention(handle, w.attn, temp, ffn_buf, qkv_buf, score_buf,
                   mask, B, S, dim, num_heads, rope_cos, rope_sin, stream);
    residual_add(x, ffn_buf, n, stream);

    // Pre-norm → FFN → residual
    rmsnorm(x, w.ffn_norm_gamma, temp, M, dim, kEpsFp16, stream);
    ffn_swiglu(handle, w.ffn, temp, ffn_buf, qkv_buf, M, dim, ffn_hidden, stream);
    residual_add(x, ffn_buf, n, stream);
}

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
                           cudaStream_t stream) {
    int Mq = B * Sq;
    int n = Mq * dim;

    // 1) Pre-norm → cross-attention → residual
    rmsnorm(x, w.query_norm_gamma, temp, Mq, dim, kEpsFp16, stream);

    // Norm context
    half* ctx_normed = ffn_buf; // borrow ffn_buf temporarily [B*Sk, ctx_dim]
    rmsnorm(ctx, w.kv_norm_gamma, ctx_normed, B * Sk, ctx_dim, kEpsFp16, stream);

    half* cross_out = qkv_buf; // [Mq, dim]
    cross_attention(handle, w.cross_attn,
                    temp, ctx_normed, cross_out, proj_buf, cross_score_buf,
                    ctx_mask, B, Sq, Sk, dim, ctx_dim, dim, num_heads,
                    q_rope_cos, q_rope_sin, k_rope_cos, k_rope_sin, stream);
    residual_add(x, cross_out, n, stream);

    // 2) Pre-norm → swin self-attention → residual (proj_buf free, reuse as scratch)
    rmsnorm(x, w.self_attn_norm_gamma, temp, Mq, dim, kEpsFp16, stream);
    swin_self_attention(handle, w.self_attn,
                        temp, ffn_buf, window_buf, qkv_buf, self_score_buf,
                        proj_buf,
                        B, patch_h, patch_w, dim, num_heads,
                        window_size, shift_size, shift_mask, stream);
    residual_add(x, ffn_buf, n, stream);

    // 3) Pre-norm → FFN → residual
    rmsnorm(x, w.ffn_norm_gamma, temp, Mq, dim, kEpsFp16, stream);
    ffn_swiglu(handle, w.ffn, temp, ffn_buf, qkv_buf, Mq, dim, ffn_hidden, stream);
    residual_add(x, ffn_buf, n, stream);
}

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
                                   cudaStream_t stream) {
    int Mq = B * Sq;
    int n = Mq * dim;

    // 1) Pre-norm → cross-attention with cached KV → residual
    rmsnorm(x, w.query_norm_gamma, temp, Mq, dim, kEpsFp16, stream);

    half* cross_out = qkv_buf;
    cross_attention_cached(handle, w.cross_attn,
                           temp, cross_out, proj_buf, cross_score_buf,
                           K_cached, V_cached, ctx_mask,
                           B, Sq, Sk, dim, dim, num_heads,
                           q_rope_cos, q_rope_sin, k_rope_cos, k_rope_sin, 0, stream);
    residual_add(x, cross_out, n, stream);

    // 2) Pre-norm → swin self-attention → residual (proj_buf free, reuse as scratch)
    rmsnorm(x, w.self_attn_norm_gamma, temp, Mq, dim, kEpsFp16, stream);
    swin_self_attention(handle, w.self_attn,
                        temp, ffn_buf, window_buf, qkv_buf, self_score_buf,
                        proj_buf,
                        B, patch_h, patch_w, dim, num_heads,
                        window_size, shift_size, shift_mask, stream);
    residual_add(x, ffn_buf, n, stream);

    // 3) Pre-norm → FFN → residual
    rmsnorm(x, w.ffn_norm_gamma, temp, Mq, dim, kEpsFp16, stream);
    ffn_swiglu(handle, w.ffn, temp, ffn_buf, qkv_buf, Mq, dim, ffn_hidden, stream);
    residual_add(x, ffn_buf, n, stream);
}

void encoder_layer_forward_mixed(cublasHandle_t handle,
                                  const EncoderLayerWeights& w,
                                  float* x, half* temp, half* qkv_buf, half* ffn_buf,
                                  float* score_buf,
                                  const bool* mask,
                                  int B, int S, int dim, int num_heads, int ffn_hidden,
                                  const half* rope_cos, const half* rope_sin,
                                  cudaStream_t stream) {
    int M = B * S;
    int n = M * dim;

    // Reuse qkv_buf as temp fp32 storage for norm output
    // qkv_buf has M*max(3*dim, 2*ffn)*sizeof(half) ≥ M*dim*sizeof(float)
    float* norm_f32 = reinterpret_cast<float*>(qkv_buf);

    // Pre-norm (fp32) → cast to fp16 → self-attention (fp16) → residual (fp32 += fp16)
    rmsnorm_f32(x, w.query_norm_gamma, norm_f32, M, dim, kEpsFp32, stream);
    cast_fp32_to_fp16(norm_f32, temp, n, stream);
    self_attention(handle, w.attn, temp, ffn_buf, qkv_buf, score_buf,
                   mask, B, S, dim, num_heads, rope_cos, rope_sin, stream);
    residual_add_mixed(x, ffn_buf, n, stream);

    // Pre-norm (fp32) → cast to fp16 → FFN (fp16) → residual (fp32 += fp16)
    norm_f32 = reinterpret_cast<float*>(qkv_buf);
    rmsnorm_f32(x, w.ffn_norm_gamma, norm_f32, M, dim, kEpsFp32, stream);
    cast_fp32_to_fp16(norm_f32, temp, n, stream);
    ffn_swiglu(handle, w.ffn, temp, ffn_buf, qkv_buf, M, dim, ffn_hidden, stream);
    residual_add_mixed(x, ffn_buf, n, stream);
}

void encoder_layer_forward_f32(cublasHandle_t handle,
                               const EncoderLayerWeights& w,
                               float* x, float* temp, float* qkv_buf, float* ffn_buf,
                               float* score_buf, float* w_workspace,
                               const bool* mask,
                               int B, int S, int dim, int num_heads, int ffn_hidden,
                               const half* rope_cos, const half* rope_sin,
                               cudaStream_t stream) {
    int M = B * S, n = M * dim;

    // Pre-norm → self-attention → residual
    rmsnorm_f32(x, w.query_norm_gamma, temp, M, dim, kEpsFp32, stream);
    self_attention_f32(handle, w.attn, temp, ffn_buf, qkv_buf, score_buf,
                       w_workspace, mask, B, S, dim, num_heads, rope_cos, rope_sin, stream);
    residual_add_f32(x, ffn_buf, n, stream);

    // Pre-norm → FFN → residual
    rmsnorm_f32(x, w.ffn_norm_gamma, temp, M, dim, kEpsFp32, stream);
    ffn_swiglu_f32(handle, w.ffn, temp, ffn_buf, qkv_buf, w_workspace, M, dim, ffn_hidden, stream);
    residual_add_f32(x, ffn_buf, n, stream);
}

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
                                  cudaStream_t stream) {
    int Mq = B * Sq;
    int n = Mq * dim;

    // Reuse qkv_buf as fp32 norm workspace (sizeof(half)*max(3*dim,2*ffn) ≥ sizeof(float)*dim)
    float* norm_f32 = reinterpret_cast<float*>(qkv_buf);

    // 1) Cross-attention: fp32 norm → fp16 → fp16 cross_attn → fp32 residual
    rmsnorm_f32(x_f32, w.query_norm_gamma, norm_f32, Mq, dim, kEpsFp32, stream);
    cast_fp32_to_fp16(norm_f32, temp, n, stream);

    // Context norm (fp16 input → fp16 output, reuse ffn_buf)
    half* ctx_normed = ffn_buf;
    rmsnorm(ctx, w.kv_norm_gamma, ctx_normed, B * Sk, ctx_dim, kEpsFp16, stream);

    half* cross_out = qkv_buf;
    cross_attention(handle, w.cross_attn,
                    temp, ctx_normed, cross_out, proj_buf, cross_score_buf,
                    ctx_mask, B, Sq, Sk, dim, ctx_dim, dim, num_heads,
                    q_rope_cos, q_rope_sin, k_rope_cos, k_rope_sin, stream);
    residual_add_mixed(x_f32, cross_out, n, stream);

    // 2) Swin self-attention: fp32 norm → fp16 → fp16 swin → fp32 residual
    norm_f32 = reinterpret_cast<float*>(qkv_buf);
    rmsnorm_f32(x_f32, w.self_attn_norm_gamma, norm_f32, Mq, dim, kEpsFp32, stream);
    cast_fp32_to_fp16(norm_f32, temp, n, stream);

    swin_self_attention(handle, w.self_attn,
                        temp, ffn_buf, window_buf, qkv_buf, self_score_buf,
                        proj_buf,
                        B, patch_h, patch_w, dim, num_heads,
                        window_size, shift_size, shift_mask, stream);
    residual_add_mixed(x_f32, ffn_buf, n, stream);

    // 3) FFN: fp32 norm → fp16 → fp16 SwiGLU → fp32 residual
    norm_f32 = reinterpret_cast<float*>(qkv_buf);
    rmsnorm_f32(x_f32, w.ffn_norm_gamma, norm_f32, Mq, dim, kEpsFp32, stream);
    cast_fp32_to_fp16(norm_f32, temp, n, stream);
    ffn_swiglu(handle, w.ffn, temp, ffn_buf, qkv_buf, Mq, dim, ffn_hidden, stream);
    residual_add_mixed(x_f32, ffn_buf, n, stream);
}

void ffn_swiglu_f32(cublasHandle_t handle, const FFNWeights& w,
                    const float* x, float* out, float* buf, float* w_workspace,
                    int M, int dim, int hidden_dim,
                    cudaStream_t stream) {
    float* w1_out = buf;
    float* w3_out = buf + M * hidden_dim;

    linear_f32(handle, x, w.w1, nullptr, w1_out, w_workspace, M, hidden_dim, dim, stream);
    linear_f32(handle, x, w.w3, nullptr, w3_out, w_workspace, M, hidden_dim, dim, stream);
    fused_silu_mul_f32(w1_out, w3_out, w1_out, M * hidden_dim, stream);
    linear_f32(handle, w1_out, w.w2, nullptr, out, w_workspace, M, dim, hidden_dim, stream);
}

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
                               cudaStream_t stream) {
    int Mq = B * Sq;
    int n = Mq * dim;

    // 1) Cross-attention in fp32
    rmsnorm_f32(x_f32, w.query_norm_gamma, temp_f32, Mq, dim, kEpsFp32, stream);

    float* ctx_normed = ffn_buf_f32;
    rmsnorm_f32(ctx_f32, w.kv_norm_gamma, ctx_normed, B * Sk, ctx_dim, kEpsFp32, stream);

    float* cross_out = qkv_buf_f32;
    cross_attention_f32(handle, w.cross_attn,
                        temp_f32, ctx_normed, cross_out, proj_buf_f32, cross_score_buf,
                        w_workspace, ctx_mask, B, Sq, Sk, dim, ctx_dim, dim, num_heads,
                        q_rope_cos, q_rope_sin, k_rope_cos, k_rope_sin, stream);
    residual_add_f32(x_f32, cross_out, n, stream);

    // 2) Swin self-attention in fp32
    rmsnorm_f32(x_f32, w.self_attn_norm_gamma, temp_f32, Mq, dim, kEpsFp32, stream);
    swin_self_attention_f32(handle, w.self_attn,
                             temp_f32, ffn_buf_f32, window_buf_f32, qkv_buf_f32, self_score_buf,
                             w_workspace, B, patch_h, patch_w, dim, num_heads,
                             window_size, shift_size, shift_mask, stream);
    residual_add_f32(x_f32, ffn_buf_f32, n, stream);

    // 3) FFN in fp32
    rmsnorm_f32(x_f32, w.ffn_norm_gamma, temp_f32, Mq, dim, kEpsFp32, stream);
    ffn_swiglu_f32(handle, w.ffn, temp_f32, ffn_buf_f32, qkv_buf_f32, w_workspace, Mq, dim, ffn_hidden, stream);
    residual_add_f32(x_f32, ffn_buf_f32, n, stream);
}

} // namespace rf
