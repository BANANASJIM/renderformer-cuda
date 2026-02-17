#include "layers/attention.h"
#include "layers/linear.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "kernels/softmax.h"
#include "kernels/flash_attn.h"
#include "kernels/cudnn_sdpa.h"
#include "core/cuda_utils.h"
#include <cmath>
#include <map>

namespace rf {

// ---- Utility kernels ----

// Convert [B, S, H, D] → [B, H, S, D] (transpose dims 1,2)
__global__ void transpose_bshd_to_bhsd(const half* __restrict__ in,
                                        half* __restrict__ out,
                                        int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * H * D;
    int b = blockIdx.y;
    if (idx >= total) return;

    int d = idx % D;
    int rem = idx / D;
    int h = rem % H;
    int s = rem / H;

    out[((b * H + h) * S + s) * D + d] = in[((b * S + s) * H + h) * D + d];
}

// Convert [B, H, S, D] → [B, S, H*D] (transpose + reshape)
__global__ void transpose_bhsd_to_bsd(const half* __restrict__ in,
                                       half* __restrict__ out,
                                       int H, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * S * D;
    int b = blockIdx.y;
    if (idx >= total) return;

    int d = idx % D;
    int rem = idx / D;
    int s = rem % S;
    int h = rem / S;

    out[(b * S + s) * (H * D) + h * D + d] = in[((b * H + h) * S + s) * D + d];
}

// Scale fp16 buffer: out[i] = in[i] * scale
__global__ void scale_fp16_to_fp32(const half* __restrict__ in,
                                    float* __restrict__ out,
                                    float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]) * scale;
}

// Convert fp32 → fp16
__global__ void fp32_to_fp16(const float* __restrict__ in,
                              half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(in[i]);
}

static void launch_1d(auto kernel, int n, cudaStream_t stream, auto... args) {
    int threads = 256;
    kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(args..., n);
}

// Deinterleave [M, 3*D] → Q[M,D], K[M,D], V[M,D]
__global__ void deinterleave_qkv(const half* __restrict__ src,
                                  half* __restrict__ q,
                                  half* __restrict__ k,
                                  half* __restrict__ v,
                                  int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int row = idx / D, col = idx % D;
    int src_idx = row * 3 * D + col;
    q[idx] = src[src_idx];
    k[idx] = src[src_idx + D];
    v[idx] = src[src_idx + 2 * D];
}

// ---- SDPA: fused softmax + zero-padded K/V for GEMM alignment ----

void sdpa(cublasHandle_t handle,
          const half* Q, const half* K, const half* V, half* O,
          float* workspace, const bool* mask,
          int B, int H, int Sq, int Sk, int D,
          cudaStream_t stream) {
    int BH = B * H;

    // cuDNN fused SDPA: tensor cores + no intermediate score matrix
    if (!mask) {
        using Key = std::tuple<int,int,int,int,int>;
        static std::map<Key, std::unique_ptr<CudnnSdpa>> cache;
        Key key{B, H, Sq, Sk, D};
        auto it = cache.find(key);
        if (it == cache.end()) {
            auto plan = CudnnSdpa::create(B, H, Sq, Sk, D);
            it = cache.emplace(key, std::move(plan)).first;
        }
        if (it->second) {
            it->second->forward(Q, K, V, O, stream);
            return;
        }
    }

    // Fallback: cuBLAS GEMM + softmax path
    cublasSetStream(handle, stream);
    float scale = 1.0f / sqrtf((float)D);
    float beta = 0.0f;

    int Sk_pad = (Sk + 7) & ~7;
    long long strideQ = (long long)Sq * D, strideO = strideQ;
    long long strideKp = (long long)Sk_pad * D;
    long long strideS = (long long)Sq * Sk_pad;

    const half *Kp = K, *Vp = V;
    half *K_pad = nullptr, *V_pad = nullptr;
    if (Sk_pad != Sk) {
        size_t psz = (size_t)BH * Sk_pad * D * sizeof(half);
        CHECK_CUDA(cudaMallocAsync(&K_pad, psz, stream));
        CHECK_CUDA(cudaMallocAsync(&V_pad, psz, stream));
        CHECK_CUDA(cudaMemsetAsync(K_pad, 0, psz, stream));
        CHECK_CUDA(cudaMemsetAsync(V_pad, 0, psz, stream));
        CHECK_CUDA(cudaMemcpy2DAsync(K_pad, Sk_pad * D * sizeof(half),
            K, (size_t)Sk * D * sizeof(half),
            (size_t)Sk * D * sizeof(half), BH,
            cudaMemcpyDeviceToDevice, stream));
        CHECK_CUDA(cudaMemcpy2DAsync(V_pad, Sk_pad * D * sizeof(half),
            V, (size_t)Sk * D * sizeof(half),
            (size_t)Sk * D * sizeof(half), BH,
            cudaMemcpyDeviceToDevice, stream));
        Kp = K_pad;
        Vp = V_pad;
    }

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N, Sk_pad, Sq, D,
        &scale, Kp, CUDA_R_16F, D, strideKp, Q, CUDA_R_16F, D, strideQ,
        &beta, workspace, CUDA_R_32F, Sk_pad, strideS,
        BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    half* P_half;
    CHECK_CUDA(cudaMallocAsync(&P_half, (size_t)BH * strideS * sizeof(half), stream));

    if (mask) softmax_cast_fp16_masked(workspace, P_half, mask, BH * Sq, Sk, Sk_pad, Sk_pad, B, stream);
    else softmax_cast_fp16(workspace, P_half, BH * Sq, Sk, Sk_pad, Sk_pad, stream);

    float one = 1.0f;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, D, Sq, Sk_pad,
        &one, Vp, CUDA_R_16F, D, strideKp, P_half, CUDA_R_16F, Sk_pad, strideS,
        &beta, O, CUDA_R_16F, D, strideO,
        BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    CHECK_CUDA(cudaFreeAsync(P_half, stream));
    if (K_pad) {
        CHECK_CUDA(cudaFreeAsync(K_pad, stream));
        CHECK_CUDA(cudaFreeAsync(V_pad, stream));
    }
}

// ---- Self-attention ----

void self_attention(cublasHandle_t handle, const MHAWeights& w,
                    const half* x, half* out, half* qkv_buf, float* score_buf,
                    const bool* mask, int B, int S, int dim, int num_heads,
                    const half* rope_cos, const half* rope_sin,
                    cudaStream_t stream) {
    int M = B * S;
    int head_dim = dim / num_heads;

    // QKV = x @ in_proj_w^T + in_proj_b → [M, 3*dim]
    linear(handle, x, w.in_proj_w, w.in_proj_b, qkv_buf, M, 3 * dim, dim, stream);

    // Deinterleave [M, 3*dim] → Q[M,dim], K[M,dim], V[M,dim]
    half* Q_bhsd; half* K_bhsd; half* V_bhsd;
    CHECK_CUDA(cudaMallocAsync(&Q_bhsd, M * dim * sizeof(half), stream));
    CHECK_CUDA(cudaMallocAsync(&K_bhsd, M * dim * sizeof(half), stream));
    CHECK_CUDA(cudaMallocAsync(&V_bhsd, M * dim * sizeof(half), stream));
    {
        int n = M * dim, threads = 256;
        deinterleave_qkv<<<(n + threads - 1) / threads, threads, 0, stream>>>(
            qkv_buf, Q_bhsd, K_bhsd, V_bhsd, M, dim);
    }

    // QK norm
    if (w.q_norm_gamma) rmsnorm(Q_bhsd, w.q_norm_gamma, Q_bhsd, M, dim, kEpsFp16, stream);
    if (w.k_norm_gamma) rmsnorm(K_bhsd, w.k_norm_gamma, K_bhsd, M, dim, kEpsFp16, stream);

    // Transpose [B, S, H, D] → [B, H, S, D]
    {
        int total = S * num_heads * head_dim;
        int threads = 256, blocks = (total + threads - 1) / threads;
        half* tmp = qkv_buf;
        transpose_bshd_to_bhsd<<<dim3(blocks, B), threads, 0, stream>>>(Q_bhsd, tmp, S, num_heads, head_dim);
        cudaMemcpyAsync(Q_bhsd, tmp, M * dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        transpose_bshd_to_bhsd<<<dim3(blocks, B), threads, 0, stream>>>(K_bhsd, tmp, S, num_heads, head_dim);
        cudaMemcpyAsync(K_bhsd, tmp, M * dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        transpose_bshd_to_bhsd<<<dim3(blocks, B), threads, 0, stream>>>(V_bhsd, tmp, S, num_heads, head_dim);
        cudaMemcpyAsync(V_bhsd, tmp, M * dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    }

    // Apply RoPE
    if (rope_cos)
        apply_rope(Q_bhsd, K_bhsd, rope_cos, rope_sin, B, num_heads, S, head_dim, stream);

    // SDPA
    half* O_bhsd = Q_bhsd;
    sdpa(handle, Q_bhsd, K_bhsd, V_bhsd, O_bhsd, score_buf, mask, B, num_heads, S, S, head_dim, stream);

    // Transpose [B, H, S, D] → [B, S, H*D]
    half* attn_out = K_bhsd;
    {
        int total = num_heads * S * head_dim;
        int thr = 256, blk = (total + thr - 1) / thr;
        transpose_bhsd_to_bsd<<<dim3(blk, B), thr, 0, stream>>>(O_bhsd, attn_out, num_heads, S, head_dim);
    }

    // Out projection
    linear(handle, attn_out, w.out_proj_w, w.out_proj_b, out, M, dim, dim, stream);

    CHECK_CUDA(cudaFreeAsync(Q_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(K_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(V_bhsd, stream));
}

// ---- Cross-attention ----

void cross_attention(cublasHandle_t handle, const CrossAttnWeights& w,
                     const half* q_in, const half* kv_in,
                     half* out, half* proj_buf, float* score_buf,
                     const bool* mask,
                     int B, int Sq, int Sk, int q_dim, int kv_dim, int dim, int num_heads,
                     const half* q_rope_cos, const half* q_rope_sin,
                     const half* k_rope_cos, const half* k_rope_sin,
                     cudaStream_t stream) {
    int Mq = B * Sq;
    int Mk = B * Sk;
    int head_dim = dim / num_heads;

    // Project Q, K, V
    half* Q = proj_buf;
    half* K = proj_buf + Mq * dim;
    half* V = proj_buf + Mq * dim + Mk * dim;

    linear(handle, q_in, w.q_proj_w, w.q_proj_b, Q, Mq, dim, q_dim, stream);
    linear(handle, kv_in, w.k_proj_w, w.k_proj_b, K, Mk, dim, kv_dim, stream);
    linear(handle, kv_in, w.v_proj_w, w.v_proj_b, V, Mk, dim, kv_dim, stream);

    // QK norm
    if (w.q_norm_gamma) rmsnorm(Q, w.q_norm_gamma, Q, Mq, dim, kEpsFp16, stream);
    if (w.k_norm_gamma) rmsnorm(K, w.k_norm_gamma, K, Mk, dim, kEpsFp16, stream);

    // Transpose to BHSD
    half *Q_bhsd, *K_bhsd, *V_bhsd;
    CHECK_CUDA(cudaMallocAsync(&Q_bhsd, Mq * dim * sizeof(half), stream));
    CHECK_CUDA(cudaMallocAsync(&K_bhsd, Mk * dim * sizeof(half), stream));
    CHECK_CUDA(cudaMallocAsync(&V_bhsd, Mk * dim * sizeof(half), stream));

    int threads = 256;
    {
        int total_q = Sq * num_heads * head_dim;
        transpose_bshd_to_bhsd<<<dim3((total_q + threads - 1) / threads, B), threads, 0, stream>>>(Q, Q_bhsd, Sq, num_heads, head_dim);
    }
    {
        int total_k = Sk * num_heads * head_dim;
        transpose_bshd_to_bhsd<<<dim3((total_k + threads - 1) / threads, B), threads, 0, stream>>>(K, K_bhsd, Sk, num_heads, head_dim);
        transpose_bshd_to_bhsd<<<dim3((total_k + threads - 1) / threads, B), threads, 0, stream>>>(V, V_bhsd, Sk, num_heads, head_dim);
    }

    // Apply RoPE
    if (q_rope_cos)
        apply_rope_one(Q_bhsd, q_rope_cos, q_rope_sin, B, num_heads, Sq, head_dim, stream);
    if (k_rope_cos)
        apply_rope_one(K_bhsd, k_rope_cos, k_rope_sin, B, num_heads, Sk, head_dim, stream);

    // SDPA
    half* O_bhsd = Q_bhsd;
    sdpa(handle, Q_bhsd, K_bhsd, V_bhsd, O_bhsd, score_buf, mask, B, num_heads, Sq, Sk, head_dim, stream);

    // Transpose back
    int total_q = Sq * num_heads * head_dim;
    half* attn_out = K_bhsd;
    transpose_bhsd_to_bsd<<<dim3((total_q + threads - 1) / threads, B), threads, 0, stream>>>(O_bhsd, attn_out, num_heads, Sq, head_dim);

    // Out projection
    linear(handle, attn_out, w.out_proj_w, w.out_proj_b, out, Mq, q_dim, dim, stream);

    CHECK_CUDA(cudaFreeAsync(Q_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(K_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(V_bhsd, stream));
}

// ---- KV Cache for cross-attention ----

void precompute_cross_kv(cublasHandle_t handle, const CrossAttnWeights& w,
                         const half* ctx_normed, half* K_bhsd_out, half* V_bhsd_out,
                         half* proj_buf,
                         int B, int Sk, int kv_dim, int dim, int num_heads,
                         cudaStream_t stream) {
    int Mk = B * Sk;
    int head_dim = dim / num_heads;

    half* K = proj_buf;
    half* V = proj_buf + Mk * dim;

    linear(handle, ctx_normed, w.k_proj_w, w.k_proj_b, K, Mk, dim, kv_dim, stream);
    linear(handle, ctx_normed, w.v_proj_w, w.v_proj_b, V, Mk, dim, kv_dim, stream);

    if (w.k_norm_gamma) rmsnorm(K, w.k_norm_gamma, K, Mk, dim, kEpsFp16, stream);

    int threads = 256;
    int total_k = Sk * num_heads * head_dim;
    transpose_bshd_to_bhsd<<<dim3((total_k + threads - 1) / threads, B), threads, 0, stream>>>(
        K, K_bhsd_out, Sk, num_heads, head_dim);
    transpose_bshd_to_bhsd<<<dim3((total_k + threads - 1) / threads, B), threads, 0, stream>>>(
        V, V_bhsd_out, Sk, num_heads, head_dim);
}

void cross_attention_cached(cublasHandle_t handle, const CrossAttnWeights& w,
                            const half* q_in, half* out,
                            half* proj_buf, float* score_buf,
                            const half* K_cached, const half* V_cached,
                            const bool* mask,
                            int B, int Sq, int Sk, int q_dim, int dim, int num_heads,
                            const half* q_rope_cos, const half* q_rope_sin,
                            const half* k_rope_cos, const half* k_rope_sin,
                            int Sk_pad_in,
                            cudaStream_t stream) {
    int Mq = B * Sq;
    int head_dim = dim / num_heads;
    int Skp = Sk_pad_in > 0 ? Sk_pad_in : Sk;

    // Q projection only (K/V are cached)
    half* Q = proj_buf;
    linear(handle, q_in, w.q_proj_w, w.q_proj_b, Q, Mq, dim, q_dim, stream);
    if (w.q_norm_gamma) rmsnorm(Q, w.q_norm_gamma, Q, Mq, dim, kEpsFp16, stream);

    // proj_buf layout: Q[Mq*dim] | Q_bhsd[Mq*dim] | K_bhsd[B*H*Skp*hd]
    half* Q_bhsd = proj_buf + Mq * dim;
    half* K_bhsd = Q_bhsd + Mq * dim;

    int threads = 256;
    int total_q = Sq * num_heads * head_dim;
    transpose_bshd_to_bhsd<<<dim3((total_q + threads - 1) / threads, B), threads, 0, stream>>>(
        Q, Q_bhsd, Sq, num_heads, head_dim);

    // Copy cached K (pre-padded to Skp) and apply per-view RoPE
    size_t k_sz = (size_t)B * num_heads * Skp * head_dim * sizeof(half);
    CHECK_CUDA(cudaMemcpyAsync(K_bhsd, K_cached, k_sz, cudaMemcpyDeviceToDevice, stream));

    if (q_rope_cos)
        apply_rope_one(Q_bhsd, q_rope_cos, q_rope_sin, B, num_heads, Sq, head_dim, stream);
    if (k_rope_cos)
        apply_rope_one(K_bhsd, k_rope_cos, k_rope_sin, B, num_heads, Sk, head_dim, stream);

    // SDPA with pre-padded K/V: pass Skp as Sk so sdpa skips padding
    half* O_bhsd = Q_bhsd;
    sdpa(handle, Q_bhsd, K_bhsd, V_cached, O_bhsd, score_buf, mask,
         B, num_heads, Sq, Skp, head_dim, stream);

    // Transpose back + out projection
    half* attn_out = K_bhsd;
    transpose_bhsd_to_bsd<<<dim3((total_q + threads - 1) / threads, B), threads, 0, stream>>>(
        O_bhsd, attn_out, num_heads, Sq, head_dim);
    linear(handle, attn_out, w.out_proj_w, w.out_proj_b, out, Mq, q_dim, dim, stream);
}

// ---- Window partition/reverse ----

__global__ void window_partition_kernel(const half* __restrict__ x,
                                        half* __restrict__ windows,
                                        int H, int W, int C, int ws) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int total = H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int rem = idx / C;
    int w_pos = rem % W;
    int h_pos = rem / W;

    int nW_w = W / ws;
    int wh = h_pos / ws, ww = w_pos / ws;
    int lh = h_pos % ws, lw = w_pos % ws;
    int nW = (H / ws) * nW_w;
    int win_idx = b * nW + wh * nW_w + ww;
    int local_idx = (lh * ws + lw) * C + c;

    windows[win_idx * ws * ws * C + local_idx] = x[(b * H * W + h_pos * W + w_pos) * C + c];
}

void window_partition(const half* x, half* windows,
                      int B, int H, int W, int C, int ws,
                      cudaStream_t stream) {
    int total = H * W * C;
    int threads = 256;
    window_partition_kernel<<<dim3((total + threads - 1) / threads, B), threads, 0, stream>>>(
        x, windows, H, W, C, ws);
}

__global__ void window_reverse_kernel(const half* __restrict__ windows,
                                       half* __restrict__ x,
                                       int H, int W, int C, int ws) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int total = H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int rem = idx / C;
    int w_pos = rem % W;
    int h_pos = rem / W;

    int nW_w = W / ws;
    int wh = h_pos / ws, ww = w_pos / ws;
    int lh = h_pos % ws, lw = w_pos % ws;
    int nW = (H / ws) * nW_w;
    int win_idx = b * nW + wh * nW_w + ww;
    int local_idx = (lh * ws + lw) * C + c;

    x[(b * H * W + h_pos * W + w_pos) * C + c] = windows[win_idx * ws * ws * C + local_idx];
}

void window_reverse(const half* windows, half* x,
                    int B, int H, int W, int C, int ws,
                    cudaStream_t stream) {
    int total = H * W * C;
    int threads = 256;
    window_reverse_kernel<<<dim3((total + threads - 1) / threads, B), threads, 0, stream>>>(
        windows, x, H, W, C, ws);
}

// ---- Cyclic shift ----

__global__ void cyclic_shift_kernel(const half* __restrict__ x,
                                     half* __restrict__ out,
                                     int H, int W, int C,
                                     int sh, int sw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int total = H * W * C;
    if (idx >= total) return;

    int c = idx % C;
    int rem = idx / C;
    int w_pos = rem % W;
    int h_pos = rem / W;

    int src_h = (h_pos + sh + H) % H;
    int src_w = (w_pos + sw + W) % W;

    out[(b * H * W + h_pos * W + w_pos) * C + c] =
        x[(b * H * W + src_h * W + src_w) * C + c];
}

void cyclic_shift_2d(const half* x, half* out,
                     int B, int H, int W, int C,
                     int shift_h, int shift_w,
                     cudaStream_t stream) {
    int total = H * W * C;
    int threads = 256;
    cyclic_shift_kernel<<<dim3((total + threads - 1) / threads, B), threads, 0, stream>>>(
        x, out, H, W, C, shift_h, shift_w);
}

// ---- Swin self-attention ----

void swin_self_attention(cublasHandle_t handle, const SwinAttnWeights& w,
                         const half* x, half* out,
                         half* window_buf, half* qkv_buf, float* score_buf,
                         half* scratch,
                         int B, int H, int W, int dim, int num_heads,
                         int window_size, int shift_size,
                         const bool* shift_mask,
                         cudaStream_t stream) {
    int ws = window_size;
    int nW = (H / ws) * (W / ws);
    int ws2 = ws * ws;
    int head_dim = dim / num_heads;
    int BnW = B * nW;
    int M = BnW * ws2;
    int BH = BnW * num_heads;

    // Scratch layout (overlapping lifetimes):
    //   [0 .. M*dim)                shifted (phase 1), then Q_t (phase 2), then shifted again (phase 3)
    //   [M*dim .. 2*M*dim)          K_t (phase 2)
    //   [2*M*dim .. 3*M*dim)        V_t (phase 2)
    //   [3*M*dim .. 3*M*dim+BH*ws4) P_half (phase 2, brief)

    half* shifted;
    if (shift_size > 0) {
        shifted = scratch;
        cyclic_shift_2d(x, shifted, B, H, W, dim, -shift_size, -shift_size, stream);
    } else {
        shifted = const_cast<half*>(x);
    }

    window_partition(shifted, window_buf, B, H, W, dim, ws, stream);

    linear(handle, window_buf, w.in_proj_w, w.in_proj_b, qkv_buf, M, 3 * dim, dim, stream);

    // Deinterleave into scratch (shifted is consumed by window_partition, safe to overwrite)
    half* Q_t = scratch;
    half* K_t = scratch + (size_t)M * dim;
    half* V_t = scratch + (size_t)2 * M * dim;
    {
        int n = M * dim, thr = 256;
        deinterleave_qkv<<<(n + thr - 1) / thr, thr, 0, stream>>>(
            qkv_buf, Q_t, K_t, V_t, M, dim);
    }

    if (w.q_norm_gamma) rmsnorm(Q_t, w.q_norm_gamma, Q_t, M, dim, kEpsFp16, stream);
    if (w.k_norm_gamma) rmsnorm(K_t, w.k_norm_gamma, K_t, M, dim, kEpsFp16, stream);

    // Transpose [BnW, ws2, H, D] → [BnW, H, ws2, D] directly to qkv_buf (no memcpy back)
    int total = ws2 * num_heads * head_dim;
    int threads = 256, blocks = (total + threads - 1) / threads;
    half* Q_bhsd = qkv_buf;
    half* K_bhsd = qkv_buf + (size_t)M * dim;
    half* V_bhsd = qkv_buf + (size_t)2 * M * dim;
    transpose_bshd_to_bhsd<<<dim3(blocks, BnW), threads, 0, stream>>>(Q_t, Q_bhsd, ws2, num_heads, head_dim);
    transpose_bshd_to_bhsd<<<dim3(blocks, BnW), threads, 0, stream>>>(K_t, K_bhsd, ws2, num_heads, head_dim);
    transpose_bshd_to_bhsd<<<dim3(blocks, BnW), threads, 0, stream>>>(V_t, V_bhsd, ws2, num_heads, head_dim);

    // Inline SDPA (Q/K/V in qkv_buf BHSD, output overwrites Q)
    half* O_bhsd = Q_bhsd;
    {
        cublasSetStream(handle, stream);
        float scale = 1.0f / sqrtf((float)head_dim);
        float beta_z = 0.0f;
        long long strideQ = (long long)ws2 * head_dim;
        long long strideS = (long long)ws2 * ws2;

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N, ws2, ws2, head_dim,
            &scale, K_bhsd, CUDA_R_16F, head_dim, strideQ,
            Q_bhsd, CUDA_R_16F, head_dim, strideQ,
            &beta_z, score_buf, CUDA_R_32F, ws2, strideS,
            BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        if (shift_mask && shift_size > 0)
            swin_masked_softmax_inplace(score_buf, shift_mask, BnW, num_heads, ws2, nW, stream);
        else
            softmax_inplace(score_buf, BH * ws2, ws2, stream);

        half* P_half = scratch + (size_t)3 * M * dim;
        launch_1d(fp32_to_fp16, BH * ws2 * ws2, stream, score_buf, P_half);

        float one = 1.0f;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N, head_dim, ws2, ws2,
            &one, V_bhsd, CUDA_R_16F, head_dim, strideQ,
            P_half, CUDA_R_16F, ws2, strideS,
            &beta_z, O_bhsd, CUDA_R_16F, head_dim, strideQ,
            BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    transpose_bhsd_to_bsd<<<dim3(blocks, BnW), threads, 0, stream>>>(O_bhsd, window_buf, num_heads, ws2, head_dim);
    linear(handle, window_buf, w.out_proj_w, w.out_proj_b, qkv_buf, M, dim, dim, stream);

    if (shift_size > 0) {
        half* shift_buf = scratch; // Q_t/K_t/V_t/P_half are dead, reuse scratch
        window_reverse(qkv_buf, shift_buf, B, H, W, dim, ws, stream);
        cyclic_shift_2d(shift_buf, out, B, H, W, dim, shift_size, shift_size, stream);
    } else {
        window_reverse(qkv_buf, out, B, H, W, dim, ws, stream);
    }
}

// ================================================================
// fp32 variants
// ================================================================

// fp32 utility kernels (must precede their call sites)
__global__ void deinterleave_qkv_f32(const float* __restrict__ src,
                                      float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
                                      int M, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * D) return;
    int row = idx / D, col = idx % D;
    q[idx] = src[row * 3 * D + col];
    k[idx] = src[row * 3 * D + D + col];
    v[idx] = src[row * 3 * D + 2 * D + col];
}

__global__ void transpose_bshd_to_bhsd_f32(const float* __restrict__ in,
                                            float* __restrict__ out,
                                            int S, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (idx >= S * H * D) return;
    int d = idx % D, rem = idx / D, h = rem % H, s = rem / H;
    out[((b * H + h) * S + s) * D + d] = in[((b * S + s) * H + h) * D + d];
}

__global__ void transpose_bhsd_to_bsd_f32(const float* __restrict__ in,
                                           float* __restrict__ out,
                                           int H, int S, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (idx >= H * S * D) return;
    int d = idx % D, rem = idx / D, s = rem % S, h = rem / S;
    out[(b * S + s) * (H * D) + h * D + d] = in[((b * H + h) * S + s) * D + d];
}

void self_attention_f32(cublasHandle_t handle, const MHAWeights& w,
                        const float* x, float* out, float* qkv_buf, float* score_buf,
                        float* w_workspace,
                        const bool* mask, int B, int S, int dim, int num_heads,
                        const half* rope_cos, const half* rope_sin,
                        cudaStream_t stream) {
    int M = B * S, hd = dim / num_heads, t = 256;

    // QKV projection: [M, dim] → [M, 3*dim]
    linear_f32(handle, x, w.in_proj_w, w.in_proj_b, qkv_buf, w_workspace, M, 3 * dim, dim, stream);

    // Deinterleave
    float *Q_bhsd, *K_bhsd, *V_bhsd;
    CHECK_CUDA(cudaMallocAsync(&Q_bhsd, M * dim * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&K_bhsd, M * dim * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&V_bhsd, M * dim * sizeof(float), stream));
    {
        int n = M * dim;
        deinterleave_qkv_f32<<<(n + t - 1) / t, t, 0, stream>>>(qkv_buf, Q_bhsd, K_bhsd, V_bhsd, M, dim);
    }

    // QK norm
    if (w.q_norm_gamma) rmsnorm_f32(Q_bhsd, w.q_norm_gamma, Q_bhsd, M, dim, kEpsFp32, stream);
    if (w.k_norm_gamma) rmsnorm_f32(K_bhsd, w.k_norm_gamma, K_bhsd, M, dim, kEpsFp32, stream);

    // Transpose [B, S, H, D] → [B, H, S, D]
    {
        int n = S * num_heads * hd, blocks = (n + t - 1) / t;
        float* tmp = qkv_buf;
        transpose_bshd_to_bhsd_f32<<<dim3(blocks, B), t, 0, stream>>>(Q_bhsd, tmp, S, num_heads, hd);
        cudaMemcpyAsync(Q_bhsd, tmp, M * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        transpose_bshd_to_bhsd_f32<<<dim3(blocks, B), t, 0, stream>>>(K_bhsd, tmp, S, num_heads, hd);
        cudaMemcpyAsync(K_bhsd, tmp, M * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        transpose_bshd_to_bhsd_f32<<<dim3(blocks, B), t, 0, stream>>>(V_bhsd, tmp, S, num_heads, hd);
        cudaMemcpyAsync(V_bhsd, tmp, M * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // RoPE
    if (rope_cos)
        apply_rope_f32(Q_bhsd, K_bhsd, rope_cos, rope_sin, B, num_heads, S, hd, stream);

    // SDPA
    float* O_bhsd = Q_bhsd;
    sdpa_f32(handle, Q_bhsd, K_bhsd, V_bhsd, O_bhsd, score_buf, mask, B, num_heads, S, S, hd, stream);

    // Transpose back [B, H, S, D] → [B, S, H*D]
    float* attn_out = K_bhsd;
    {
        int n = num_heads * S * hd, blocks = (n + t - 1) / t;
        transpose_bhsd_to_bsd_f32<<<dim3(blocks, B), t, 0, stream>>>(O_bhsd, attn_out, num_heads, S, hd);
    }

    // Out projection
    linear_f32(handle, attn_out, w.out_proj_w, w.out_proj_b, out, w_workspace, M, dim, dim, stream);

    CHECK_CUDA(cudaFreeAsync(Q_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(K_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(V_bhsd, stream));
}

// fp32 window partition/reverse/cyclic_shift
__global__ void window_partition_f32_kernel(const float* __restrict__ x, float* __restrict__ windows,
                                             int H, int W, int C, int ws) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (idx >= H * W * C) return;
    int c = idx % C, rem = idx / C, w_pos = rem % W, h_pos = rem / W;
    int nW_w = W / ws, wh = h_pos / ws, ww = w_pos / ws;
    int lh = h_pos % ws, lw = w_pos % ws;
    int nW = (H / ws) * nW_w;
    int win_idx = b * nW + wh * nW_w + ww;
    windows[win_idx * ws * ws * C + (lh * ws + lw) * C + c] = x[(b * H * W + h_pos * W + w_pos) * C + c];
}

__global__ void window_reverse_f32_kernel(const float* __restrict__ windows, float* __restrict__ x,
                                           int H, int W, int C, int ws) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (idx >= H * W * C) return;
    int c = idx % C, rem = idx / C, w_pos = rem % W, h_pos = rem / W;
    int nW_w = W / ws, wh = h_pos / ws, ww = w_pos / ws;
    int lh = h_pos % ws, lw = w_pos % ws;
    int nW = (H / ws) * nW_w;
    int win_idx = b * nW + wh * nW_w + ww;
    x[(b * H * W + h_pos * W + w_pos) * C + c] = windows[win_idx * ws * ws * C + (lh * ws + lw) * C + c];
}

__global__ void cyclic_shift_f32_kernel(const float* __restrict__ x, float* __restrict__ out,
                                         int H, int W, int C, int sh, int sw) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (idx >= H * W * C) return;
    int c = idx % C, rem = idx / C, w_pos = rem % W, h_pos = rem / W;
    int src_h = (h_pos + sh + H) % H, src_w = (w_pos + sw + W) % W;
    out[(b * H * W + h_pos * W + w_pos) * C + c] = x[(b * H * W + src_h * W + src_w) * C + c];
}

void sdpa_f32(cublasHandle_t handle,
              const float* Q, const float* K, const float* V, float* O,
              float* workspace,
              const bool* mask, int B, int H, int Sq, int Sk, int D,
              cudaStream_t stream) {
    cublasSetStream(handle, stream);
    int BH = B * H;
    float scale = 1.0f / sqrtf((float)D);
    float beta = 0.0f;

    long long strideQ = (long long)Sq * D, strideK = (long long)Sk * D;
    long long strideS = (long long)Sq * Sk, strideV = strideK, strideO = strideQ;

    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_T, CUBLAS_OP_N, Sk, Sq, D,
        &scale, K, CUDA_R_32F, D, strideK, Q, CUDA_R_32F, D, strideQ,
        &beta, workspace, CUDA_R_32F, Sk, strideS,
        BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

    if (mask) masked_softmax_inplace(workspace, mask, BH * Sq, Sk, B, stream);
    else softmax_inplace(workspace, BH * Sq, Sk, stream);

    float one = 1.0f;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, D, Sq, Sk,
        &one, V, CUDA_R_32F, D, strideV, workspace, CUDA_R_32F, Sk, strideS,
        &beta, O, CUDA_R_32F, D, strideO,
        BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

void cross_attention_f32(cublasHandle_t handle, const CrossAttnWeights& w,
                         const float* q_in, const float* kv_in,
                         float* out, float* proj_buf, float* score_buf,
                         float* w_workspace,
                         const bool* mask,
                         int B, int Sq, int Sk, int q_dim, int kv_dim, int dim, int num_heads,
                         const half* q_rope_cos, const half* q_rope_sin,
                         const half* k_rope_cos, const half* k_rope_sin,
                         cudaStream_t stream) {
    int Mq = B * Sq, Mk = B * Sk, hd = dim / num_heads;
    int t = 256;

    // proj_buf layout: Q[Mq*dim] + K[Mk*dim] + V[Mk*dim]
    float* Q = proj_buf;
    float* K = proj_buf + Mq * dim;
    float* V = proj_buf + Mq * dim + Mk * dim;

    linear_f32(handle, q_in, w.q_proj_w, w.q_proj_b, Q, w_workspace, Mq, dim, q_dim, stream);
    linear_f32(handle, kv_in, w.k_proj_w, w.k_proj_b, K, w_workspace, Mk, dim, kv_dim, stream);
    linear_f32(handle, kv_in, w.v_proj_w, w.v_proj_b, V, w_workspace, Mk, dim, kv_dim, stream);

    // QK norm
    if (w.q_norm_gamma) rmsnorm_f32(Q, w.q_norm_gamma, Q, Mq, dim, kEpsFp32, stream);
    if (w.k_norm_gamma) rmsnorm_f32(K, w.k_norm_gamma, K, Mk, dim, kEpsFp32, stream);

    // Transpose [B,S,H,D] → [B,H,S,D]
    float *Q_bhsd, *K_bhsd, *V_bhsd;
    CHECK_CUDA(cudaMallocAsync(&Q_bhsd, Mq * dim * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&K_bhsd, Mk * dim * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&V_bhsd, Mk * dim * sizeof(float), stream));

    {
        int n_q = Sq * num_heads * hd, n_k = Sk * num_heads * hd;
        transpose_bshd_to_bhsd_f32<<<dim3((n_q + t - 1) / t, B), t, 0, stream>>>(Q, Q_bhsd, Sq, num_heads, hd);
        transpose_bshd_to_bhsd_f32<<<dim3((n_k + t - 1) / t, B), t, 0, stream>>>(K, K_bhsd, Sk, num_heads, hd);
        transpose_bshd_to_bhsd_f32<<<dim3((n_k + t - 1) / t, B), t, 0, stream>>>(V, V_bhsd, Sk, num_heads, hd);
    }

    // RoPE
    if (q_rope_cos) apply_rope_one_f32(Q_bhsd, q_rope_cos, q_rope_sin, B, num_heads, Sq, hd, stream);
    if (k_rope_cos) apply_rope_one_f32(K_bhsd, k_rope_cos, k_rope_sin, B, num_heads, Sk, hd, stream);

    // SDPA
    float* O_bhsd = Q_bhsd;
    sdpa_f32(handle, Q_bhsd, K_bhsd, V_bhsd, O_bhsd, score_buf, mask, B, num_heads, Sq, Sk, hd, stream);

    // Transpose back [B,H,S,D] → [B,S,H*D]
    float* attn_out = K_bhsd;
    {
        int n = Sq * num_heads * hd;
        transpose_bhsd_to_bsd_f32<<<dim3((n + t - 1) / t, B), t, 0, stream>>>(O_bhsd, attn_out, num_heads, Sq, hd);
    }

    // Out projection
    linear_f32(handle, attn_out, w.out_proj_w, w.out_proj_b, out, w_workspace, Mq, q_dim, dim, stream);

    CHECK_CUDA(cudaFreeAsync(Q_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(K_bhsd, stream));
    CHECK_CUDA(cudaFreeAsync(V_bhsd, stream));
}

void swin_self_attention_f32(cublasHandle_t handle, const SwinAttnWeights& w,
                              const float* x, float* out,
                              float* window_buf, float* qkv_buf, float* score_buf,
                              float* w_workspace,
                              int B, int H, int W, int dim, int num_heads,
                              int window_size, int shift_size,
                              const bool* shift_mask,
                              cudaStream_t stream) {
    int ws = window_size, nW = (H / ws) * (W / ws), ws2 = ws * ws;
    int hd = dim / num_heads, t = 256;
    int BnW = B * nW, M = BnW * ws2;

    float* shifted;
    if (shift_size > 0) {
        CHECK_CUDA(cudaMallocAsync(&shifted, (size_t)B * H * W * dim * sizeof(float), stream));
        int total = H * W * dim;
        cyclic_shift_f32_kernel<<<dim3((total + t - 1) / t, B), t, 0, stream>>>(
            x, shifted, H, W, dim, -shift_size, -shift_size);
    } else {
        shifted = const_cast<float*>(x);
    }

    // Window partition
    {
        int total = H * W * dim;
        window_partition_f32_kernel<<<dim3((total + t - 1) / t, B), t, 0, stream>>>(
            shifted, window_buf, H, W, dim, ws);
    }

    // QKV projection
    linear_f32(handle, window_buf, w.in_proj_w, w.in_proj_b, qkv_buf, w_workspace, M, 3 * dim, dim, stream);

    // Deinterleave
    float *Q_t, *K_t, *V_t;
    CHECK_CUDA(cudaMallocAsync(&Q_t, M * dim * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&K_t, M * dim * sizeof(float), stream));
    CHECK_CUDA(cudaMallocAsync(&V_t, M * dim * sizeof(float), stream));
    {
        int n = M * dim;
        deinterleave_qkv_f32<<<(n + t - 1) / t, t, 0, stream>>>(qkv_buf, Q_t, K_t, V_t, M, dim);
    }

    // QK norm
    if (w.q_norm_gamma) rmsnorm_f32(Q_t, w.q_norm_gamma, Q_t, M, dim, kEpsFp32, stream);
    if (w.k_norm_gamma) rmsnorm_f32(K_t, w.k_norm_gamma, K_t, M, dim, kEpsFp32, stream);

    // Transpose [BnW, ws2, H, D] → [BnW, H, ws2, D]
    int total = ws2 * num_heads * hd;
    int blocks = (total + t - 1) / t;
    {
        float* tmp = qkv_buf;
        transpose_bshd_to_bhsd_f32<<<dim3(blocks, BnW), t, 0, stream>>>(Q_t, tmp, ws2, num_heads, hd);
        cudaMemcpyAsync(Q_t, tmp, M * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        transpose_bshd_to_bhsd_f32<<<dim3(blocks, BnW), t, 0, stream>>>(K_t, tmp, ws2, num_heads, hd);
        cudaMemcpyAsync(K_t, tmp, M * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        transpose_bshd_to_bhsd_f32<<<dim3(blocks, BnW), t, 0, stream>>>(V_t, tmp, ws2, num_heads, hd);
        cudaMemcpyAsync(V_t, tmp, M * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Inline SDPA
    float* O_t = Q_t;
    {
        cublasSetStream(handle, stream);
        float scale = 1.0f / sqrtf((float)hd), beta_z = 0.0f;
        int BH = BnW * num_heads;
        long long sQ = (long long)ws2 * hd, sS = (long long)ws2 * ws2;

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_T, CUBLAS_OP_N, ws2, ws2, hd,
            &scale, K_t, CUDA_R_32F, hd, sQ, Q_t, CUDA_R_32F, hd, sQ,
            &beta_z, score_buf, CUDA_R_32F, ws2, sS,
            BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        if (shift_mask && shift_size > 0)
            swin_masked_softmax_inplace(score_buf, shift_mask, BnW, num_heads, ws2, nW, stream);
        else
            softmax_inplace(score_buf, BH * ws2, ws2, stream);

        float one = 1.0f;
        CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N, hd, ws2, ws2,
            &one, V_t, CUDA_R_32F, hd, sQ, score_buf, CUDA_R_32F, ws2, sS,
            &beta_z, O_t, CUDA_R_32F, hd, sQ,
            BH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }

    // Transpose back
    transpose_bhsd_to_bsd_f32<<<dim3(blocks, BnW), t, 0, stream>>>(O_t, window_buf, num_heads, ws2, hd);

    // Out projection
    linear_f32(handle, window_buf, w.out_proj_w, w.out_proj_b, qkv_buf, w_workspace, M, dim, dim, stream);

    // Window reverse
    if (shift_size > 0) {
        int total = H * W * dim;
        window_reverse_f32_kernel<<<dim3((total + t - 1) / t, B), t, 0, stream>>>(qkv_buf, shifted, H, W, dim, ws);
        cyclic_shift_f32_kernel<<<dim3((total + t - 1) / t, B), t, 0, stream>>>(
            shifted, out, H, W, dim, shift_size, shift_size);
    } else {
        int total = H * W * dim;
        window_reverse_f32_kernel<<<dim3((total + t - 1) / t, B), t, 0, stream>>>(qkv_buf, out, H, W, dim, ws);
    }

    CHECK_CUDA(cudaFreeAsync(Q_t, stream));
    CHECK_CUDA(cudaFreeAsync(K_t, stream));
    CHECK_CUDA(cudaFreeAsync(V_t, stream));
    if (shift_size > 0) CHECK_CUDA(cudaFreeAsync(shifted, stream));
}

} // namespace rf
