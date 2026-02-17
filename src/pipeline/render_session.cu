#include "pipeline/render_session.h"
#include "core/cuda_utils.h"
#include "kernels/rmsnorm.h"
#include "kernels/nerf_pe.h"
#include "kernels/rope.h"
#include "kernels/activations.h"
#include "kernels/ray_gen.h"
#include "kernels/coord_transform.h"
#include "layers/linear.h"
#include "layers/transformer.h"
#include "layers/dpt.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

namespace rf {

// ---- Utility kernels (shared with renderer.cu) ----

__global__ void k_fp32_to_fp16_s(const float* __restrict__ in, half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
__global__ void k_fp16_to_fp32_s(const half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}
__global__ void k_extract_tex_s(const half* __restrict__ in, half* __restrict__ out, int NC, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NC) out[i] = in[i * HW];
}
__global__ void k_log_encode_s(half* __restrict__ data, int N, int C, int last_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * last_n) return;
    int n = i / last_n, idx = n * C + (C - last_n) + (i % last_n);
    data[idx] = __float2half(log10f(__half2float(data[idx]) + 1.0f));
}
__global__ void k_add3_bc_f32_s(float* __restrict__ out, const float* __restrict__ a,
                                 const float* __restrict__ b, const half* __restrict__ c, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i] + __half2float(c[i % dim]);
}
__global__ void k_broadcast_add_s(half* __restrict__ out, const half* __restrict__ bc, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(out[i]) + __half2float(bc[i % dim]));
}
__global__ void k_patchify_s(const half* __restrict__ in, half* __restrict__ out,
                              int H, int W, int C, int ps, int pH, int pW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pH * pW * C * ps * ps;
    if (idx >= total) return;
    int p2 = idx % ps, r = idx / ps;
    int p1 = r % ps; r /= ps;
    int c = r % C, patch = r / C;
    int h1 = patch / pW, w1 = patch % pW;
    out[patch * (C * ps * ps) + c * (ps * ps) + p1 * ps + p2] =
        in[((h1 * ps + p1) * W + w1 * ps + p2) * C + c];
}

// CPU helper: compute world-space centroid from fp16 triangles (one-time at session creation)
static void compute_world_centroid(const half* tri_gpu, int N, float out[3], cudaStream_t s) {
    std::vector<half> h(N * 9);
    CHECK_CUDA(cudaMemcpyAsync(h.data(), tri_gpu, N * 9 * sizeof(half), cudaMemcpyDeviceToHost, s));
    CHECK_CUDA(cudaStreamSynchronize(s));
    double m[9] = {};
    for (int i = 0; i < N; i++) for (int d = 0; d < 9; d++) m[d] += __half2float(h[i * 9 + d]);
    for (auto& v : m) v /= N;
    for (int j = 0; j < 3; j++) out[j] = (float)((m[j] + m[3+j] + m[6+j]) / 3.0);
}

// GPU: transform world centroid to camera space, broadcast to nr register positions [nr, 9]
__global__ void k_center_pos_from_c2w(const float* __restrict__ c2w,
                                       const float* __restrict__ world_center,
                                       half* __restrict__ out, int nr) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nr) return;
    float cx = world_center[0], cy = world_center[1], cz = world_center[2];
    // R^T(c - t), same as world_to_cam
    float vx = cx - c2w[3], vy = cy - c2w[7], vz = cz - c2w[11];
    half hx = __float2half(c2w[0]*vx + c2w[4]*vy + c2w[8]*vz);
    half hy = __float2half(c2w[1]*vx + c2w[5]*vy + c2w[9]*vz);
    half hz = __float2half(c2w[2]*vx + c2w[6]*vy + c2w[10]*vz);
    int base = r * 9;
    out[base]=hx; out[base+1]=hy; out[base+2]=hz;
    out[base+3]=hx; out[base+4]=hy; out[base+5]=hz;
    out[base+6]=hx; out[base+7]=hy; out[base+8]=hz;
}

// CPU helper: compute world-space centroid and fill register positions (encoder, one-time)
static void compute_center_pos(const half* tri_gpu, int N, half* out, int nr, cudaStream_t s) {
    float c[3];
    compute_world_centroid(tri_gpu, N, c, s);
    std::vector<half> hc(nr * 9);
    for (int r = 0; r < nr; r++) for (int d = 0; d < 9; d++) hc[r*9+d] = __float2half(c[d%3]);
    CHECK_CUDA(cudaMemcpyAsync(out, hc.data(), nr * 9 * sizeof(half), cudaMemcpyHostToDevice, s));
}

static GpuBuffer make_shift_mask(int pH, int pW, int ws, int shift) {
    int nH = pH/ws, nW = pW/ws, nWin = nH*nW, ws2 = ws*ws;
    std::vector<int> reg(pH*pW);
    for (int i = 0; i < pH; i++) for (int j = 0; j < pW; j++)
        reg[i*pW+j] = ((i+shift)%pH/ws)*nW + ((j+shift)%pW/ws);
    std::vector<char> mask(nWin*ws2*ws2, 1);
    for (int wi = 0; wi < nH; wi++) for (int wj = 0; wj < nW; wj++) {
        int wid = wi*nW+wj;
        for (int a = 0; a < ws2; a++) {
            int ra = reg[(wi*ws+a/ws)*pW+wj*ws+a%ws];
            for (int b = 0; b < ws2; b++) {
                int rb = reg[(wi*ws+b/ws)*pW+wj*ws+b%ws];
                mask[wid*ws2*ws2+a*ws2+b] = (ra == rb);
            }
        }
    }
    return upload(mask.data(), mask.size());
}

#define L(n) (((n) + 255) / 256)

// ---- Session implementation ----

struct RenderSession::Impl {
    CudaStream stream;
    CublasHandle cublas;
    const RenderFormerModel* model;

    int N, img_res;
    int Se, Se_pad, pH, pW, nP;

    // Preprocessed scene (reused across frames)
    GpuBuffer tri16, vn16;      // fp16 tri/vn [N,9]
    GpuBuffer ctx_fp16;         // encoder output [Se, dim] fp16
    float world_center[3];      // precomputed world-space centroid
    GpuBuffer world_center_gpu; // [3] fp32 on device

    // KV cache [12 layers], pre-padded to Se_pad
    std::vector<GpuBuffer> kv_K, kv_V;

    // Per-frame reusable buffers
    GpuBuffer c2w_gpu;          // [16] fp32, reused across frames
    GpuBuffer vtri, vvn, vpos;
    GpuBuffer rays_o, rays_d, ray_pat, rtok;
    GpuBuffer k_cos, k_sin, q_cos, q_sin;
    GpuBuffer dec_temp, dec_proj, dec_qkv, dec_ffn, dec_win;
    GpuBuffer score, self_score;
    GpuBuffer shift_mask, id_c2w_buf, fov_gpu;
    GpuBuffer dft[4], dpt_out, col;
    GpuBuffer out_fp32; // [3, H, W] fp32 output

    // Async support
    cudaEvent_t frame_done = nullptr;
    bool async_pending = false;

    // CUDA Graph (captured after warmup)
    float* pinned_c2w = nullptr;
    float* pinned_fov = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    // Frame timing
    GpuTimer timer;

    Impl() {
        CHECK_CUDA(cudaEventCreateWithFlags(&frame_done, cudaEventDisableTiming));
    }
    ~Impl() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (pinned_c2w) cudaFreeHost(pinned_c2w);
        if (pinned_fov) cudaFreeHost(pinned_fov);
        if (frame_done) cudaEventDestroy(frame_done);
    }

    void render_view(const float* c2w_src, const float* fov_src);
};

void RenderSession::Impl::render_view(const float* c2w_src, const float* fov_src) {
    constexpr int dim = Config::latent_dim;
    constexpr int nh = Config::num_heads;
    constexpr int hd = Config::head_dim;
    constexpr int nr = Config::num_register_tokens;
    constexpr int ffn = Config::ffn_hidden;
    constexpr int ps = Config::patch_size;
    constexpr int ws = Config::window_size;

    cudaStream_t s = stream;

    // Upload c2w and fov from host (pinned memory for graph capture)
    CHECK_CUDA(cudaMemcpyAsync(c2w_gpu.data, c2w_src, 16 * sizeof(float), cudaMemcpyHostToDevice, s));

    // Transform to camera space
    CHECK_CUDA(cudaMemcpyAsync(vtri.data, tri16.data, N * 9 * sizeof(half), cudaMemcpyDeviceToDevice, s));
    CHECK_CUDA(cudaMemcpyAsync(vvn.data, vn16.data, N * 9 * sizeof(half), cudaMemcpyDeviceToDevice, s));
    world_to_cam((float*)c2w_gpu.data, (half*)vtri.data, (half*)vvn.data, 1, N, s);

    // K RoPE: register positions from precomputed centroid (GPU, no sync)
    k_center_pos_from_c2w<<<L(nr), 256, 0, s>>>(
        (float*)c2w_gpu.data, (float*)world_center_gpu.data, (half*)vpos.data, nr);
    CHECK_CUDA(cudaMemcpyAsync((half*)vpos.data + nr * 9, vtri.data,
                                N * 9 * sizeof(half), cudaMemcpyDeviceToDevice, s));
    triangle_rope_freqs((half*)vpos.data, model->vt_rope_freqs,
                        (half*)k_cos.data, (half*)k_sin.data,
                        1, Se_pad, Config::vt_rope_dim / 2, hd, s);

    // Ray generation
    CHECK_CUDA(cudaMemcpyAsync(fov_gpu.data, fov_src, sizeof(float), cudaMemcpyHostToDevice, s));
    ray_gen((float*)id_c2w_buf.data, (float*)fov_gpu.data,
            (float*)rays_o.data, (half*)rays_d.data, 1, img_res, s);

    // Patchify + ray token encoding
    k_patchify_s<<<L(nP * 3 * ps * ps), 256, 0, s>>>(
        (half*)rays_d.data, (half*)ray_pat.data, img_res, img_res, 3, ps, pH, pW);
    linear(cublas, (half*)ray_pat.data, model->ray_enc_w, model->ray_enc_b,
           (half*)rtok.data, nP, dim, Config::ray_encoder_in, s);
    rmsnorm((half*)rtok.data, model->ray_enc_norm, (half*)rtok.data, nP, dim, kEpsFp16, s);
    k_broadcast_add_s<<<L(nP * dim), 256, 0, s>>>(
        (half*)rtok.data, model->ray_map_patch_token, nP * dim, dim);

    // Decoder (12 layers)
    for (int i = 0; i < Config::vt_num_layers; i++) {
        int cur_shift = (i % 2 == 1) ? Config::shift_size : 0;
        const bool* smask = cur_shift > 0 ? (const bool*)shift_mask.data : nullptr;
        decoder_layer_forward_cached(cublas, model->dec_layers[i],
                              (half*)rtok.data, (half*)dec_temp.data,
                              (half*)dec_proj.data, (half*)dec_qkv.data,
                              (half*)dec_ffn.data, (half*)dec_win.data,
                              (float*)score.data, (float*)self_score.data,
                              (half*)kv_K[i].data, (half*)kv_V[i].data,
                              nullptr,
                              1, nP, Se_pad, dim, nh, ffn,
                              pH, pW, ws, cur_shift, smask,
                              (half*)q_cos.data, (half*)q_sin.data,
                              (half*)k_cos.data, (half*)k_sin.data, s);

        for (int j = 0; j < 4; j++)
            if (i == Config::dpt_out_layers[j])
                CHECK_CUDA(cudaMemcpyAsync(dft[j].data, rtok.data,
                                           nP * dim * sizeof(half), cudaMemcpyDeviceToDevice, s));
    }

    // DPT head
    const half* feat_ptrs[4] = {(half*)dft[0].data, (half*)dft[1].data,
                                 (half*)dft[2].data, (half*)dft[3].data};
    dpt_forward(cublas, model->dpt, feat_ptrs, (half*)dpt_out.data, (half*)col.data,
                1, dim, pH, pW, ps, Config::dpt_features, Config::dpt_out_dim,
                Config::dpt_out_channels, s);

    // Postprocess
    int px_total = Config::dpt_out_dim * img_res * img_res;
    elu((half*)dpt_out.data, (half*)dpt_out.data, px_total, 1e-3f, s);
    k_fp16_to_fp32_s<<<L(px_total), 256, 0, s>>>(
        (half*)dpt_out.data, (float*)out_fp32.data, px_total);
    pow10_minus1((float*)out_fp32.data, (float*)out_fp32.data, px_total, s);
}

// ---- Public API ----

RenderSession RenderSession::create(const RenderFormerModel& model,
                                     const Scene& scene, int img_res) {
    RenderSession sess;
    sess.img_res_ = img_res;
    sess.impl_ = std::make_unique<Impl>();
    auto& d = *sess.impl_;

    d.model = &model;
    d.cublas.set_stream(d.stream);

    // Configure memory pool
    cudaMemPool_t pool;
    CHECK_CUDA(cudaDeviceGetDefaultMemPool(&pool, 0));
    uint64_t threshold = UINT64_MAX;
    CHECK_CUDA(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    constexpr int dim = Config::latent_dim;
    constexpr int nh = Config::num_heads;
    constexpr int hd = Config::head_dim;
    constexpr int nr = Config::num_register_tokens;
    constexpr int ffn = Config::ffn_hidden;
    constexpr int ps = Config::patch_size;
    constexpr int ws = Config::window_size;

    d.N = (int)scene.num_triangles;
    d.img_res = img_res;
    d.Se = nr + d.N;
    d.Se_pad = (d.Se + 7) & ~7;
    d.pH = img_res / ps;
    d.pW = img_res / ps;
    d.nP = d.pH * d.pW;

    int N = d.N, Se = d.Se, Se_pad = d.Se_pad, nP = d.nP, pH = d.pH, pW = d.pW;
    cudaStream_t s = d.stream;

    printf("RenderSession: %d tris, %dx%d (patches %dx%d)\n", N, img_res, img_res, pH, pW);
    fflush(stdout);

    // ---- Preprocessing ----
    GpuBuffer tex_buf(N * 13 * sizeof(half));
    k_extract_tex_s<<<L(N * 13), 256, 0, s>>>(scene.texture().fp16(), (half*)tex_buf.data, N * 13, 32 * 32);
    k_log_encode_s<<<L(N * 3), 256, 0, s>>>((half*)tex_buf.data, N, 13, 3);

    d.tri16 = GpuBuffer(N * 9 * sizeof(half));
    d.vn16 = GpuBuffer(N * 9 * sizeof(half));
    k_fp32_to_fp16_s<<<L(N * 9), 256, 0, s>>>(scene.triangles().fp32(), (half*)d.tri16.data, N * 9);
    k_fp32_to_fp16_s<<<L(N * 9), 256, 0, s>>>(scene.vn().fp32(), (half*)d.vn16.data, N * 9);

    // Precompute world-space centroid (one-time, used per-frame by GPU kernel)
    compute_world_centroid((half*)d.tri16.data, N, d.world_center, s);
    d.world_center_gpu = upload(d.world_center, sizeof(d.world_center));

    // VN: nerf_pe → linear → rmsnorm_f32
    GpuBuffer vn_pe(N * Config::vn_pe_out_dim * sizeof(half));
    nerf_pe_f32(scene.vn().fp32(), (half*)vn_pe.data, 1, N, 9, Config::vn_pe_num_freqs, 0, 5, true, s);
    GpuBuffer vn_emb(N * dim * sizeof(half));
    linear(d.cublas, (half*)vn_pe.data, model.vn_proj_w, model.vn_proj_b, (half*)vn_emb.data, N, dim, Config::vn_pe_out_dim, s);
    vn_pe = GpuBuffer();
    GpuBuffer vn_normed(N * dim * sizeof(float));
    cast_fp16_to_fp32((half*)vn_emb.data, (float*)vn_normed.data, N * dim, s);
    rmsnorm_f32((float*)vn_normed.data, model.vn_enc_norm, (float*)vn_normed.data, N, dim, kEpsFp16, s);
    vn_emb = GpuBuffer();

    // Texture: linear → rmsnorm_f32
    GpuBuffer tex_emb(N * dim * sizeof(half));
    linear(d.cublas, (half*)tex_buf.data, model.tex_enc_w, model.tex_enc_b, (half*)tex_emb.data, N, dim, Config::texture_channels, s);
    tex_buf = GpuBuffer();
    GpuBuffer tex_normed(N * dim * sizeof(float));
    cast_fp16_to_fp32((half*)tex_emb.data, (float*)tex_normed.data, N * dim, s);
    rmsnorm_f32((float*)tex_normed.data, model.tex_enc_norm, (float*)tex_normed.data, N, dim, kEpsFp16, s);
    tex_emb = GpuBuffer();

    // seq = [reg_tokens | tri_emb]
    GpuBuffer tri_emb(N * dim * sizeof(float));
    k_add3_bc_f32_s<<<L(N * dim), 256, 0, s>>>((float*)tri_emb.data, (float*)tex_normed.data,
                                                 (float*)vn_normed.data, model.tri_token, N * dim, dim);
    tex_normed = GpuBuffer(); vn_normed = GpuBuffer();

    GpuBuffer seq(Se * dim * sizeof(float));
    cast_fp16_to_fp32(model.reg_tokens, (float*)seq.data, nr * dim, s);
    CHECK_CUDA(cudaMemcpyAsync((float*)seq.data + nr * dim, tri_emb.data,
                                N * dim * sizeof(float), cudaMemcpyDeviceToDevice, s));
    tri_emb = GpuBuffer();

    // Encoder positions + RoPE
    GpuBuffer enc_pos(Se * 9 * sizeof(half));
    compute_center_pos((half*)d.tri16.data, N, (half*)enc_pos.data, nr, s);
    CHECK_CUDA(cudaMemcpyAsync((half*)enc_pos.data + nr * 9, d.tri16.data,
                                N * 9 * sizeof(half), cudaMemcpyDeviceToDevice, s));
    GpuBuffer enc_cos(Se * hd * sizeof(half)), enc_sin(Se * hd * sizeof(half));
    triangle_rope_freqs((half*)enc_pos.data, model.enc_rope_freqs,
                        (half*)enc_cos.data, (half*)enc_sin.data,
                        1, Se, Config::rope_dim / 2, hd, s);

    // Score buffer (shared encoder/decoder)
    size_t max_scores = std::max((size_t)nh * Se * Se_pad, (size_t)nh * nP * Se_pad);
    d.score = GpuBuffer(max_scores * sizeof(float));

    // ---- Encoder ----
    GpuBuffer temp_enc(Se * dim * sizeof(half));
    GpuBuffer qkv_enc((size_t)Se * std::max(3 * dim, 2 * ffn) * sizeof(half));
    GpuBuffer ffn_enc(Se * std::max(dim, ffn) * sizeof(half));

    printf("Running encoder (%d layers, Se=%d)...\n", Config::num_layers, Se); fflush(stdout);
    GpuTimer enc_timer;
    enc_timer.start(s);
    for (int i = 0; i < Config::num_layers; i++)
        encoder_layer_forward_mixed(d.cublas, model.enc_layers[i],
                              (float*)seq.data, (half*)temp_enc.data,
                              (half*)qkv_enc.data, (half*)ffn_enc.data,
                              (float*)d.score.data, nullptr,
                              1, Se, dim, nh, ffn, (half*)enc_cos.data, (half*)enc_sin.data, s);
    enc_timer.stop(s);
    printf("Encoder done (%.1f ms).\n", enc_timer.ms()); fflush(stdout);

    enc_pos = GpuBuffer(); enc_cos = GpuBuffer(); enc_sin = GpuBuffer();
    temp_enc = GpuBuffer(); qkv_enc = GpuBuffer(); ffn_enc = GpuBuffer();

    // Cast encoder to fp16
    d.ctx_fp16 = GpuBuffer(Se * dim * sizeof(half));
    cast_fp32_to_fp16((float*)seq.data, (half*)d.ctx_fp16.data, Se * dim, s);
    seq = GpuBuffer();

    // ---- KV Cache (pre-padded to Se_pad for GEMM alignment) ----
    size_t kv_sz_pad = (size_t)nh * Se_pad * hd * sizeof(half);
    GpuBuffer kv_proj(2 * Se * dim * sizeof(half));
    d.kv_K.resize(Config::vt_num_layers);
    d.kv_V.resize(Config::vt_num_layers);
    {
        GpuBuffer ctx_normed(Se * dim * sizeof(half));
        // Temp unpadded buffers for precompute_cross_kv
        GpuBuffer kv_tmp_K(Se * dim * sizeof(half));
        GpuBuffer kv_tmp_V(Se * dim * sizeof(half));
        for (int i = 0; i < Config::vt_num_layers; i++) {
            rmsnorm((half*)d.ctx_fp16.data, model.dec_layers[i].kv_norm_gamma,
                    (half*)ctx_normed.data, Se, dim, kEpsFp16, s);
            precompute_cross_kv(d.cublas, model.dec_layers[i].cross_attn,
                                (half*)ctx_normed.data,
                                (half*)kv_tmp_K.data, (half*)kv_tmp_V.data,
                                (half*)kv_proj.data,
                                1, Se, dim, dim, nh, s);
            // Pad [B,H,Se,D] → [B,H,Se_pad,D] with zero padding rows
            d.kv_K[i] = GpuBuffer(kv_sz_pad);
            d.kv_V[i] = GpuBuffer(kv_sz_pad);
            CHECK_CUDA(cudaMemsetAsync(d.kv_K[i].data, 0, kv_sz_pad, s));
            CHECK_CUDA(cudaMemsetAsync(d.kv_V[i].data, 0, kv_sz_pad, s));
            // Copy each head's Se rows into Se_pad-strided layout
            for (int h = 0; h < nh; h++) {
                CHECK_CUDA(cudaMemcpyAsync(
                    (half*)d.kv_K[i].data + h * Se_pad * hd,
                    (half*)kv_tmp_K.data + h * Se * hd,
                    Se * hd * sizeof(half), cudaMemcpyDeviceToDevice, s));
                CHECK_CUDA(cudaMemcpyAsync(
                    (half*)d.kv_V[i].data + h * Se_pad * hd,
                    (half*)kv_tmp_V.data + h * Se * hd,
                    Se * hd * sizeof(half), cudaMemcpyDeviceToDevice, s));
            }
        }
    }
    kv_proj = GpuBuffer();

    // ---- Preallocate per-frame buffers ----
    d.c2w_gpu = GpuBuffer(16 * sizeof(float));
    d.vtri = GpuBuffer(N * 9 * sizeof(half));
    d.vvn = GpuBuffer(N * 9 * sizeof(half));
    d.vpos = GpuBuffer(Se_pad * 9 * sizeof(half));
    CHECK_CUDA(cudaMemsetAsync(d.vpos.data, 0, Se_pad * 9 * sizeof(half), s));
    d.rays_o = GpuBuffer(3 * sizeof(float));
    d.rays_d = GpuBuffer(img_res * img_res * 3 * sizeof(half));
    d.ray_pat = GpuBuffer(nP * Config::ray_encoder_in * sizeof(half));
    d.rtok = GpuBuffer(nP * dim * sizeof(half));

    d.shift_mask = make_shift_mask(pH, pW, ws, Config::shift_size);
    float id_c2w[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    d.id_c2w_buf = upload(id_c2w, sizeof(id_c2w));
    d.fov_gpu = GpuBuffer(sizeof(float));

    int nWin = (pH / ws) * (pW / ws);
    d.self_score = GpuBuffer(nWin * nh * ws * ws * ws * ws * sizeof(float));
    d.dec_temp = GpuBuffer(nP * dim * sizeof(half));
    // proj_buf also serves as swin scratch (Q/K/V/P_half/shifted)
    size_t proj_sz = (size_t)(2 * nP + Se_pad) * dim;
    size_t swin_scratch_sz = (size_t)3 * nP * dim + (size_t)nWin * nh * ws * ws * ws * ws;
    d.dec_proj = GpuBuffer(std::max(proj_sz, swin_scratch_sz) * sizeof(half));
    d.dec_qkv = GpuBuffer(nP * std::max(3 * dim, 2 * ffn) * sizeof(half));
    d.dec_ffn = GpuBuffer(nP * std::max(dim, ffn) * sizeof(half));
    d.dec_win = GpuBuffer(nP * dim * sizeof(half));

    // Q RoPE (camera at origin → constant)
    d.q_cos = GpuBuffer(nP * hd * sizeof(half));
    d.q_sin = GpuBuffer(nP * hd * sizeof(half));
    GpuBuffer ray_pos(nP * 9 * sizeof(half));
    CHECK_CUDA(cudaMemsetAsync(ray_pos.data, 0, nP * 9 * sizeof(half), s));
    triangle_rope_freqs((half*)ray_pos.data, model.vt_rope_freqs,
                        (half*)d.q_cos.data, (half*)d.q_sin.data,
                        1, nP, Config::vt_rope_dim / 2, hd, s);
    ray_pos = GpuBuffer();

    d.k_cos = GpuBuffer(Se_pad * hd * sizeof(half));
    d.k_sin = GpuBuffer(Se_pad * hd * sizeof(half));

    for (auto& b : d.dft) b = GpuBuffer(nP * dim * sizeof(half));
    d.dpt_out = GpuBuffer(Config::dpt_out_dim * img_res * img_res * sizeof(half));
    d.col = GpuBuffer((size_t)dim * 9 * img_res * img_res * sizeof(half));
    d.out_fp32 = GpuBuffer(3 * img_res * img_res * sizeof(float));

    // Warmup: run one frame to JIT-compile cuDNN conv kernels (DPT, SDPA)
    std::vector<float> c2w(16);
    CHECK_CUDA(cudaMemcpy(c2w.data(), scene.c2w().fp32(), 16 * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> fov(1);
    CHECK_CUDA(cudaMemcpy(fov.data(), scene.fov().fp32(), sizeof(float), cudaMemcpyDeviceToHost));
    float fov_rad = fov[0] * (float)M_PI / 180.0f;
    d.render_view(c2w.data(), &fov_rad);
    CHECK_CUDA(cudaStreamSynchronize(s));

    // Capture render_view as CUDA Graph for zero-overhead replay
    CHECK_CUDA(cudaHostAlloc(&d.pinned_c2w, 16 * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&d.pinned_fov, sizeof(float), cudaHostAllocDefault));
    memcpy(d.pinned_c2w, c2w.data(), 16 * sizeof(float));
    *d.pinned_fov = fov_rad;

    CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    d.render_view(d.pinned_c2w, d.pinned_fov);
    auto cap_err = cudaStreamEndCapture(s, &d.graph);
    if (cap_err == cudaSuccess && d.graph) {
        size_t num_nodes = 0;
        cudaGraphGetNodes(d.graph, nullptr, &num_nodes);
        cap_err = cudaGraphInstantiate(&d.graph_exec, d.graph, nullptr, nullptr, 0);
        if (cap_err == cudaSuccess) {
            printf("CUDA Graph: %zu nodes captured\n", num_nodes);
        } else {
            fprintf(stderr, "Graph instantiate failed: %s\n", cudaGetErrorString(cap_err));
            cudaGraphDestroy(d.graph); d.graph = nullptr;
        }
    } else {
        fprintf(stderr, "Graph capture failed: %s\n", cudaGetErrorString(cap_err));
        d.graph = nullptr;
    }

    printf("RenderSession ready.\n"); fflush(stdout);
    return sess;
}

const float* RenderSession::render_frame(const float* h_c2w, float fov_rad) {
    auto& d = *impl_;
    memcpy(d.pinned_c2w, h_c2w, 16 * sizeof(float));
    *d.pinned_fov = fov_rad;
    d.timer.start(d.stream);
    if (d.graph_exec)
        CHECK_CUDA(cudaGraphLaunch(d.graph_exec, d.stream));
    else
        d.render_view(d.pinned_c2w, d.pinned_fov);
    d.timer.stop(d.stream);
    last_ms_ = d.timer.ms();
    return (const float*)d.out_fp32.data;
}

void RenderSession::render_frame_async(const float* h_c2w, float fov_rad) {
    auto& d = *impl_;
    memcpy(d.pinned_c2w, h_c2w, 16 * sizeof(float));
    *d.pinned_fov = fov_rad;
    d.timer.start(d.stream);
    if (d.graph_exec)
        CHECK_CUDA(cudaGraphLaunch(d.graph_exec, d.stream));
    else
        d.render_view(d.pinned_c2w, d.pinned_fov);
    d.timer.stop(d.stream);
    CHECK_CUDA(cudaEventRecord(d.frame_done, d.stream));
    d.async_pending = true;
}

bool RenderSession::is_frame_ready() {
    if (!impl_->async_pending) return true;
    auto err = cudaEventQuery(impl_->frame_done);
    if (err == cudaSuccess) {
        impl_->async_pending = false;
        last_ms_ = impl_->timer.ms();
        return true;
    }
    return false;
}

const float* RenderSession::get_frame_result() {
    auto& d = *impl_;
    if (d.async_pending) {
        CHECK_CUDA(cudaEventSynchronize(d.frame_done));
        d.async_pending = false;
        last_ms_ = d.timer.ms();
    }
    return (const float*)d.out_fp32.data;
}

void RenderSession::download_frame_hwc(float* h_out) {
    auto& d = *impl_;
    int total = 3 * img_res_ * img_res_;
    std::vector<float> chw(total);
    CHECK_CUDA(cudaMemcpy(chw.data(), d.out_fp32.data, total * sizeof(float), cudaMemcpyDeviceToHost));
    for (int y = 0; y < img_res_; y++)
        for (int x = 0; x < img_res_; x++)
            for (int c = 0; c < 3; c++)
                h_out[(y * img_res_ + x) * 3 + c] = chw[c * img_res_ * img_res_ + y * img_res_ + x];
}

RenderSession::~RenderSession() = default;
RenderSession::RenderSession(RenderSession&&) noexcept = default;
RenderSession& RenderSession::operator=(RenderSession&&) noexcept = default;

} // namespace rf
