// Compare C++ intermediates with Python dumps
#include "model/renderformer.h"
#include "pipeline/renderer.h"
#include "io/hdf5_reader.h"
#include "io/safetensors.h"
#include "core/cuda_utils.h"
#include "core/tensor.h"
#include "kernels/rmsnorm.h"
#include "kernels/nerf_pe.h"
#include "kernels/rope.h"
#include "layers/linear.h"
#include "layers/transformer.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

using namespace rf;

static std::vector<float> load_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> data(sz / sizeof(float));
    fread(data.data(), 1, sz, f);
    fclose(f);
    return data;
}

static std::vector<float> gpu_fp16_to_host(const half* gpu, int n) {
    std::vector<half> h(n);
    cudaMemcpy(h.data(), gpu, n * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) out[i] = __half2float(h[i]);
    return out;
}

static std::vector<float> gpu_fp32_to_host(const float* gpu, int n) {
    std::vector<float> out(n);
    cudaMemcpy(out.data(), gpu, n * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

static void compare(const std::vector<float>& a, const std::vector<float>& b,
                    const char* name, int max_print = 10) {
    if (a.size() != b.size()) {
        printf("  %s: SIZE MISMATCH (%zu vs %zu)\n", name, a.size(), b.size());
        return;
    }
    double mse = 0, max_err = 0;
    int max_idx = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i];
        mse += d * d;
        if (std::abs(d) > max_err) { max_err = std::abs(d); max_idx = (int)i; }
    }
    mse /= a.size();
    double rmse = sqrt(mse);

    // Compute relative error: normalized by range of reference
    float ref_min = *std::min_element(b.begin(), b.end());
    float ref_max = *std::max_element(b.begin(), b.end());
    double ref_range = ref_max - ref_min;

    printf("  %s: RMSE=%.6f max_err=%.6f (at %d: cuda=%.6f py=%.6f) ref_range=[%.4f,%.4f]\n",
           name, rmse, max_err, max_idx, a[max_idx], b[max_idx], ref_min, ref_max);

    // Print first few values
    int n_print = std::min((int)a.size(), max_print);
    printf("    first %d: cuda=[", n_print);
    for (int i = 0; i < n_print; i++) printf("%.4f%s", a[i], i < n_print - 1 ? ", " : "");
    printf("]\n    first %d: py  =[", n_print);
    for (int i = 0; i < n_print; i++) printf("%.4f%s", b[i], i < n_print - 1 ? ", " : "");
    printf("]\n");
}

// Kernels from renderer.cu
__global__ void k_fp32_to_fp16_(const float* in, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}
__global__ void k_extract_tex_(const half* in, half* out, int NC, int HW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NC) out[i] = in[i * HW];
}
__global__ void k_log_encode_(half* data, int N, int C, int last_n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * last_n) return;
    int n = i / last_n;
    int idx = n * C + (C - last_n) + (i % last_n);
    data[idx] = __float2half(log10f(__half2float(data[idx]) + 1.0f));
}
__global__ void k_add3_bc_(half* out, const half* a, const half* b, const half* c, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__half2float(a[i]) + __half2float(b[i]) + __half2float(c[i % dim]));
}
__global__ void k_add3_bc_f32_(float* out, const float* a, const float* b, const half* c, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i] + __half2float(c[i % dim]);
}
#define L(n) (((n) + 255) / 256)

static void compute_center_pos(const half* tri_pos_gpu, int N,
                                half* center_out_gpu, int num_reg,
                                cudaStream_t stream) {
    std::vector<half> h_pos(N * 9);
    cudaMemcpyAsync(h_pos.data(), tri_pos_gpu, N * 9 * sizeof(half), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    double mean9[9] = {};
    for (int i = 0; i < N; i++)
        for (int d = 0; d < 9; d++)
            mean9[d] += __half2float(h_pos[i * 9 + d]);
    for (auto& v : mean9) v /= N;
    float centroid[3];
    for (int j = 0; j < 3; j++)
        centroid[j] = (float)((mean9[j] + mean9[3 + j] + mean9[6 + j]) / 3.0);
    std::vector<half> h_center(num_reg * 9);
    for (int r = 0; r < num_reg; r++)
        for (int d = 0; d < 9; d++)
            h_center[r * 9 + d] = __float2half(centroid[d % 3]);
    cudaMemcpyAsync(center_out_gpu, h_center.data(), num_reg * 9 * sizeof(half), cudaMemcpyHostToDevice, stream);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: test_compare <model.safetensors> <scene.h5>\n");
        return 1;
    }

    printf("Loading model: %s\n", argv[1]);
    auto sf = SafeTensorsFile::load(argv[1]);
    RenderFormerModel model;
    model.load(sf);

    printf("Loading scene: %s\n", argv[2]);
    auto scene = Scene::load(argv[2]);

    CudaStream stream;
    CublasHandle cublas;
    cublas.set_stream(stream);

    const int N = (int)scene.num_triangles;
    constexpr int dim = Config::latent_dim;
    constexpr int nh = Config::num_heads;
    constexpr int hd = Config::head_dim;
    constexpr int nr = Config::num_register_tokens;
    constexpr int ffn = Config::ffn_hidden;
    const int Se = nr + N;

    printf("N=%d, Se=%d\n", N, Se);

    auto chk = [&](const char* msg) {
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaGetLastError());
        printf("  [OK] %s\n", msg); fflush(stdout);
    };

    // ---- Replicate preprocessing from renderer ----

    // Extract texture [N, 13]
    GpuBuffer tex_buf(N * 13 * sizeof(half));
    k_extract_tex_<<<L(N * 13), 256, 0, stream>>>(
        scene.texture().fp16(), (half*)tex_buf.data, N * 13, 32 * 32);
    k_log_encode_<<<L(N * 3), 256, 0, stream>>>((half*)tex_buf.data, N, 13, 3);
    chk("tex extract");

    // Convert tri/vn
    GpuBuffer tri16(N * 9 * sizeof(half)), vn16(N * 9 * sizeof(half));
    k_fp32_to_fp16_<<<L(N * 9), 256, 0, stream>>>(scene.triangles().fp32(), (half*)tri16.data, N * 9);
    k_fp32_to_fp16_<<<L(N * 9), 256, 0, stream>>>(scene.vn().fp32(), (half*)vn16.data, N * 9);
    chk("fp32→fp16");

    // VN: nerf_pe (fp32 input) → linear (fp16) → cast → rmsnorm_f32
    GpuBuffer vn_pe(N * Config::vn_pe_out_dim * sizeof(half));
    nerf_pe_f32(scene.vn().fp32(), (half*)vn_pe.data, 1, N, 9, Config::vn_pe_num_freqs, 0.0f, 5.0f, true, stream);
    GpuBuffer vn_emb(N * dim * sizeof(half));
    linear(cublas, (half*)vn_pe.data, model.vn_proj_w, model.vn_proj_b,
           (half*)vn_emb.data, N, dim, Config::vn_pe_out_dim, stream);
    chk("vn encode");

    // Compare vn_pe, vn_emb
    {
        auto py_vn_pe = load_bin("output/dump/vn_pe.bin");
        auto cu_vn_pe = gpu_fp16_to_host((half*)vn_pe.data, N * Config::vn_pe_out_dim);
        compare(cu_vn_pe, py_vn_pe, "vn_pe");
        auto py_vn_emb = load_bin("output/dump/vn_emb.bin");
        auto cu_vn_emb = gpu_fp16_to_host((half*)vn_emb.data, N * dim);
        compare(cu_vn_emb, py_vn_emb, "vn_emb");
    }

    // RMSNorm in fp32 (matches PyTorch: fp32 gamma → fp32 output)
    GpuBuffer vn_normed(N * dim * sizeof(float));
    cast_fp16_to_fp32((half*)vn_emb.data, (float*)vn_normed.data, N * dim, stream);
    rmsnorm_f32((float*)vn_normed.data, model.vn_enc_norm, (float*)vn_normed.data, N, dim, kEpsFp16, stream);
    vn_emb = GpuBuffer();
    chk("vn rmsnorm (fp32)");

    // Texture: linear (fp16) → cast → rmsnorm_f32
    GpuBuffer tex_emb(N * dim * sizeof(half));
    linear(cublas, (half*)tex_buf.data, model.tex_enc_w, model.tex_enc_b,
           (half*)tex_emb.data, N, dim, Config::texture_channels, stream);
    chk("tex encode");

    // Compare tex_emb
    {
        auto py_tex_emb = load_bin("output/dump/tex_emb.bin");
        auto cu_tex_emb = gpu_fp16_to_host((half*)tex_emb.data, N * dim);
        compare(cu_tex_emb, py_tex_emb, "tex_emb");
    }

    GpuBuffer tex_normed(N * dim * sizeof(float));
    cast_fp16_to_fp32((half*)tex_emb.data, (float*)tex_normed.data, N * dim, stream);
    rmsnorm_f32((float*)tex_normed.data, model.tex_enc_norm, (float*)tex_normed.data, N, dim, kEpsFp16, stream);
    tex_emb = GpuBuffer();
    chk("tex rmsnorm (fp32)");

    // Compare tex_normed, vn_normed (both fp32 now)
    {
        auto py_tex_n = load_bin("output/dump/tex_normed.bin");
        auto cu_tex_n = gpu_fp32_to_host((float*)tex_normed.data, N * dim);
        compare(cu_tex_n, py_tex_n, "tex_normed");

        auto py_vn_n = load_bin("output/dump/vn_normed.bin");
        auto cu_vn_n = gpu_fp32_to_host((float*)vn_normed.data, N * dim);
        compare(cu_vn_n, py_vn_n, "vn_normed");
    }

    // tri_emb_f32 = tex_normed_f32 + vn_normed_f32 + broadcast(tri_token fp16→f32)
    GpuBuffer tri_emb(N * dim * sizeof(float));
    k_add3_bc_f32_<<<L(N * dim), 256, 0, stream>>>(
        (float*)tri_emb.data, (float*)tex_normed.data, (float*)vn_normed.data,
        model.tri_token, N * dim, dim);

    // seq_f32 = [reg_tokens(fp16→fp32) | tri_emb_f32]
    GpuBuffer seq(Se * dim * sizeof(float));
    cast_fp16_to_fp32(model.reg_tokens, (float*)seq.data, nr * dim, stream);
    cudaMemcpyAsync((float*)seq.data + nr * dim, tri_emb.data,
                    N * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    chk("construct_seq (fp32)");

    // Compare seq (both fp32)
    printf("\n=== Compare seq ===\n");
    auto py_seq = load_bin("output/dump/seq.bin");
    auto cu_seq = gpu_fp32_to_host((float*)seq.data, Se * dim);
    compare(cu_seq, py_seq, "seq");

    // Encoder positions
    GpuBuffer enc_pos(Se * 9 * sizeof(half));
    compute_center_pos((half*)tri16.data, N, (half*)enc_pos.data, nr, stream);
    cudaMemcpyAsync((half*)enc_pos.data + nr * 9, tri16.data, N * 9 * sizeof(half), cudaMemcpyDeviceToDevice, stream);

    printf("\n=== Compare positions ===\n");
    auto py_pos = load_bin("output/dump/positions.bin");
    auto cu_pos = gpu_fp16_to_host((half*)enc_pos.data, Se * 9);
    compare(cu_pos, py_pos, "positions");

    // Encoder RoPE
    GpuBuffer enc_cos(Se * hd * sizeof(half)), enc_sin(Se * hd * sizeof(half));
    triangle_rope_freqs((half*)enc_pos.data, model.enc_rope_freqs,
                        (half*)enc_cos.data, (half*)enc_sin.data,
                        1, Se, Config::rope_dim / 2, hd, stream);
    chk("enc rope");

    printf("\n=== Compare RoPE ===\n");
    auto py_cos = load_bin("output/dump/enc_rope_cos.bin");
    auto cu_cos = gpu_fp16_to_host((half*)enc_cos.data, Se * hd);
    compare(cu_cos, py_cos, "rope_cos");

    // Mixed-precision encoder buffers (fp16 sub-layers)
    GpuBuffer temp(Se * dim * sizeof(half));
    size_t work_sz = (size_t)Se * std::max(3 * dim, 2 * ffn) * sizeof(half);
    GpuBuffer qkv(work_sz);
    GpuBuffer ffn_buf(Se * std::max(dim, ffn) * sizeof(half));
    size_t max_scores = (size_t)nh * Se * Se;
    GpuBuffer score(max_scores * sizeof(float));

    // Test mixed-precision encoder layer 0 with C++ seq
    {
        // Make a copy since encoder modifies seq in-place
        GpuBuffer seq_copy(Se * dim * sizeof(float));
        cudaMemcpyAsync(seq_copy.data, seq.data, Se * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        encoder_layer_forward_mixed(cublas, model.enc_layers[0],
                              (float*)seq_copy.data, (half*)temp.data,
                              (half*)qkv.data, (half*)ffn_buf.data,
                              (float*)score.data, nullptr,
                              1, Se, dim, nh, ffn, (half*)enc_cos.data, (half*)enc_sin.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        chk("enc layer 0 mixed (C++ seq)");

        printf("\n=== Compare enc_layer_0 (mixed, C++ seq) ===\n");
        auto py_enc0 = load_bin("output/dump/enc_layer_0.bin");
        auto cu_enc0 = gpu_fp32_to_host((float*)seq_copy.data, Se * dim);
        compare(cu_enc0, py_enc0, "enc_layer_0 (mixed, C++ seq)");
    }

    // Test mixed-precision encoder with Python seq+rope
    printf("\n=== Testing mixed encoder with Python seq ===\n");
    {
        GpuBuffer py_seq_gpu(Se * dim * sizeof(float));
        CHECK_CUDA(cudaMemcpy(py_seq_gpu.data, py_seq.data(), Se * dim * sizeof(float), cudaMemcpyHostToDevice));

        auto py_sin = load_bin("output/dump/enc_rope_sin.bin");
        GpuBuffer py_cos_gpu(Se * hd * sizeof(half));
        GpuBuffer py_sin_gpu(Se * hd * sizeof(half));
        {
            std::vector<half> h16(Se * hd);
            for (int i = 0; i < Se * hd; i++) h16[i] = __float2half(py_cos[i]);
            CHECK_CUDA(cudaMemcpy(py_cos_gpu.data, h16.data(), Se * hd * sizeof(half), cudaMemcpyHostToDevice));
            for (int i = 0; i < Se * hd; i++) h16[i] = __float2half(py_sin[i]);
            CHECK_CUDA(cudaMemcpy(py_sin_gpu.data, h16.data(), Se * hd * sizeof(half), cudaMemcpyHostToDevice));
        }

        encoder_layer_forward_mixed(cublas, model.enc_layers[0],
                              (float*)py_seq_gpu.data, (half*)temp.data,
                              (half*)qkv.data, (half*)ffn_buf.data,
                              (float*)score.data, nullptr,
                              1, Se, dim, nh, ffn, (half*)py_cos_gpu.data, (half*)py_sin_gpu.data, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        chk("enc layer 0 mixed (Python seq+rope)");

        auto py_enc0 = load_bin("output/dump/enc_layer_0.bin");
        auto cu_enc0 = gpu_fp32_to_host((float*)py_seq_gpu.data, Se * dim);
        compare(cu_enc0, py_enc0, "enc_layer_0 (mixed, Python seq+rope)");
    }

    // All encoder layers with C++ seq (mixed precision)
    {
        for (int i = 0; i < Config::num_layers; i++) {
            encoder_layer_forward_mixed(cublas, model.enc_layers[i],
                                  (float*)seq.data, (half*)temp.data,
                                  (half*)qkv.data, (half*)ffn_buf.data,
                                  (float*)score.data, nullptr,
                                  1, Se, dim, nh, ffn, (half*)enc_cos.data, (half*)enc_sin.data, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
        }
        chk("all encoder layers (mixed)");

        printf("\n=== Compare encoder_out ===\n");
        auto py_encout = load_bin("output/dump/encoder_out.bin");
        auto cu_encout = gpu_fp32_to_host((float*)seq.data, Se * dim);
        compare(cu_encout, py_encout, "encoder_out");
    }

    printf("\nDone.\n");
    return 0;
}
