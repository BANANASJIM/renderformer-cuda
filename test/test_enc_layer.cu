// Fine-grained encoder layer 0 comparison
#include "model/renderformer.h"
#include "io/safetensors.h"
#include "core/cuda_utils.h"
#include "core/tensor.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "layers/linear.h"
#include "layers/attention.h"
#include "layers/transformer.h"
#include "kernels/activations.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace rf;

static std::vector<float> load_bin(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> d(sz / sizeof(float));
    fread(d.data(), 1, sz, f); fclose(f); return d;
}

static std::vector<float> gpu_to_host(const half* gpu, int n) {
    std::vector<half> h(n);
    cudaMemcpy(h.data(), gpu, n * sizeof(half), cudaMemcpyDeviceToHost);
    std::vector<float> out(n);
    for (int i = 0; i < n; i++) out[i] = __half2float(h[i]);
    return out;
}

static void compare(const std::vector<float>& a, const std::vector<float>& b, const char* name) {
    if (a.size() != b.size()) { printf("  %s: SIZE MISMATCH (%zu vs %zu)\n", name, a.size(), b.size()); return; }
    double mse = 0, max_err = 0; int max_idx = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i]; mse += d * d;
        if (std::abs(d) > max_err) { max_err = std::abs(d); max_idx = (int)i; }
    }
    mse /= a.size();
    printf("  %-20s RMSE=%.4f max=%.4f @%d (cu=%.4f py=%.4f)\n",
           name, sqrt(mse), max_err, max_idx, a[max_idx], b[max_idx]);
}

static GpuBuffer fp32_to_gpu_fp16(const std::vector<float>& fp32) {
    int n = fp32.size();
    std::vector<half> fp16(n);
    for (int i = 0; i < n; i++) fp16[i] = __float2half(fp32[i]);
    GpuBuffer buf(n * sizeof(half));
    cudaMemcpy(buf.data, fp16.data(), n * sizeof(half), cudaMemcpyHostToDevice);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "Usage: test_enc_layer <model.safetensors>\n"); return 1; }

    auto sf = SafeTensorsFile::load(argv[1]);
    RenderFormerModel model; model.load(sf);

    CudaStream stream;
    CublasHandle cublas;
    cublas.set_stream(stream);

    constexpr int dim = Config::latent_dim, nh = Config::num_heads, hd = Config::head_dim;
    constexpr int ffn_h = Config::ffn_hidden;
    constexpr int Se = 5665;

    auto py_seq_f = load_bin("output/dump/seq.bin");
    auto py_cos_f = load_bin("output/dump/enc_rope_cos.bin");
    auto py_sin_f = load_bin("output/dump/enc_rope_sin.bin");
    auto py_seq = fp32_to_gpu_fp16(py_seq_f);
    auto py_cos = fp32_to_gpu_fp16(py_cos_f);
    auto py_sin = fp32_to_gpu_fp16(py_sin_f);
    auto& w = model.enc_layers[0];

    printf("=== Step-by-step encoder layer 0 (using Python seq) ===\n");

    // 1. RMSNorm
    GpuBuffer normed(Se * dim * sizeof(half));
    rmsnorm((half*)py_seq.data, w.query_norm_gamma, (half*)normed.data, Se, dim, kEpsFp16, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host((half*)normed.data, Se * dim), load_bin("output/dump/l0_normed.bin"), "normed");

    // 2. QKV projection
    GpuBuffer qkv_buf(Se * 3 * dim * sizeof(half));
    linear(cublas, (half*)normed.data, w.attn.in_proj_w, nullptr,
           (half*)qkv_buf.data, Se, 3 * dim, dim, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host((half*)qkv_buf.data, Se * 3 * dim), load_bin("output/dump/l0_qkv.bin"), "qkv");

    // 3. QK norm (in-place)
    half* Q = (half*)qkv_buf.data;
    half* K = Q + Se * dim;
    rmsnorm(Q, w.attn.q_norm_gamma, Q, Se, dim, kEpsFp16, stream);
    rmsnorm(K, w.attn.k_norm_gamma, K, Se, dim, kEpsFp16, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host(Q, Se * dim), load_bin("output/dump/l0_q_normed.bin"), "q_normed");
    compare(gpu_to_host(K, Se * dim), load_bin("output/dump/l0_k_normed.bin"), "k_normed");

    // 4-7: self_attention (full) — pass Python seq through the full function
    printf("\n--- Full self_attention ---\n");
    GpuBuffer attn_out(Se * dim * sizeof(half));
    GpuBuffer qkv_buf2(Se * 3 * dim * sizeof(half));
    GpuBuffer score(nh * Se * Se * sizeof(float));
    self_attention(cublas, w.attn,
                   (half*)normed.data, (half*)attn_out.data,
                   (half*)qkv_buf2.data, (float*)score.data,
                   nullptr, 1, Se, dim, nh,
                   (half*)py_cos.data, (half*)py_sin.data, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host((half*)attn_out.data, Se * dim), load_bin("output/dump/l0_out_proj.bin"), "out_proj");

    // After residual
    printf("\n--- After attention residual ---\n");
    GpuBuffer x(Se * dim * sizeof(half));
    CHECK_CUDA(cudaMemcpy(x.data, py_seq.data, Se * dim * sizeof(half), cudaMemcpyDeviceToDevice));
    residual_add((half*)x.data, (half*)attn_out.data, Se * dim, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host((half*)x.data, Se * dim), load_bin("output/dump/l0_after_attn_residual.bin"), "after_attn_res");

    // FFN norm
    GpuBuffer ffn_normed(Se * dim * sizeof(half));
    rmsnorm((half*)x.data, w.ffn_norm_gamma, (half*)ffn_normed.data, Se, dim, kEpsFp16, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host((half*)ffn_normed.data, Se * dim), load_bin("output/dump/l0_ffn_normed.bin"), "ffn_normed");

    // FFN
    GpuBuffer ffn_out(Se * dim * sizeof(half));
    GpuBuffer ffn_work(Se * 2 * ffn_h * sizeof(half));
    ffn_swiglu(cublas, w.ffn, (half*)ffn_normed.data, (half*)ffn_out.data,
               (half*)ffn_work.data, Se, dim, ffn_h, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    compare(gpu_to_host((half*)ffn_out.data, Se * dim), load_bin("output/dump/l0_ffn_out.bin"), "ffn_out");

    printf("\nDone.\n");
    return 0;
}
