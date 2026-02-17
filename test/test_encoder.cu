#include "model/renderformer.h"
#include "layers/linear.h"
#include "layers/transformer.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "core/cuda_utils.h"
#include "core/tensor.h"
#include "io/safetensors.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

using namespace rf;

static void print_mem() {
    size_t free_b, total_b;
    cudaMemGetInfo(&free_b, &total_b);
    printf("  GPU mem: %.0f MB free / %.0f MB total\n", free_b / 1e6, total_b / 1e6);
}

// Fill buffer with small random fp16 values
static void fill_random(half* gpu, int n) {
    std::vector<half> h(n);
    for (int i = 0; i < n; i++) h[i] = __float2half(0.01f * (rand() % 100 - 50));
    CHECK_CUDA(cudaMemcpy(gpu, h.data(), n * sizeof(half), cudaMemcpyHostToDevice));
}

// Check for NaN/Inf in gpu fp16 buffer
static bool check_finite(const half* gpu, int n, const char* name) {
    std::vector<half> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), gpu, n * sizeof(half), cudaMemcpyDeviceToHost));
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < n; i++) {
        float v = __half2float(h[i]);
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
    }
    if (nan_count || inf_count) {
        printf("  FAIL %s: %d NaN, %d Inf out of %d\n", name, nan_count, inf_count, n);
        return false;
    }
    printf("  OK %s: %d values, all finite\n", name, n);
    return true;
}

#define RUN(name, code) do { \
    printf("\n=== %s ===\n", name); \
    print_mem(); \
    code; \
    CHECK_CUDA(cudaStreamSynchronize(stream)); \
    CHECK_CUDA(cudaGetLastError()); \
    printf("  PASS\n"); \
} while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_encoder <model.safetensors>\n");
        return 1;
    }

    printf("Loading model: %s\n", argv[1]);
    auto sf = SafeTensorsFile::load(argv[1]);
    RenderFormerModel model;
    model.load(sf);

    CudaStream stream;
    CublasHandle cublas;
    cublas.set_stream(stream);

    constexpr int dim = Config::latent_dim;
    constexpr int nh = Config::num_heads;
    constexpr int hd = Config::head_dim;
    constexpr int ffn_h = Config::ffn_hidden;
    constexpr int nr = Config::num_register_tokens;

    // Use a small sequence for testing (avoid OOM)
    const int N = 64;
    const int Se = nr + N; // 96

    printf("Test dims: Se=%d, dim=%d, nh=%d, hd=%d, ffn=%d\n", Se, dim, nh, hd, ffn_h);
    print_mem();

    // ---- Test 1: Linear standalone ----
    RUN("Linear (no bias)", {
        GpuBuffer x(Se * dim * sizeof(half));
        GpuBuffer y(Se * 3 * dim * sizeof(half));
        fill_random((half*)x.data, Se * dim);
        linear(cublas, (half*)x.data, model.enc_layers[0].attn.in_proj_w, nullptr,
               (half*)y.data, Se, 3 * dim, dim, stream);
    });

    // ---- Test 2: RMSNorm → Linear ----
    RUN("RMSNorm -> Linear", {
        GpuBuffer x(Se * dim * sizeof(half));
        GpuBuffer normed(Se * dim * sizeof(half));
        GpuBuffer y(Se * 3 * dim * sizeof(half));
        fill_random((half*)x.data, Se * dim);
        rmsnorm((half*)x.data, model.enc_layers[0].query_norm_gamma,
                (half*)normed.data, Se, dim, kEpsFp16, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        check_finite((half*)normed.data, Se * dim, "rmsnorm output");
        linear(cublas, (half*)normed.data, model.enc_layers[0].attn.in_proj_w, nullptr,
               (half*)y.data, Se, 3 * dim, dim, stream);
    });

    // ---- Test 3: Self-attention (small Se) ----
    RUN("Self-attention (small Se)", {
        GpuBuffer x(Se * dim * sizeof(half));
        GpuBuffer out(Se * dim * sizeof(half));
        GpuBuffer qkv(Se * 3 * dim * sizeof(half));
        GpuBuffer score(nh * Se * Se * sizeof(float));
        fill_random((half*)x.data, Se * dim);
        self_attention(cublas, model.enc_layers[0].attn,
                       (half*)x.data, (half*)out.data,
                       (half*)qkv.data, (float*)score.data,
                       nullptr, 1, Se, dim, nh, nullptr, nullptr, stream);
    });

    // ---- Test 4: FFN SwiGLU (check buffer sizing) ----
    RUN("FFN SwiGLU", {
        GpuBuffer x(Se * dim * sizeof(half));
        GpuBuffer out(Se * dim * sizeof(half));
        // buf needs M * 2 * ffn_hidden
        size_t buf_sz = (size_t)Se * 2 * ffn_h * sizeof(half);
        printf("  FFN buf needs: %zu bytes (Se*2*ffn=%d)\n", buf_sz, Se * 2 * ffn_h);
        printf("  qkv buf would be: %zu bytes (Se*3*dim=%d)\n",
               (size_t)Se * 3 * dim * sizeof(half), Se * 3 * dim);
        GpuBuffer buf(buf_sz);
        fill_random((half*)x.data, Se * dim);
        ffn_swiglu(cublas, model.enc_layers[0].ffn,
                   (half*)x.data, (half*)out.data, (half*)buf.data,
                   Se, dim, ffn_h, stream);
    });

    // ---- Test 5: Encoder layer (small Se) ----
    RUN("Encoder layer (small Se)", {
        GpuBuffer x(Se * dim * sizeof(half));
        GpuBuffer temp(Se * dim * sizeof(half));
        // Correctly sized: max(3*dim, 2*ffn_hidden) = max(3072, 8192) = 8192
        size_t work_sz = (size_t)Se * std::max(3 * dim, 2 * ffn_h) * sizeof(half);
        GpuBuffer qkv(work_sz);
        GpuBuffer ffn_buf(Se * ffn_h * sizeof(half));
        GpuBuffer score(nh * Se * Se * sizeof(float));
        fill_random((half*)x.data, Se * dim);
        encoder_layer_forward(cublas, model.enc_layers[0],
                              (half*)x.data, (half*)temp.data,
                              (half*)qkv.data, (half*)ffn_buf.data,
                              (float*)score.data, nullptr,
                              1, Se, dim, nh, ffn_h, nullptr, nullptr, stream);
        check_finite((half*)x.data, Se * dim, "encoder layer output");
    });

    // ---- Test 6: Encoder layer with REAL sequence length ----
    // Only run if we have enough memory
    {
        const int Se_real = 5665; // typical for cbox scene
        size_t score_sz = (size_t)nh * Se_real * Se_real * sizeof(float);
        size_t need = score_sz + Se_real * dim * 3 * sizeof(half) + Se_real * 2 * ffn_h * sizeof(half);

        size_t free_b, total_b;
        cudaMemGetInfo(&free_b, &total_b);
        printf("\n=== Encoder layer (real Se=%d) ===\n", Se_real);
        printf("  Need ~%.0f MB, have %.0f MB free\n", need / 1e6, free_b / 1e6);

        if (free_b > need + 500'000'000) { // 500MB headroom
            GpuBuffer x(Se_real * dim * sizeof(half));
            GpuBuffer temp(Se_real * dim * sizeof(half));
            size_t work_sz = (size_t)Se_real * std::max(3 * dim, 2 * ffn_h) * sizeof(half);
            GpuBuffer qkv(work_sz);
            GpuBuffer ffn_buf(Se_real * ffn_h * sizeof(half));
            GpuBuffer score(score_sz);
            fill_random((half*)x.data, Se_real * dim);
            print_mem();

            encoder_layer_forward(cublas, model.enc_layers[0],
                                  (half*)x.data, (half*)temp.data,
                                  (half*)qkv.data, (half*)ffn_buf.data,
                                  (float*)score.data, nullptr,
                                  1, Se_real, dim, nh, ffn_h,
                                  nullptr, nullptr, stream);
            CHECK_CUDA(cudaStreamSynchronize(stream));
            CHECK_CUDA(cudaGetLastError());
            check_finite((half*)x.data, Se_real * dim, "encoder layer (real Se)");
            printf("  PASS\n");
        } else {
            printf("  SKIP: not enough memory\n");
        }
    }

    printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
