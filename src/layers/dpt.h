#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace rf {

// Conv2d via im2col + GEMM
// in: [N, Cin, H, W], weight: [Cout, Cin, kH, kW], bias: [Cout] or nullptr
void conv2d(cublasHandle_t handle,
            const half* in, const half* weight, const half* bias,
            half* out, half* col_buf,
            int N, int Cin, int H, int W,
            int Cout, int kH, int kW,
            int stride, int padding,
            cudaStream_t stream = nullptr);

// ConvTranspose2d via col2im + GEMM
void conv_transpose2d(cublasHandle_t handle,
                      const half* in, const half* weight, const half* bias,
                      half* out, half* col_buf,
                      int N, int Cin, int H, int W,
                      int Cout, int kH, int kW,
                      int stride, int padding,
                      cudaStream_t stream = nullptr);

// Bilinear upsample: [N, C, H, W] → [N, C, H_out, W_out] (align_corners=true)
void bilinear_upsample(const half* in, half* out,
                       int N, int C, int H_in, int W_in, int H_out, int W_out,
                       cudaStream_t stream = nullptr);

// SiLU activation in-place
void silu_inplace(half* x, int n, cudaStream_t stream = nullptr);

struct RCUWeights {
    const half* conv1_w; // [F, F, 3, 3]
    const half* conv1_b; // [F]
    const half* conv2_w; // [F, F, 3, 3]
    const half* conv2_b; // [F]
};

struct FFBWeights {
    RCUWeights rcu1;        // used only when has_resconv1
    RCUWeights rcu2;
    const half* out_conv_w; // [Fout, F, 1, 1]
    const half* out_conv_b; // [Fout]
    bool has_resconv1;
};

struct DPTWeights {
    const half* project_w[4];
    const half* project_b[4];
    const half* resize_w[4];
    const half* resize_b[4];
    const half* layer_rn_w[4];
    FFBWeights refinenet[4];
    const half* output_conv1_w;
    const half* output_conv1_b;
    const half* output_conv2a_w;
    const half* output_conv2a_b;
    const half* output_conv2b_w;
    const half* output_conv2b_b;
};

void dpt_forward(cublasHandle_t handle,
                 const DPTWeights& w,
                 const half* features[4],
                 half* out, half* col_buf,
                 int B, int dim, int patch_h, int patch_w, int patch_size,
                 int features_dim, int out_dim,
                 const int out_channels[4],
                 cudaStream_t stream = nullptr);

} // namespace rf
