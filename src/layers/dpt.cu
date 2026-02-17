#include "layers/dpt.h"
#include "layers/linear.h"
#include "layers/transformer.h"
#include "core/cuda_utils.h"
#include <cudnn.h>
#include <map>

namespace rf {

namespace {

cudnnHandle_t get_cudnn() {
    static cudnnHandle_t h = nullptr;
    if (!h) CHECK_CUDNN(cudnnCreate(&h));
    return h;
}

// Cached cuDNN convolution plan (descriptors + algorithm)
struct CudnnConvPlan {
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    size_t ws_size;
    bool has_bias_desc;
};

// Cached cuDNN transpose conv (backward data) plan
struct CudnnConvTransposePlan {
    cudnnTensorDescriptor_t dy_desc, dx_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionBwdDataAlgo_t algo;
    size_t ws_size;
    bool has_bias_desc;
};

using ConvKey = std::tuple<int,int,int,int,int,int,int,int,int,bool>;

std::map<ConvKey, CudnnConvPlan>& fwd_cache() {
    static std::map<ConvKey, CudnnConvPlan> c;
    return c;
}

std::map<ConvKey, CudnnConvTransposePlan>& bwd_cache() {
    static std::map<ConvKey, CudnnConvTransposePlan> c;
    return c;
}

} // anonymous namespace

// ---- col2im kernel (for ConvTranspose2d) ----

__global__ void col2im_kernel(const half* __restrict__ col,
                               half* __restrict__ out,
                               int Cout, int H_out, int W_out,
                               int kH, int kW, int stride, int pad,
                               int H_in, int W_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Cout * H_out * W_out;
    int n = blockIdx.y;
    if (idx >= total) return;

    int w_o = idx % W_out;
    int rem = idx / W_out;
    int h_o = rem % H_out;
    int c = rem / H_out;

    float sum = 0.0f;
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            int h_tmp = h_o + pad - kh;
            int w_tmp = w_o + pad - kw;
            if (h_tmp % stride != 0 || w_tmp % stride != 0) continue;
            int h_i = h_tmp / stride;
            int w_i = w_tmp / stride;
            if (h_i < 0 || h_i >= H_in || w_i < 0 || w_i >= W_in) continue;

            int col_row = (c * kH + kh) * kW + kw;
            sum += __half2float(col[(n * Cout * kH * kW + col_row) * H_in * W_in + h_i * W_in + w_i]);
        }
    }
    out[(n * Cout + c) * H_out * W_out + h_o * W_out + w_o] = __float2half(sum);
}

// ---- add_bias_nchw: add bias[C] to [N, C, H, W] ----

__global__ void add_bias_nchw_kernel(half* __restrict__ out, const half* __restrict__ bias,
                                      int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    int total = C * HW;
    if (idx >= total) return;
    int c = idx / HW;
    out[n * total + idx] = __float2half(__half2float(out[n * total + idx]) + __half2float(bias[c]));
}

// ---- Conv2d (cuDNN implicit GEMM for kH>1, cuBLAS GEMM for 1x1) ----

void conv2d(cublasHandle_t handle,
            const half* in, const half* weight, const half* bias,
            half* out, half* col_buf,
            int N, int Cin, int H, int W,
            int Cout, int kH, int kW,
            int stride, int padding,
            cudaStream_t stream) {
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    if (kH == 1 && kW == 1 && stride == 1 && padding == 0) {
        cublasSetStream(handle, stream);
        for (int n = 0; n < N; n++) {
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUBLAS(cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                H_out * W_out, Cout, Cin,
                &alpha,
                in + n * Cin * H * W, CUDA_R_16F, H * W,
                weight, CUDA_R_16F, Cin,
                &beta,
                out + n * Cout * H_out * W_out, CUDA_R_16F, H_out * W_out,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        }
    } else {
        auto cudnn = get_cudnn();
        cudnnSetStream(cudnn, stream);

        ConvKey key{N, Cin, H, W, Cout, kH, kW, stride, padding, bias != nullptr};
        auto& cache = fwd_cache();
        auto it = cache.find(key);
        if (it == cache.end()) {
            CudnnConvPlan p{};
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&p.x_desc));
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&p.y_desc));
            CHECK_CUDNN(cudnnCreateFilterDescriptor(&p.w_desc));
            CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&p.conv_desc));

            CHECK_CUDNN(cudnnSetTensor4dDescriptor(p.x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, Cin, H, W));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(p.y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, Cout, H_out, W_out));
            CHECK_CUDNN(cudnnSetFilter4dDescriptor(p.w_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, Cout, Cin, kH, kW));
            CHECK_CUDNN(cudnnSetConvolution2dDescriptor(p.conv_desc, padding, padding, stride, stride, 1, 1,
                                                         CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
            CHECK_CUDNN(cudnnSetConvolutionMathType(p.conv_desc, CUDNN_TENSOR_OP_MATH));

            int returned = 0;
            cudnnConvolutionFwdAlgoPerf_t perf[4];
            CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, p.x_desc, p.w_desc, p.conv_desc, p.y_desc,
                                                                4, &returned, perf));
            p.algo = perf[0].algo;
            CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, p.x_desc, p.w_desc, p.conv_desc, p.y_desc,
                                                                 p.algo, &p.ws_size));
            p.has_bias_desc = (bias != nullptr);
            if (p.has_bias_desc) {
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&p.b_desc));
                CHECK_CUDNN(cudnnSetTensor4dDescriptor(p.b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, Cout, 1, 1));
            }
            it = cache.emplace(key, p).first;
        }

        auto& p = it->second;
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, p.x_desc, in, p.w_desc, weight,
                                             p.conv_desc, p.algo, col_buf, p.ws_size,
                                             &beta, p.y_desc, out));
        if (bias) {
            float one = 1.0f;
            CHECK_CUDNN(cudnnAddTensor(cudnn, &one, p.b_desc, bias, &one, p.y_desc, out));
        }
        return;
    }

    if (bias) {
        int total = Cout * H_out * W_out;
        int threads = 256;
        add_bias_nchw_kernel<<<dim3((total + threads - 1) / threads, N), threads, 0, stream>>>(
            out, bias, Cout, H_out * W_out);
    }
}

// ---- ConvTranspose2d (cuDNN backward data) ----

void conv_transpose2d(cublasHandle_t handle,
                      const half* in, const half* weight, const half* bias,
                      half* out, half* col_buf,
                      int N, int Cin, int H, int W,
                      int Cout, int kH, int kW,
                      int stride, int padding,
                      cudaStream_t stream) {
    int H_out = (H - 1) * stride - 2 * padding + kH;
    int W_out = (W - 1) * stride - 2 * padding + kW;

    auto cudnn = get_cudnn();
    cudnnSetStream(cudnn, stream);

    ConvKey key{N, Cin, H, W, Cout, kH, kW, stride, padding, bias != nullptr};
    auto& cache = bwd_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        CudnnConvTransposePlan p{};
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&p.dy_desc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&p.dx_desc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&p.w_desc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&p.conv_desc));

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(p.dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, Cin, H, W));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(p.dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, Cout, H_out, W_out));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(p.w_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, Cin, Cout, kH, kW));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(p.conv_desc, padding, padding, stride, stride, 1, 1,
                                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolutionMathType(p.conv_desc, CUDNN_TENSOR_OP_MATH));

        int returned = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perf[4];
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn, p.w_desc, p.dy_desc, p.conv_desc, p.dx_desc,
                                                                 4, &returned, perf));
        p.algo = perf[0].algo;
        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, p.w_desc, p.dy_desc, p.conv_desc, p.dx_desc,
                                                                  p.algo, &p.ws_size));
        p.has_bias_desc = (bias != nullptr);
        if (p.has_bias_desc) {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&p.b_desc));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(p.b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, Cout, 1, 1));
        }
        it = cache.emplace(key, p).first;
    }

    auto& p = it->second;
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn, &alpha, p.w_desc, weight, p.dy_desc, in,
                                              p.conv_desc, p.algo, col_buf, p.ws_size,
                                              &beta, p.dx_desc, out));
    if (bias) {
        float one = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(cudnn, &one, p.b_desc, bias, &one, p.dx_desc, out));
    }
}

// ---- Bilinear upsample ----

__global__ void bilinear_upsample_kernel(const half* __restrict__ in,
                                          half* __restrict__ out,
                                          int C, int H_in, int W_in,
                                          int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    int total = C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int rem = idx / W_out;
    int h_out = rem % H_out;
    int c = rem / H_out;

    float h_scale = (H_out > 1) ? (float)(H_in - 1) / (H_out - 1) : 0.0f;
    float w_scale = (W_out > 1) ? (float)(W_in - 1) / (W_out - 1) : 0.0f;
    float h_in_f = h_out * h_scale;
    float w_in_f = w_out * w_scale;

    int h0 = (int)h_in_f;
    int w0 = (int)w_in_f;
    int h1 = min(h0 + 1, H_in - 1);
    int w1 = min(w0 + 1, W_in - 1);
    float hf = h_in_f - h0;
    float wf = w_in_f - w0;

    const half* plane = in + (n * C + c) * H_in * W_in;
    float v00 = __half2float(plane[h0 * W_in + w0]);
    float v01 = __half2float(plane[h0 * W_in + w1]);
    float v10 = __half2float(plane[h1 * W_in + w0]);
    float v11 = __half2float(plane[h1 * W_in + w1]);

    float val = v00 * (1 - hf) * (1 - wf) + v01 * (1 - hf) * wf +
                v10 * hf * (1 - wf) + v11 * hf * wf;

    out[(n * C + c) * H_out * W_out + h_out * W_out + w_out] = __float2half(val);
}

void bilinear_upsample(const half* in, half* out,
                       int N, int C, int H_in, int W_in, int H_out, int W_out,
                       cudaStream_t stream) {
    int total = C * H_out * W_out;
    int threads = 256;
    bilinear_upsample_kernel<<<dim3((total + threads - 1) / threads, N), threads, 0, stream>>>(
        in, out, C, H_in, W_in, H_out, W_out);
}

// ---- SiLU in-place ----

__global__ void silu_kernel(half* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    x[i] = __float2half(v / (1.0f + expf(-v)));
}

void silu_inplace(half* x, int n, cudaStream_t stream) {
    int threads = 256;
    silu_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(x, n);
}

// ---- Permute [B, S, D] → [B, D, pH, pW] ----

__global__ void permute_bsd_to_nchw(const half* __restrict__ in,
                                     half* __restrict__ out,
                                     int S, int D, int pH, int pW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    int total = D * pH * pW;
    if (idx >= total) return;

    int w = idx % pW;
    int rem = idx / pW;
    int h = rem % pH;
    int d = rem / pH;
    int s = h * pW + w;

    out[(b * D + d) * pH * pW + h * pW + w] = in[(b * S + s) * D + d];
}

// ---- DPT Forward (stream-ordered allocation) ----

void dpt_forward(cublasHandle_t handle,
                 const DPTWeights& w,
                 const half* features[4],
                 half* out, half* col_buf,
                 int B, int dim, int patch_h, int patch_w, int patch_size,
                 int features_dim, int out_dim,
                 const int out_channels[4],
                 cudaStream_t stream) {
    int S = patch_h * patch_w;

    // Stream-ordered allocation: near-zero overhead after first call (pool reuse)
    auto alloc = [stream](size_t bytes) -> half* {
        half* ptr;
        CHECK_CUDA(cudaMallocAsync(&ptr, bytes, stream));
        return ptr;
    };
    auto dealloc = [stream](half* ptr) {
        CHECK_CUDA(cudaFreeAsync(ptr, stream));
    };

    half* projected[4];
    half* resized[4];
    int rh[4], rw[4];

    for (int i = 0; i < 4; i++) {
        half* nchw = alloc(B * dim * S * sizeof(half));
        int total = dim * patch_h * patch_w;
        int threads = 256;
        permute_bsd_to_nchw<<<dim3((total + threads - 1) / threads, B), threads, 0, stream>>>(
            features[i], nchw, S, dim, patch_h, patch_w);

        projected[i] = alloc(B * out_channels[i] * S * sizeof(half));
        conv2d(handle, nchw, w.project_w[i], w.project_b[i],
               projected[i], col_buf, B, dim, patch_h, patch_w,
               out_channels[i], 1, 1, 1, 0, stream);
        dealloc(nchw);

        if (i == 0) {
            rh[i] = patch_h * 4; rw[i] = patch_w * 4;
            resized[i] = alloc(B * out_channels[i] * rh[i] * rw[i] * sizeof(half));
            conv_transpose2d(handle, projected[i], w.resize_w[i], w.resize_b[i],
                             resized[i], col_buf, B, out_channels[i], patch_h, patch_w,
                             out_channels[i], 4, 4, 4, 0, stream);
        } else if (i == 1) {
            rh[i] = patch_h * 2; rw[i] = patch_w * 2;
            resized[i] = alloc(B * out_channels[i] * rh[i] * rw[i] * sizeof(half));
            conv_transpose2d(handle, projected[i], w.resize_w[i], w.resize_b[i],
                             resized[i], col_buf, B, out_channels[i], patch_h, patch_w,
                             out_channels[i], 2, 2, 2, 0, stream);
        } else if (i == 2) {
            rh[i] = patch_h; rw[i] = patch_w;
            resized[i] = projected[i]; // identity
        } else {
            rh[i] = (patch_h + 1) / 2; rw[i] = (patch_w + 1) / 2;
            resized[i] = alloc(B * out_channels[i] * rh[i] * rw[i] * sizeof(half));
            conv2d(handle, projected[i], w.resize_w[i], w.resize_b[i],
                   resized[i], col_buf, B, out_channels[i], patch_h, patch_w,
                   out_channels[i], 3, 3, 2, 1, stream);
        }
    }

    // layer_rn: 3×3 conv, no bias
    half* layer_rn[4];
    for (int i = 0; i < 4; i++) {
        layer_rn[i] = alloc(B * features_dim * rh[i] * rw[i] * sizeof(half));
        conv2d(handle, resized[i], w.layer_rn_w[i], nullptr,
               layer_rn[i], col_buf, B, out_channels[i], rh[i], rw[i],
               features_dim, 3, 3, 1, 1, stream);
    }

    // Free projected/resized — no longer needed
    for (int i = 0; i < 4; i++) {
        if (i != 2) { dealloc(projected[i]); dealloc(resized[i]); }
        else dealloc(projected[i]); // resized[2] aliases projected[2]
    }

    // RCU helper
    auto rcu = [&](const RCUWeights& rw_p, half* x, int F, int H, int W) {
        int sz = B * F * H * W;
        half* x_orig = alloc(sz * sizeof(half));
        cudaMemcpyAsync(x_orig, x, sz * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        half* t1 = alloc(sz * sizeof(half));
        silu_inplace(x, sz, stream);
        conv2d(handle, x, rw_p.conv1_w, rw_p.conv1_b, t1, col_buf, B, F, H, W, F, 3, 3, 1, 1, stream);
        silu_inplace(t1, sz, stream);
        conv2d(handle, t1, rw_p.conv2_w, rw_p.conv2_b, x, col_buf, B, F, H, W, F, 3, 3, 1, 1, stream);
        residual_add(x, x_orig, sz, stream);
        dealloc(t1); dealloc(x_orig);
    };

    auto refinenet_fwd = [&](int i, half* path_in, half* lr, int ph, int pw, int target_h, int target_w) -> half* {
        if (w.refinenet[i].has_resconv1) {
            rcu(w.refinenet[i].rcu1, lr, features_dim, ph, pw);
            residual_add(path_in, lr, B * features_dim * ph * pw, stream);
        }
        rcu(w.refinenet[i].rcu2, path_in, features_dim, ph, pw);
        half* up = alloc(B * features_dim * target_h * target_w * sizeof(half));
        bilinear_upsample(path_in, up, B, features_dim, ph, pw, target_h, target_w, stream);
        half* result = alloc(B * features_dim * target_h * target_w * sizeof(half));
        conv2d(handle, up, w.refinenet[i].out_conv_w, w.refinenet[i].out_conv_b,
               result, col_buf, B, features_dim, target_h, target_w, features_dim, 1, 1, 1, 0, stream);
        dealloc(up);
        return result;
    };

    // RefineNet cascade
    half* path4 = refinenet_fwd(3, layer_rn[3], nullptr, rh[3], rw[3], rh[2], rw[2]);
    dealloc(layer_rn[3]);

    half* path3 = refinenet_fwd(2, path4, layer_rn[2], rh[2], rw[2], rh[1], rw[1]);
    dealloc(path4); dealloc(layer_rn[2]);

    half* path2 = refinenet_fwd(1, path3, layer_rn[1], rh[1], rw[1], rh[0], rw[0]);
    dealloc(path3); dealloc(layer_rn[1]);

    int oh = rh[0] * 2, ow = rw[0] * 2;
    half* path1 = refinenet_fwd(0, path2, layer_rn[0], rh[0], rw[0], oh, ow);
    dealloc(path2); dealloc(layer_rn[0]);

    // Output head
    int fh = features_dim / 2;
    half* oc1 = alloc(B * fh * oh * ow * sizeof(half));
    conv2d(handle, path1, w.output_conv1_w, w.output_conv1_b,
           oc1, col_buf, B, features_dim, oh, ow, fh, 3, 3, 1, 1, stream);
    dealloc(path1);

    int full_h = patch_h * patch_size, full_w = patch_w * patch_size;
    half* oc1_up = alloc(B * fh * full_h * full_w * sizeof(half));
    bilinear_upsample(oc1, oc1_up, B, fh, oh, ow, full_h, full_w, stream);
    dealloc(oc1);

    half* oc2a = alloc(B * 32 * full_h * full_w * sizeof(half));
    conv2d(handle, oc1_up, w.output_conv2a_w, w.output_conv2a_b,
           oc2a, col_buf, B, fh, full_h, full_w, 32, 3, 3, 1, 1, stream);
    dealloc(oc1_up);
    silu_inplace(oc2a, B * 32 * full_h * full_w, stream);
    conv2d(handle, oc2a, w.output_conv2b_w, w.output_conv2b_b,
           out, col_buf, B, 32, full_h, full_w, out_dim, 1, 1, 1, 0, stream);
    dealloc(oc2a);
}

} // namespace rf
