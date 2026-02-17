#include "model/renderformer.h"
#include "core/cuda_utils.h"
#include <cstdio>
#include <cuda_fp16.h>

namespace rf {

// Get fp32 weight on GPU (for norm gamma — already on device)
static const float* fp32(SafeTensorsFile& sf, const std::string& key) {
    auto t = sf.get(key);
    return reinterpret_cast<const float*>(t.data);
}

// Copy fp32 from GPU to host, convert to fp16, upload back
static const half* to_fp16(SafeTensorsFile& sf, const std::string& key,
                           std::vector<GpuBuffer>& bufs) {
    auto t = sf.get(key);
    int64_t n = t.numel();

    // D2H copy
    std::vector<float> h_fp32(n);
    CHECK_CUDA(cudaMemcpy(h_fp32.data(), t.data, n * sizeof(float), cudaMemcpyDeviceToHost));

    // fp32 → fp16 on CPU
    std::vector<half> h_fp16(n);
    for (int64_t i = 0; i < n; i++)
        h_fp16[i] = __float2half_rn(h_fp32[i]);

    // Upload fp16
    auto& buf = bufs.emplace_back(n * sizeof(half));
    CHECK_CUDA(cudaMemcpy(buf.data, h_fp16.data(), n * sizeof(half), cudaMemcpyHostToDevice));
    return static_cast<const half*>(buf.data);
}

void RenderFormerModel::load(SafeTensorsFile& sf) {
    auto& b = weight_buffers;
    auto h = [&](const std::string& k) { return to_fp16(sf, k, b); };
    auto f = [&](const std::string& k) { return fp32(sf, k); };

    // Learnable tokens
    tri_token = h("tri_token");
    reg_tokens = h("reg_tokens");

    // Texture encoder
    tex_enc_w = h("texture_encoder.weight");
    tex_enc_b = h("texture_encoder.bias");
    tex_enc_norm = f("texture_encoder_norm.weight");

    // VN encoder
    vn_proj_w = h("vn_encoding_proj.weight");
    vn_proj_b = h("vn_encoding_proj.bias");
    vn_enc_norm = f("vn_encoder_norm.weight");

    // Encoder RoPE
    enc_rope_freqs = f("transformer.rope_emb.freqs");

    // Encoder layers
    for (int i = 0; i < Config::num_layers; i++) {
        auto p = "transformer.layers." + std::to_string(i) + ".";
        enc_layers[i] = {
            .query_norm_gamma = f(p + "query_norm.weight"),
            .attn = {
                .in_proj_w = h(p + "multihead_attn.in_proj.weight"),
                .in_proj_b = nullptr, // no bias
                .out_proj_w = h(p + "multihead_attn.out_proj.weight"),
                .out_proj_b = nullptr,
                .q_norm_gamma = f(p + "multihead_attn.q_norm.weight"),
                .k_norm_gamma = f(p + "multihead_attn.k_norm.weight"),
            },
            .ffn_norm_gamma = f(p + "ffn_norm.weight"),
            .ffn = {
                .w1 = h(p + "ffn.w1.weight"),
                .w2 = h(p + "ffn.w2.weight"),
                .w3 = h(p + "ffn.w3.weight"),
            },
        };
    }

    // View transformer
    ray_map_patch_token = h("view_transformer.ray_map_patch_token");
    ray_enc_w = h("view_transformer.ray_map_encoder.weight");
    ray_enc_b = h("view_transformer.ray_map_encoder.bias");
    ray_enc_norm = f("view_transformer.ray_map_encoder_norm.weight");
    vt_rope_freqs = f("view_transformer.transformer.rope_emb.freqs");

    // Decoder layers
    for (int i = 0; i < Config::vt_num_layers; i++) {
        auto p = "view_transformer.transformer.layers." + std::to_string(i) + ".";
        dec_layers[i] = {
            .query_norm_gamma = f(p + "query_norm.weight"),
            .kv_norm_gamma = f(p + "kv_norm.weight"),
            .cross_attn = {
                .q_proj_w = h(p + "multihead_attn.q_proj.weight"),
                .q_proj_b = nullptr,
                .k_proj_w = h(p + "multihead_attn.k_proj.weight"),
                .k_proj_b = nullptr,
                .v_proj_w = h(p + "multihead_attn.v_proj.weight"),
                .v_proj_b = nullptr,
                .out_proj_w = h(p + "multihead_attn.out_proj.weight"),
                .out_proj_b = nullptr,
                .q_norm_gamma = f(p + "multihead_attn.q_norm.weight"),
                .k_norm_gamma = f(p + "multihead_attn.k_norm.weight"),
            },
            .self_attn_norm_gamma = f(p + "self_attn_norm.weight"),
            .self_attn = {
                .in_proj_w = h(p + "self_attn.in_proj.weight"),
                .in_proj_b = nullptr,
                .out_proj_w = h(p + "self_attn.out_proj.weight"),
                .out_proj_b = nullptr,
                .q_norm_gamma = f(p + "self_attn.q_norm.weight"),
                .k_norm_gamma = f(p + "self_attn.k_norm.weight"),
            },
            .ffn_norm_gamma = f(p + "ffn_norm.weight"),
            .ffn = {
                .w1 = h(p + "ffn.w1.weight"),
                .w2 = h(p + "ffn.w2.weight"),
                .w3 = h(p + "ffn.w3.weight"),
            },
        };
    }

    // DPT head
    std::string dp = "view_transformer.out_dpt.";
    for (int i = 0; i < 4; i++) {
        auto pi = dp + "projects." + std::to_string(i) + ".";
        dpt.project_w[i] = h(pi + "weight");
        dpt.project_b[i] = h(pi + "bias");
    }
    // Resize layers: 0=ConvTranspose(4,4,s4), 1=ConvTranspose(2,2,s2), 2=Identity, 3=Conv(3,3,s2,p1)
    auto rl = dp + "resize_layers.";
    dpt.resize_w[0] = h(rl + "0.weight"); dpt.resize_b[0] = h(rl + "0.bias");
    dpt.resize_w[1] = h(rl + "1.weight"); dpt.resize_b[1] = h(rl + "1.bias");
    dpt.resize_w[2] = nullptr; dpt.resize_b[2] = nullptr; // identity
    dpt.resize_w[3] = h(rl + "3.weight"); dpt.resize_b[3] = h(rl + "3.bias");

    // layer_rn
    auto sc = dp + "scratch.";
    dpt.layer_rn_w[0] = h(sc + "layer1_rn.weight");
    dpt.layer_rn_w[1] = h(sc + "layer2_rn.weight");
    dpt.layer_rn_w[2] = h(sc + "layer3_rn.weight");
    dpt.layer_rn_w[3] = h(sc + "layer4_rn.weight");

    // RefineNets
    auto load_rcu = [&](const std::string& prefix) -> RCUWeights {
        return {
            h(prefix + "conv1.weight"), h(prefix + "conv1.bias"),
            h(prefix + "conv2.weight"), h(prefix + "conv2.bias"),
        };
    };
    for (int i = 0; i < 4; i++) {
        auto rn = sc + "refinenet" + std::to_string(i + 1) + ".";
        dpt.refinenet[i].out_conv_w = h(rn + "out_conv.weight");
        dpt.refinenet[i].out_conv_b = h(rn + "out_conv.bias");
        dpt.refinenet[i].rcu2 = load_rcu(rn + "resConvUnit2.");
        if (i < 3) { // refinenet1-3 have resConvUnit1, refinenet4 does not
            dpt.refinenet[i].has_resconv1 = true;
            dpt.refinenet[i].rcu1 = load_rcu(rn + "resConvUnit1.");
        } else {
            dpt.refinenet[i].has_resconv1 = false;
        }
    }

    // Output convs
    dpt.output_conv1_w = h(sc + "output_conv1.weight");
    dpt.output_conv1_b = h(sc + "output_conv1.bias");
    dpt.output_conv2a_w = h(sc + "output_conv2.0.weight");
    dpt.output_conv2a_b = h(sc + "output_conv2.0.bias");
    dpt.output_conv2b_w = h(sc + "output_conv2.2.weight");
    dpt.output_conv2b_b = h(sc + "output_conv2.2.bias");

    printf("Model loaded: %zu weight buffers\n", b.size());
}

} // namespace rf
