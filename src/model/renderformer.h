#pragma once
#include "model/config.h"
#include "layers/transformer.h"
#include "layers/dpt.h"
#include "io/safetensors.h"
#include "core/tensor.h"
#include <string>
#include <vector>

namespace rf {

struct RenderFormerModel {
    // Learnable tokens
    const half* tri_token;      // [1, 1, dim]
    const half* reg_tokens;     // [1, num_reg, dim]

    // Texture encoder
    const half* tex_enc_w;      // [dim, 13]
    const half* tex_enc_b;      // [dim]
    const float* tex_enc_norm;  // [dim]

    // VN encoder
    const half* vn_proj_w;      // [dim, 117]
    const half* vn_proj_b;      // [dim]
    const float* vn_enc_norm;   // [dim]

    // Encoder RoPE
    const float* enc_rope_freqs; // [6]

    // Encoder layers
    EncoderLayerWeights enc_layers[Config::num_layers];

    // View transformer
    const half* ray_map_patch_token; // [1, 1, dim]
    const half* ray_enc_w;           // [dim, 192]
    const half* ray_enc_b;           // [dim]
    const float* ray_enc_norm;       // [dim]
    const float* vt_rope_freqs;      // [6]

    // Decoder layers
    DecoderLayerWeights dec_layers[Config::vt_num_layers];

    // DPT head
    DPTWeights dpt;

    // GPU buffers for fp16-converted weights
    std::vector<GpuBuffer> weight_buffers;

    void load(SafeTensorsFile& sf);
};

} // namespace rf
