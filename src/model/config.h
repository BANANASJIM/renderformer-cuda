#pragma once

namespace rf {

// Hardcoded config for renderformer-v1.1-swin-large
struct Config {
    static constexpr int latent_dim = 1024;
    static constexpr int num_heads = 8;
    static constexpr int head_dim = latent_dim / num_heads; // 128
    static constexpr int num_layers = 12;
    static constexpr int ffn_hidden = 4096;
    static constexpr int num_register_tokens = 32;

    static constexpr int texture_channels = 13;
    static constexpr int texture_patch_size = 1;
    static constexpr int vn_pe_num_freqs = 6;
    static constexpr int vn_pe_out_dim = 9 + 9 * vn_pe_num_freqs * 2; // 117

    static constexpr int rope_num_freqs = 12; // vertex_pe_num_freqs
    static constexpr int rope_dim = 12;       // = rope_num_freqs (for 'triangle' type)
    static constexpr bool rope_double_max_freq = true;

    // View transformer (decoder)
    static constexpr int vt_latent_dim = 1024;
    static constexpr int vt_num_heads = 8;
    static constexpr int vt_num_layers = 12;
    static constexpr int vt_ffn_hidden = 4096;
    static constexpr int vt_rope_dim = 12; // min(12, head_dim/18*2)

    static constexpr int patch_size = 8;
    static constexpr int window_size = 8;
    static constexpr int shift_size = 4;

    static constexpr int vdir_num_freqs = 0;
    static constexpr int vdir_pe_out_dim = 3; // 3 + 3*0*2 = 3 (no encoding)
    static constexpr int ray_encoder_in = vdir_pe_out_dim * patch_size * patch_size; // 192

    // DPT
    static constexpr int dpt_features = 256;
    static constexpr int dpt_out_channels[4] = {256, 512, 1024, 1024};
    static constexpr int dpt_out_layers[4] = {2, 5, 8, 11};
    static constexpr int dpt_out_dim = 3; // no alpha
};

} // namespace rf
