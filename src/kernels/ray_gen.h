#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

// Generate rays from c2w and fov (Blender convention: -Z forward, +Y up)
// c2w: [B, 4, 4] fp32, fov: [B] fp32 (radians)
// rays_o: [B, 3] fp32 (camera origin = c2w translation)
// rays_d: [B, H, W, 3] fp16 (normalized world-space directions)
void ray_gen(const float* c2w, const float* fov,
             float* rays_o, half* rays_d,
             int batch, int img_res, cudaStream_t stream = nullptr);

} // namespace rf
