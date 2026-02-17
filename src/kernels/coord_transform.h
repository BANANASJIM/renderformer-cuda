#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rf {

// Transform triangles and vertex normals from world to camera coordinates
// c2w: [B, 4, 4] fp32 — inverted internally (w2c = c2w^{-1})
// tri: [B, N, 3, 3] fp16 — vertices, transformed as R^T(v - t)
// vn:  [B, N, 3, 3] fp16 — normals, transformed as R^T·n (rotation only)
// Results written in-place
void world_to_cam(const float* c2w, half* tri, half* vn,
                  int batch, int n_tris, cudaStream_t stream = nullptr);

} // namespace rf
