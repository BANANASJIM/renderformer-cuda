#include "kernels/coord_transform.h"

namespace rf {

// Each thread transforms one vertex (3 floats) of one triangle
// tri layout: [B, N, 3_verts, 3_xyz], vn same layout
__global__ void world_to_cam_kernel(const float* __restrict__ c2w,
                                    half* __restrict__ tri,
                                    half* __restrict__ vn,
                                    int n_tris) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tris * 3; // 3 vertices per triangle
    int b = blockIdx.y;
    if (idx >= total) return;

    const float* M = c2w + b * 16;

    // R^T (transpose of rotation)
    float r00 = M[0], r10 = M[1], r20 = M[2];
    float r01 = M[4], r11 = M[5], r21 = M[6];
    float r02 = M[8], r12 = M[9], r22 = M[10];
    // Translation
    float tx = M[3], ty = M[7], tz = M[11];

    int base = (b * n_tris * 3 + idx) * 3;

    // Transform vertex: R^T(v - t)
    {
        float vx = __half2float(tri[base + 0]) - tx;
        float vy = __half2float(tri[base + 1]) - ty;
        float vz = __half2float(tri[base + 2]) - tz;
        tri[base + 0] = __float2half(r00 * vx + r01 * vy + r02 * vz);
        tri[base + 1] = __float2half(r10 * vx + r11 * vy + r12 * vz);
        tri[base + 2] = __float2half(r20 * vx + r21 * vy + r22 * vz);
    }

    // Transform normal: R^T·n (rotation only)
    if (vn) {
        float nx = __half2float(vn[base + 0]);
        float ny = __half2float(vn[base + 1]);
        float nz = __half2float(vn[base + 2]);
        vn[base + 0] = __float2half(r00 * nx + r01 * ny + r02 * nz);
        vn[base + 1] = __float2half(r10 * nx + r11 * ny + r12 * nz);
        vn[base + 2] = __float2half(r20 * nx + r21 * ny + r22 * nz);
    }
}

void world_to_cam(const float* c2w, half* tri, half* vn,
                  int batch, int n_tris, cudaStream_t stream) {
    int total = n_tris * 3;
    int threads = 256;
    int blocks_x = (total + threads - 1) / threads;
    dim3 grid(blocks_x, batch);
    world_to_cam_kernel<<<grid, threads, 0, stream>>>(c2w, tri, vn, n_tris);
}

} // namespace rf
