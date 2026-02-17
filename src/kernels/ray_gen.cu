#include "kernels/ray_gen.h"
#include <cmath>

namespace rf {

__global__ void ray_gen_kernel(const float* __restrict__ c2w,
                               const float* __restrict__ fov,
                               float* __restrict__ rays_o,
                               half* __restrict__ rays_d,
                               int img_res) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (px >= img_res || py >= img_res) return;

    const float* M = c2w + b * 16; // [4,4] row-major

    // Camera origin = translation column
    if (px == 0 && py == 0) {
        rays_o[b * 3 + 0] = M[3];  // M[0,3]
        rays_o[b * 3 + 1] = M[7];  // M[1,3]
        rays_o[b * 3 + 2] = M[11]; // M[2,3]
    }

    // Pinhole: pixel → camera-space direction
    float cx = img_res * 0.5f;
    float fx = cx / tanf(0.5f * fov[b]);
    float x = (px + 0.5f - cx) / fx;
    float y = -(py + 0.5f - cx) / fx; // fy == fx, negate y for Blender
    float z = -1.0f;

    // Rotate by R = c2w[:3,:3]: d_world = R @ d_cam
    float dx = M[0] * x + M[1] * y + M[2] * z;
    float dy = M[4] * x + M[5] * y + M[6] * z;
    float dz = M[8] * x + M[9] * y + M[10] * z;

    // Normalize
    float inv_len = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv_len; dy *= inv_len; dz *= inv_len;

    int idx = (b * img_res * img_res + py * img_res + px) * 3;
    rays_d[idx + 0] = __float2half(dx);
    rays_d[idx + 1] = __float2half(dy);
    rays_d[idx + 2] = __float2half(dz);
}

void ray_gen(const float* c2w, const float* fov,
             float* rays_o, half* rays_d,
             int batch, int img_res, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((img_res + 15) / 16, (img_res + 15) / 16, batch);
    ray_gen_kernel<<<grid, block, 0, stream>>>(c2w, fov, rays_o, rays_d, img_res);
}

} // namespace rf
