#pragma once
#include "core/tensor.h"
#include <string>
#include <vector>

namespace rf {

// Scene data loaded from HDF5
struct Scene {
    GpuBuffer triangles_buf;  // [N, 3, 3] fp32
    GpuBuffer texture_buf;    // [N, 13, 32, 32] fp16
    GpuBuffer vn_buf;         // [N, 3, 3] fp32
    GpuBuffer c2w_buf;        // [K, 4, 4] fp32
    GpuBuffer fov_buf;        // [K] fp32

    int64_t num_triangles = 0;
    int64_t num_cameras = 0;

    Tensor triangles() const { return triangles_buf.as_tensor({num_triangles, 3, 3}, Dtype::FP32); }
    Tensor texture() const { return texture_buf.as_tensor({num_triangles, 13, 32, 32}, Dtype::FP16); }
    Tensor vn() const { return vn_buf.as_tensor({num_triangles, 3, 3}, Dtype::FP32); }
    Tensor c2w() const { return c2w_buf.as_tensor({num_cameras, 4, 4}, Dtype::FP32); }
    Tensor fov() const { return fov_buf.as_tensor({num_cameras}, Dtype::FP32); }

    static Scene load(const std::string& h5_path);
};

} // namespace rf
