#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include <numeric>
#include <span>
#include <cassert>
#include "core/cuda_utils.h"

namespace rf {

enum class Dtype : uint8_t { FP16, FP32 };

inline size_t dtype_size(Dtype d) { return d == Dtype::FP16 ? 2 : 4; }

// Non-owning GPU tensor view
struct Tensor {
    void* data = nullptr;
    std::array<int64_t, 8> shape{};
    int ndim = 0;
    Dtype dtype = Dtype::FP32;

    int64_t numel() const {
        return std::accumulate(shape.begin(), shape.begin() + ndim, int64_t(1), std::multiplies<>{});
    }
    size_t nbytes() const { return numel() * dtype_size(dtype); }
    int64_t dim(int i) const { return shape[i < 0 ? ndim + i : i]; }

    half* fp16() const { assert(dtype == Dtype::FP16); return static_cast<half*>(data); }
    float* fp32() const { assert(dtype == Dtype::FP32); return static_cast<float*>(data); }

    // Reshape (same data, different view)
    Tensor view(std::initializer_list<int64_t> new_shape) const {
        Tensor t = *this;
        t.ndim = static_cast<int>(new_shape.size());
        std::copy(new_shape.begin(), new_shape.end(), t.shape.begin());
        assert(t.numel() == numel());
        return t;
    }
};

// Owning GPU buffer
struct GpuBuffer {
    void* data = nullptr;
    size_t size = 0;

    GpuBuffer() = default;
    explicit GpuBuffer(size_t bytes) : size(bytes) {
        CHECK_CUDA(cudaMalloc(&data, bytes));
    }
    ~GpuBuffer() { if (data) cudaFree(data); }
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&& o) noexcept : data(o.data), size(o.size) { o.data = nullptr; o.size = 0; }
    GpuBuffer& operator=(GpuBuffer&& o) noexcept {
        if (this != &o) { if (data) cudaFree(data); data = o.data; size = o.size; o.data = nullptr; o.size = 0; }
        return *this;
    }

    Tensor as_tensor(std::initializer_list<int64_t> shape, Dtype dtype) const {
        Tensor t;
        t.data = data;
        t.ndim = static_cast<int>(shape.size());
        std::copy(shape.begin(), shape.end(), t.shape.begin());
        t.dtype = dtype;
        assert(t.nbytes() <= size);
        return t;
    }
};

// Upload CPU data to new GPU buffer
inline GpuBuffer upload(const void* host_data, size_t bytes) {
    GpuBuffer buf(bytes);
    CHECK_CUDA(cudaMemcpy(buf.data, host_data, bytes, cudaMemcpyHostToDevice));
    return buf;
}

} // namespace rf
