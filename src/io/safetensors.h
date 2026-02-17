#pragma once
#include "core/tensor.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

namespace rf {

// Loads a .safetensors file: mmap on host, bulk copy to GPU
struct SafeTensorsFile {
    struct TensorInfo {
        std::string name;
        Dtype dtype;
        std::vector<int64_t> shape;
        size_t data_offset; // offset from start of data section
        size_t data_size;
    };

    std::unordered_map<std::string, TensorInfo> entries;
    GpuBuffer gpu_buf;        // all weights contiguous on GPU
    size_t data_section_offset = 0; // offset of data section in file

    // Load from file path
    static SafeTensorsFile load(const std::string& path);

    // Get a tensor view by key (references gpu_buf)
    Tensor get(const std::string& key) const;

    // Check if key exists
    bool has(const std::string& key) const { return entries.count(key); }
};

} // namespace rf
