#include "io/safetensors.h"
#include <fstream>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Minimal JSON parsing for safetensors header (no external dependency)
namespace {

struct JsonParser {
    const char* p;
    const char* end;

    void skip_ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }

    std::string parse_string() {
        skip_ws();
        if (*p != '"') throw std::runtime_error("expected '\"'");
        ++p;
        std::string s;
        while (p < end && *p != '"') {
            if (*p == '\\') { ++p; s += *p++; }
            else s += *p++;
        }
        ++p; // closing "
        return s;
    }

    void expect(char c) { skip_ws(); if (*p != c) throw std::runtime_error(std::string("expected '") + c + "'"); ++p; }

    int64_t parse_int() {
        skip_ws();
        int64_t v = 0;
        while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
        return v;
    }
};

} // anonymous namespace

namespace rf {

SafeTensorsFile SafeTensorsFile::load(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open: " + path);

    struct stat st;
    fstat(fd, &st);
    size_t file_size = st.st_size;

    auto* mapped = static_cast<const char*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    std::unique_ptr<char[]> read_buf;
    if (mapped == MAP_FAILED) {
        // Fallback for environments where mmap is unavailable (e.g. ncu profiler)
        read_buf = std::make_unique<char[]>(file_size);
        lseek(fd, 0, SEEK_SET);
        size_t total = 0;
        while (total < file_size) {
            auto n = read(fd, read_buf.get() + total, file_size - total);
            if (n <= 0) { close(fd); throw std::runtime_error("read failed"); }
            total += n;
        }
        mapped = read_buf.get();
    }

    // Parse header size (first 8 bytes, little-endian uint64)
    uint64_t header_size;
    memcpy(&header_size, mapped, 8);

    SafeTensorsFile result;
    result.data_section_offset = 8 + header_size;
    size_t data_size = file_size - result.data_section_offset;

    // Parse JSON header
    JsonParser jp{mapped + 8, mapped + 8 + header_size};
    jp.expect('{');
    jp.skip_ws();

    while (*jp.p != '}') {
        auto key = jp.parse_string();
        jp.expect(':');
        jp.expect('{');

        TensorInfo info;
        info.name = key;

        // Parse tensor metadata fields
        while (*jp.p != '}') {
            jp.skip_ws();
            auto field = jp.parse_string();
            jp.expect(':');

            if (field == "dtype") {
                auto dt = jp.parse_string();
                info.dtype = (dt == "F16") ? Dtype::FP16 : Dtype::FP32;
            } else if (field == "shape") {
                jp.expect('[');
                jp.skip_ws();
                while (*jp.p != ']') {
                    info.shape.push_back(jp.parse_int());
                    jp.skip_ws();
                    if (*jp.p == ',') ++jp.p;
                }
                jp.expect(']');
            } else if (field == "data_offsets") {
                jp.expect('[');
                info.data_offset = jp.parse_int();
                jp.expect(',');
                size_t end_off = jp.parse_int();
                info.data_size = end_off - info.data_offset;
                jp.expect(']');
            }
            jp.skip_ws();
            if (*jp.p == ',') ++jp.p;
        }
        jp.expect('}');

        if (key != "__metadata__") {
            result.entries[key] = std::move(info);
        }
        jp.skip_ws();
        if (*jp.p == ',') ++jp.p;
        jp.skip_ws();
    }

    // Bulk copy entire data section to GPU
    result.gpu_buf = upload(mapped + result.data_section_offset, data_size);

    if (!read_buf) munmap(const_cast<char*>(mapped), file_size);
    close(fd);
    return result;
}

Tensor SafeTensorsFile::get(const std::string& key) const {
    auto it = entries.find(key);
    if (it == entries.end()) throw std::runtime_error("key not found: " + key);

    const auto& info = it->second;
    Tensor t;
    t.data = static_cast<char*>(gpu_buf.data) + info.data_offset;
    t.dtype = info.dtype;
    t.ndim = static_cast<int>(info.shape.size());
    for (int i = 0; i < t.ndim; ++i) t.shape[i] = info.shape[i];
    return t;
}

} // namespace rf
