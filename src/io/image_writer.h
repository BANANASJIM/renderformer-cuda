#pragma once
#include <string>
#include <vector>

namespace rf {

// Write float32 HDR image as EXR
void write_exr(const std::string& path, const float* data, int width, int height, int channels = 3);

// Write uint8 LDR image as PNG
void write_png(const std::string& path, const unsigned char* data, int width, int height, int channels = 3);

// Simple AGX-like tone mapping: HDR float → LDR uint8
void tonemap_agx(const float* hdr, unsigned char* ldr, int width, int height, int channels = 3);

} // namespace rf
