#include "io/image_writer.h"
#include <cmath>
#include <algorithm>
#include <cstring>

#include <zlib.h>
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 0
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace rf {

void write_exr(const std::string& path, const float* data, int width, int height, int channels) {
    // Deinterleave RGB channels
    std::vector<float> r(width * height), g(width * height), b(width * height);
    for (int i = 0; i < width * height; ++i) {
        r[i] = data[i * channels + 0];
        g[i] = data[i * channels + 1];
        b[i] = data[i * channels + 2];
    }

    const float* channel_ptrs[] = {b.data(), g.data(), r.data()}; // EXR is BGR order
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;
    image.images = reinterpret_cast<unsigned char**>(const_cast<float**>(channel_ptrs));
    image.width = width;
    image.height = height;

    header.num_channels = 3;
    std::vector<EXRChannelInfo> channels_info(3);
    strcpy(channels_info[0].name, "B");
    strcpy(channels_info[1].name, "G");
    strcpy(channels_info[2].name, "R");
    header.channels = channels_info.data();

    std::vector<int> pixel_types(3, TINYEXR_PIXELTYPE_FLOAT);
    std::vector<int> requested_pixel_types(3, TINYEXR_PIXELTYPE_FLOAT);
    header.pixel_types = pixel_types.data();
    header.requested_pixel_types = requested_pixel_types.data();

    const char* err = nullptr;
    SaveEXRImageToFile(&image, &header, path.c_str(), &err);
    if (err) {
        std::string msg(err);
        FreeEXRErrorMessage(err);
        throw std::runtime_error("EXR write failed: " + msg);
    }
}

void write_png(const std::string& path, const unsigned char* data, int width, int height, int channels) {
    stbi_write_png(path.c_str(), width, height, channels, data, width * channels);
}

void tonemap_agx(const float* hdr, unsigned char* ldr, int width, int height, int channels) {
    // Simplified AGX-inspired tone mapping
    for (int i = 0; i < width * height * channels; ++i) {
        float v = hdr[i];
        v = std::max(0.0f, v);
        // Simple filmic curve: x / (x + 1) with gamma
        v = v / (v + 1.0f);
        v = std::pow(v, 1.0f / 2.2f);
        ldr[i] = static_cast<unsigned char>(std::clamp(v * 255.0f, 0.0f, 255.0f));
    }
}

} // namespace rf
