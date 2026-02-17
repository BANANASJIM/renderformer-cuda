#pragma once
#include "model/renderformer.h"
#include "io/hdf5_reader.h"
#include "core/tensor.h"
#include <vector>
#include <memory>

namespace rf {

struct RenderSession {
    static RenderSession create(const RenderFormerModel& model,
                                const Scene& scene, int img_res = 512);

    // Synchronous: blocks until frame is done, returns GPU fp32 HDR [3,H,W]
    const float* render_frame(const float* h_c2w_4x4, float fov_rad);

    // Async: non-blocking render
    void render_frame_async(const float* h_c2w_4x4, float fov_rad);
    bool is_frame_ready();
    const float* get_frame_result(); // blocks if not ready

    // Copy last frame to host buffer [H,W,3] fp32 HWC
    void download_frame_hwc(float* h_out);

    float last_frame_ms() const { return last_ms_; }
    int width() const { return img_res_; }
    int height() const { return img_res_; }

    ~RenderSession();
    RenderSession(RenderSession&&) noexcept;
    RenderSession& operator=(RenderSession&&) noexcept;

private:
    RenderSession() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int img_res_ = 0;
    float last_ms_ = 0;
};

} // namespace rf
