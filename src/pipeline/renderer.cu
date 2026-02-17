#include "pipeline/renderer.h"
#include "pipeline/render_session.h"
#include "io/image_writer.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <sys/stat.h>

namespace rf {

void render(const RenderFormerModel& model, const Scene& scene,
            const std::string& output_dir, int img_res) {
    auto session = RenderSession::create(model, scene, img_res);

    int K = (int)scene.num_cameras;
    std::vector<float> h_fov(K);
    CHECK_CUDA(cudaMemcpy(h_fov.data(), scene.fov().fp32(), K * sizeof(float), cudaMemcpyDeviceToHost));

    // Read c2w matrices to host
    std::vector<float> h_c2w(K * 16);
    CHECK_CUDA(cudaMemcpy(h_c2w.data(), scene.c2w().fp32(), K * 16 * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> h_hwc(img_res * img_res * 3);
    std::vector<unsigned char> h_ldr(img_res * img_res * 3);
    mkdir(output_dir.c_str(), 0755);

    printf("Rendering %d views...\n", K); fflush(stdout);
    float total_ms = 0;

    for (int v = 0; v < K; v++) {
        printf("View %d/%d: ", v + 1, K);
        float fov_rad = h_fov[v] * (float)(M_PI / 180.0);
        session.render_frame(h_c2w.data() + v * 16, fov_rad);
        total_ms += session.last_frame_ms();

        session.download_frame_hwc(h_hwc.data());
        auto base = output_dir + "/view_" + std::to_string(v);
        write_exr(base + ".exr", h_hwc.data(), img_res, img_res, 3);
        tonemap_agx(h_hwc.data(), h_ldr.data(), img_res, img_res, 3);
        write_png(base + ".png", h_ldr.data(), img_res, img_res, 3);
        printf("saved %s.{exr,png} (%.1f ms)\n", base.c_str(), session.last_frame_ms());
    }

    printf("\n=== Timing Summary (%d views) ===\n", K);
    printf("  %-20s %8.2f ms/view\n", "TOTAL (per view)", total_ms / std::max(K, 1));
    printf("================================\n");
    printf("Done: %d views rendered.\n", K);
}

} // namespace rf
