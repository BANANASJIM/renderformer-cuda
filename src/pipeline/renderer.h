#pragma once
#include "model/renderformer.h"
#include "io/hdf5_reader.h"
#include <string>

namespace rf {

void render(const RenderFormerModel& model, const Scene& scene,
            const std::string& output_dir, int img_res = 512);

} // namespace rf
