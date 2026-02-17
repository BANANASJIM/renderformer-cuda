#include "model/renderformer.h"
#include "pipeline/renderer.h"
#include "io/hdf5_reader.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::string model_path, scene_path, output_dir = "output/";
    int img_res = 512;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--scene" && i + 1 < argc) scene_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--res" && i + 1 < argc) img_res = std::stoi(argv[++i]);
    }

    if (scene_path.empty() || model_path.empty()) {
        fprintf(stderr, "Usage: renderformer --scene <h5> --model <safetensors> [--output <dir>] [--res <int>]\n");
        return 1;
    }

    printf("Loading scene: %s\n", scene_path.c_str());
    auto scene = rf::Scene::load(scene_path);
    printf("  triangles: %lld, cameras: %lld\n",
           (long long)scene.num_triangles, (long long)scene.num_cameras);

    printf("Loading model: %s\n", model_path.c_str());
    auto sf = rf::SafeTensorsFile::load(model_path);
    printf("  %zu tensors in safetensors\n", sf.entries.size());

    rf::RenderFormerModel model;
    model.load(sf);

    rf::render(model, scene, output_dir, img_res);
    return 0;
}
