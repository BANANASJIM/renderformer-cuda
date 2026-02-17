#include "kernels/cudnn_sdpa.h"
#include "core/cuda_utils.h"
#include <cudnn_frontend.h>
#include <cmath>
#include <unordered_map>

namespace fe = cudnn_frontend;

namespace rf {

enum : int64_t { UID_Q = 1, UID_K, UID_V, UID_O };

struct CudnnSdpa::Impl {
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> graph;
    void* workspace = nullptr;
    int64_t workspace_size = 0;
    bool owns_handle = false;

    ~Impl() {
        if (workspace) cudaFree(workspace);
        if (owns_handle && handle) cudnnDestroy(handle);
    }
};

CudnnSdpa::~CudnnSdpa() = default;

std::unique_ptr<CudnnSdpa> CudnnSdpa::create(int B, int H, int Sq, int Sk, int D) {
    auto sdpa = std::make_unique<CudnnSdpa>();
    sdpa->impl_ = std::make_unique<Impl>();
    auto& d = *sdpa->impl_;

    CHECK_CUDNN(cudnnCreate(&d.handle));
    d.owns_handle = true;

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    int64_t b = B, h = H, sq = Sq, sk = Sk, dv = D;

    // BHSD-contiguous strides
    auto bhsd_stride = [](int64_t h_, int64_t s_, int64_t d_) {
        return std::vector<int64_t>{h_ * s_ * d_, s_ * d_, d_, 1};
    };

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q").set_uid(UID_Q)
        .set_dim({b, h, sq, dv}).set_stride(bhsd_stride(h, sq, dv)));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_uid(UID_K)
        .set_dim({b, h, sk, dv}).set_stride(bhsd_stride(h, sk, dv)));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_uid(UID_V)
        .set_dim({b, h, sk, dv}).set_stride(bhsd_stride(h, sk, dv)));

    float scale = 1.0f / std::sqrt((float)D);

    auto [O, Stats] = graph->sdpa(Q, K, V,
        fe::graph::SDPA_attributes()
            .set_name("sdpa")
            .set_attn_scale(scale)
            .set_generate_stats(false));

    O->set_output(true)
      .set_dim({b, h, sq, dv})
      .set_stride(bhsd_stride(h, sq, dv))
      .set_uid(UID_O);

    auto status = graph->build(d.handle, {fe::HeurMode_t::A});
    if (!status.is_good()) {
        fprintf(stderr, "[cuDNN SDPA] graph build failed: %s\n", status.get_message().c_str());
        return nullptr;
    }

    status = graph->get_workspace_size(d.workspace_size);
    if (!status.is_good() || d.workspace_size < 0) {
        fprintf(stderr, "[cuDNN SDPA] workspace query failed\n");
        return nullptr;
    }

    if (d.workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&d.workspace, d.workspace_size));
    }

    d.graph = graph;
    printf("[cuDNN SDPA] plan built: B=%d H=%d Sq=%d Sk=%d D=%d workspace=%ldB\n",
           B, H, Sq, Sk, D, (long)d.workspace_size);
    return sdpa;
}

void CudnnSdpa::forward(const half* Q, const half* K, const half* V, half* O, cudaStream_t stream) {
    auto& d = *impl_;
    cudnnSetStream(d.handle, stream);

    std::unordered_map<int64_t, void*> variant_pack = {
        {UID_Q, const_cast<half*>(Q)},
        {UID_K, const_cast<half*>(K)},
        {UID_V, const_cast<half*>(V)},
        {UID_O, O},
    };

    auto status = d.graph->execute(d.handle, variant_pack, d.workspace);
    if (!status.is_good()) {
        fprintf(stderr, "[cuDNN SDPA] execute failed: %s\n", status.get_message().c_str());
    }
}

} // namespace rf
