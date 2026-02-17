#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <memory>

namespace rf {

// Cached cuDNN fused SDPA (scaled dot-product attention) execution plan.
// Build once for a given shape, then execute many times with zero overhead.
// Uses tensor cores + fused softmax — no intermediate score matrix materialized.
struct CudnnSdpa {
    ~CudnnSdpa();

    // Build execution plan for BHSD-contiguous Q[B,H,Sq,D], K[B,H,Sk,D], V[B,H,Sk,D] → O[B,H,Sq,D]
    // fp16 I/O, fp32 accumulation. scale = 1/sqrt(D).
    static std::unique_ptr<CudnnSdpa> create(int B, int H, int Sq, int Sk, int D);

    // Execute fused SDPA on given BHSD-contiguous buffers.
    void forward(const half* Q, const half* K, const half* V, half* O, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rf
