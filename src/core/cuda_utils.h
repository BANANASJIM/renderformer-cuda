#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,          \
                    __LINE__, (int)status);                                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUDNN(call)                                                      \
    do {                                                                       \
        cudnnStatus_t status = (call);                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__,           \
                    __LINE__, cudnnGetErrorString(status));                     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

namespace rf {

struct CudaStream {
    cudaStream_t stream = nullptr;
    CudaStream() { CHECK_CUDA(cudaStreamCreate(&stream)); }
    ~CudaStream() { if (stream) cudaStreamDestroy(stream); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& o) noexcept : stream(o.stream) { o.stream = nullptr; }
    operator cudaStream_t() const { return stream; }
    void sync() const { CHECK_CUDA(cudaStreamSynchronize(stream)); }
};

struct CublasHandle {
    cublasHandle_t handle = nullptr;
    CublasHandle() { CHECK_CUBLAS(cublasCreate(&handle)); }
    ~CublasHandle() { if (handle) cublasDestroy(handle); }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    void set_stream(cudaStream_t s) { CHECK_CUBLAS(cublasSetStream(handle, s)); }
    operator cublasHandle_t() const { return handle; }
};

// CUDA event-based GPU timer — zero overhead when not queried
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer() { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    void start(cudaStream_t s) { cudaEventRecord(start_, s); }
    void stop(cudaStream_t s)  { cudaEventRecord(stop_, s); }
    float ms() { cudaEventSynchronize(stop_); float t; cudaEventElapsedTime(&t, start_, stop_); return t; }
};

// Accumulates timing across multiple start/stop pairs
struct StageTimer {
    GpuTimer timer;
    float total_ms = 0;
    int count = 0;
    void start(cudaStream_t s) { timer.start(s); }
    void stop(cudaStream_t s)  { timer.stop(s); total_ms += timer.ms(); ++count; }
    void report(const char* name) const {
        if (count > 0) printf("  %-20s %8.2f ms (%d calls, %.2f ms/call)\n",
                               name, total_ms, count, total_ms / count);
    }
    void reset() { total_ms = 0; count = 0; }
};

} // namespace rf
