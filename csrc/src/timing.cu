#include "llm_kernel_lab/timing.h"

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                       \
                std::string("CUDA error: ") + cudaGetErrorString(err) +     \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));        \
        }                                                                   \
    } while (0)

namespace llm_kernel_lab {

CudaEventTimer::CudaEventTimer() {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

CudaEventTimer::~CudaEventTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void CudaEventTimer::start(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(start_event_, stream));
}

void CudaEventTimer::stop(cudaStream_t stream) {
    CUDA_CHECK(cudaEventRecord(stop_event_, stream));
}

float CudaEventTimer::elapsed_ms() {
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
    return ms;
}

std::vector<float> time_kernel_iterations(
    const std::function<void(cudaStream_t)>& fn,
    cudaStream_t stream,
    int warmup,
    int iterations
) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        fn(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    std::vector<float> times;
    times.reserve(iterations);

    CudaEventTimer timer;
    for (int i = 0; i < iterations; ++i) {
        timer.start(stream);
        fn(stream);
        timer.stop(stream);
        times.push_back(timer.elapsed_ms());
    }

    return times;
}

}  // namespace llm_kernel_lab
