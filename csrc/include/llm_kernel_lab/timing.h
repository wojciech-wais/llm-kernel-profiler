#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace llm_kernel_lab {

/**
 * RAII wrapper for CUDA events used in timing.
 */
class CudaEventTimer {
public:
    CudaEventTimer();
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    /** Record the start event on the given stream. */
    void start(cudaStream_t stream = nullptr);

    /** Record the stop event on the given stream. */
    void stop(cudaStream_t stream = nullptr);

    /** Synchronize and return elapsed time in milliseconds. */
    float elapsed_ms();

private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
};

/**
 * Time a callable over multiple iterations, returning per-iteration times.
 *
 * @param fn  Callable that takes a cudaStream_t.
 * @param stream  CUDA stream to use.
 * @param warmup  Number of warmup calls.
 * @param iterations  Number of timed calls.
 * @return  Vector of per-iteration times in milliseconds.
 */
std::vector<float> time_kernel_iterations(
    const std::function<void(cudaStream_t)>& fn,
    cudaStream_t stream,
    int warmup,
    int iterations
);

}  // namespace llm_kernel_lab
