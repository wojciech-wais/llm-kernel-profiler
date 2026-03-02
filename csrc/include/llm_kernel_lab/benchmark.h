#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <string>

namespace llm_kernel_lab {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    size_t shared_mem_bytes = 0;
};

using KernelLauncher = std::function<void(cudaStream_t, const LaunchConfig&)>;

enum class ProfileLevel {
    None,
    Timing,
    Basic,
    Full,
};

struct CudaBenchmarkResult {
    float runtime_ms = 0.0f;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float stddev_ms = 0.0f;
    int iterations = 0;
};

/**
 * Run a kernel benchmark with the specified launcher and configuration.
 *
 * @param launcher  Function that launches the kernel on a given stream.
 * @param warmup_iters  Number of warmup iterations (not timed).
 * @param iters  Number of timed iterations.
 * @param level  Profiling level.
 * @return  Benchmark result with timing statistics.
 */
CudaBenchmarkResult run_kernel_benchmark(
    KernelLauncher launcher,
    int warmup_iters,
    int iters,
    ProfileLevel level = ProfileLevel::Timing
);

/**
 * Parse a profile level string into the enum.
 */
ProfileLevel parse_profile_level(const std::string& level_str);

}  // namespace llm_kernel_lab
