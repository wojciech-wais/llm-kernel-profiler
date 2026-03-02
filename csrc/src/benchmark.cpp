#include "llm_kernel_lab/benchmark.h"
#include "llm_kernel_lab/timing.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace llm_kernel_lab {

CudaBenchmarkResult run_kernel_benchmark(
    KernelLauncher launcher,
    int warmup_iters,
    int iters,
    ProfileLevel level
) {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    LaunchConfig config{};  // Caller should configure via launcher closure

    auto fn = [&](cudaStream_t s) {
        launcher(s, config);
    };

    auto times = time_kernel_iterations(fn, stream, warmup_iters, iters);

    cudaStreamDestroy(stream);

    CudaBenchmarkResult result;
    result.iterations = iters;

    if (times.empty()) {
        return result;
    }

    float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    result.runtime_ms = sum / static_cast<float>(times.size());
    result.min_ms = *std::min_element(times.begin(), times.end());
    result.max_ms = *std::max_element(times.begin(), times.end());

    // Standard deviation
    float sq_sum = 0.0f;
    for (float t : times) {
        float diff = t - result.runtime_ms;
        sq_sum += diff * diff;
    }
    result.stddev_ms = std::sqrt(sq_sum / static_cast<float>(times.size()));

    return result;
}

ProfileLevel parse_profile_level(const std::string& level_str) {
    if (level_str == "none") return ProfileLevel::None;
    if (level_str == "timing") return ProfileLevel::Timing;
    if (level_str == "basic") return ProfileLevel::Basic;
    if (level_str == "full") return ProfileLevel::Full;
    throw std::invalid_argument("Unknown profile level: " + level_str);
}

}  // namespace llm_kernel_lab
