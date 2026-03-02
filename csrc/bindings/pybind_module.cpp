#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "llm_kernel_lab/benchmark.h"

namespace py = pybind11;
using namespace llm_kernel_lab;

PYBIND11_MODULE(llm_kernel_lab_cuda, m) {
    m.doc() = "LLM Kernel Lab CUDA harness — native benchmark utilities";

    py::enum_<ProfileLevel>(m, "ProfileLevel")
        .value("NONE", ProfileLevel::None)
        .value("TIMING", ProfileLevel::Timing)
        .value("BASIC", ProfileLevel::Basic)
        .value("FULL", ProfileLevel::Full);

    py::class_<CudaBenchmarkResult>(m, "CudaBenchmarkResult")
        .def(py::init<>())
        .def_readwrite("runtime_ms", &CudaBenchmarkResult::runtime_ms)
        .def_readwrite("min_ms", &CudaBenchmarkResult::min_ms)
        .def_readwrite("max_ms", &CudaBenchmarkResult::max_ms)
        .def_readwrite("stddev_ms", &CudaBenchmarkResult::stddev_ms)
        .def_readwrite("iterations", &CudaBenchmarkResult::iterations)
        .def("__repr__", [](const CudaBenchmarkResult& r) {
            return "<CudaBenchmarkResult runtime_ms=" + std::to_string(r.runtime_ms) +
                   " min_ms=" + std::to_string(r.min_ms) +
                   " max_ms=" + std::to_string(r.max_ms) + ">";
        });

    m.def("run_kernel_benchmark",
        [](py::function launcher, int warmup, int iters, const std::string& level) {
            auto cpp_launcher = [&launcher](cudaStream_t stream, const LaunchConfig& config) {
                launcher(reinterpret_cast<uintptr_t>(stream));
            };
            return run_kernel_benchmark(cpp_launcher, warmup, iters, parse_profile_level(level));
        },
        py::arg("launcher"),
        py::arg("warmup_iters") = 10,
        py::arg("iters") = 100,
        py::arg("profile_level") = "timing",
        "Run a kernel benchmark with timing measurements."
    );
}
