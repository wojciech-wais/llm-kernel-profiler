"""Basic benchmark example using the Python API.

This example shows how to use llm-kernel-lab to benchmark a simple
matrix multiplication and generate a report.
"""

import time

from llm_kernel_lab.bench.runner import BenchmarkRunner
from llm_kernel_lab.model import HardwareProfile, KernelVariant, ProblemShape
from llm_kernel_lab.serialization import save_results


def simple_matmul():
    """A simple CPU-based matmul for demonstration."""
    import numpy as np

    A = np.random.randn(1024, 1024).astype(np.float32)
    B = np.random.randn(1024, 1024).astype(np.float32)
    return A @ B


def main():
    # Define what we want to benchmark
    kernels = [
        KernelVariant(
            id="numpy_matmul",
            op_type="matmul",
            backend="pytorch",
            source="examples.basic_benchmark.simple_matmul",
            config={},
            metadata={"description": "NumPy baseline matmul"},
        ),
    ]

    shapes = [
        ProblemShape(
            id="small_1k",
            op_type="matmul",
            params={"M": 1024, "N": 1024, "K": 1024},
        ),
    ]

    # Create a hardware profile (would be auto-detected with a GPU)
    hardware = HardwareProfile(
        gpu_name="CPU (demo)",
        sm_version=0,
        num_sms=0,
        memory_bandwidth_gbps=50.0,
        peak_fp16_tflops=0.0,
        peak_fp32_tflops=1.0,
        l2_size_mb=0.0,
        driver_version="N/A",
        cuda_version="N/A",
    )

    runner = BenchmarkRunner(backend="pytorch", hardware=hardware)

    # Run with a kernel function mapping
    results = runner.run(
        kernels=kernels,
        shapes=shapes,
        repetitions=10,
        warmup=2,
        profile_level="timing",
        kernel_fns={"numpy_matmul": simple_matmul},
    )

    for run in results:
        print(f"Kernel: {run.variant.id}")
        print(f"  Runtime: {run.metrics.runtime_ms:.2f} ms")
        print(f"  TFLOP/s: {run.metrics.achieved_tflops:.3f}")

    # Save results
    save_results(results, hardware, "./results/basic_benchmark.json")
    print("\nResults saved to ./results/basic_benchmark.json")


if __name__ == "__main__":
    main()
