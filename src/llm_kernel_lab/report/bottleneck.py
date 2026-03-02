"""Bottleneck classification for kernel runs."""

from __future__ import annotations

from llm_kernel_lab.model import HardwareProfile, KernelRun, MetricSet


class Bottleneck:
    MEMORY_BOUND = "memory-bound"
    COMPUTE_BOUND = "compute-bound"
    LATENCY_BOUND = "latency-bound"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


def classify_bottleneck(
    metrics: MetricSet,
    hardware: HardwareProfile,
) -> str:
    """Classify the primary bottleneck of a kernel run.

    Rules (from spec):
    - If dram_bw_gbps > 0.8 * peak and sm_efficiency < 0.7 -> memory-bound
    - If sm_efficiency < 0.5 and stall_execution_dependency > 0.3 -> latency-bound
    - If sm_efficiency > 0.8 and achieved_tflops ~ peak -> compute-bound / near-optimal
    - Otherwise -> balanced
    """
    peak_bw = hardware.memory_bandwidth_gbps
    peak_tflops = hardware.peak_fp16_tflops

    # Memory-bound check
    if peak_bw > 0 and metrics.dram_bw_gbps > 0.8 * peak_bw and metrics.sm_efficiency < 0.7:
        return Bottleneck.MEMORY_BOUND

    # Latency-bound check
    if metrics.sm_efficiency < 0.5 and metrics.stall_execution_dependency > 0.3:
        return Bottleneck.LATENCY_BOUND

    # Compute-bound check
    if peak_tflops > 0 and metrics.sm_efficiency > 0.8:
        utilization = metrics.achieved_tflops / peak_tflops if peak_tflops > 0 else 0
        if utilization > 0.6:
            return Bottleneck.COMPUTE_BOUND

    # If we have enough data to classify but it doesn't match clear categories
    if metrics.sm_efficiency > 0 or metrics.dram_bw_gbps > 0:
        return Bottleneck.BALANCED

    return Bottleneck.UNKNOWN


def classify_bottleneck_from_run(
    run: KernelRun,
) -> str:
    """Classify bottleneck from a KernelRun object."""
    return classify_bottleneck(run.metrics, run.hardware)
