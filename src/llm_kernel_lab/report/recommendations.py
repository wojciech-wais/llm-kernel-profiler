"""Rule-based recommendation engine for kernel optimization."""

from __future__ import annotations

from llm_kernel_lab.model import HardwareProfile, MetricSet
from llm_kernel_lab.report.bottleneck import Bottleneck, classify_bottleneck


def get_recommendations(
    metrics: MetricSet,
    hardware: HardwareProfile,
) -> list[str]:
    """Generate optimization recommendations based on metrics.

    Returns a list of human-readable recommendation strings.
    """
    recs: list[str] = []
    bottleneck = classify_bottleneck(metrics, hardware)

    if bottleneck == Bottleneck.MEMORY_BOUND:
        recs.append(
            "Kernel is memory-bound. Consider increasing tile sizes along the K "
            "dimension or using cp.async for asynchronous data loading."
        )
        recs.append(
            "Check shared memory vs L2 cache usage — moving data to shared memory "
            "can reduce DRAM traffic."
        )
        if metrics.l2_hit_rate < 0.5 and metrics.l2_hit_rate > 0:
            recs.append(
                f"L2 hit rate is low ({metrics.l2_hit_rate:.0%}). Consider restructuring "
                "memory access patterns for better locality."
            )

    elif bottleneck == Bottleneck.LATENCY_BOUND:
        recs.append(
            "Kernel is latency-bound. Consider increasing parallelism (more blocks) "
            "or reducing per-thread register pressure."
        )
        if metrics.stall_sync > 0.2:
            recs.append(
                f"High synchronization stalls ({metrics.stall_sync:.0%}). "
                "Review __syncthreads() placement and consider reducing barrier usage."
            )

    elif bottleneck == Bottleneck.COMPUTE_BOUND:
        recs.append(
            "Kernel is near compute-bound. Performance is close to hardware peak. "
            "Consider algorithmic improvements to reduce total FLOPs."
        )

    # Low occupancy
    if 0 < metrics.occupancy < 0.5:
        recs.append(
            f"Occupancy is low ({metrics.occupancy:.0%}). Consider reducing registers "
            f"per thread (currently {metrics.registers_per_thread}) or shared memory per block "
            f"({metrics.shared_mem_bytes} bytes)."
        )

    # High register usage
    if metrics.registers_per_thread > 128:
        recs.append(
            f"High register usage ({metrics.registers_per_thread} per thread). "
            "This may limit occupancy. Consider using __launch_bounds__ to cap register usage."
        )

    if not recs:
        recs.append("No specific optimization recommendations. Profile with 'full' level for detailed analysis.")

    return recs
