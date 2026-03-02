"""CUDA event-based timing profiler."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from llm_kernel_lab.model import MetricSet


class TimingProfiler:
    """Profile kernels using CUDA events for timing only.

    This is the lowest-overhead profiling mode. It measures wall-clock time
    using CUDA events (when available) or Python time as fallback.
    """

    def __init__(self, warmup: int = 10, repetitions: int = 100) -> None:
        self.warmup = warmup
        self.repetitions = repetitions
        self._torch_available = _check_torch()

    def time_kernel(
        self,
        kernel_fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        flops: float = 0.0,
        bytes_accessed: float = 0.0,
    ) -> MetricSet:
        """Time a kernel function and return basic metrics."""
        kwargs = kwargs or {}

        if self._torch_available:
            return self._time_with_cuda_events(kernel_fn, args, kwargs, flops, bytes_accessed)
        return self._time_with_python(kernel_fn, args, kwargs, flops, bytes_accessed)

    def _time_with_cuda_events(
        self,
        kernel_fn: Callable,
        args: tuple,
        kwargs: dict,
        flops: float,
        bytes_accessed: float,
    ) -> MetricSet:
        import torch

        # Warmup
        for _ in range(self.warmup):
            kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        times_ms: list[float] = []
        for _ in range(self.repetitions):
            start.record()
            kernel_fn(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

        avg_ms = sum(times_ms) / len(times_ms)
        return _build_timing_metrics(avg_ms, flops, bytes_accessed)

    def _time_with_python(
        self,
        kernel_fn: Callable,
        args: tuple,
        kwargs: dict,
        flops: float,
        bytes_accessed: float,
    ) -> MetricSet:
        # Warmup
        for _ in range(self.warmup):
            kernel_fn(*args, **kwargs)

        times_ms: list[float] = []
        for _ in range(self.repetitions):
            t0 = time.perf_counter()
            kernel_fn(*args, **kwargs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        avg_ms = sum(times_ms) / len(times_ms)
        return _build_timing_metrics(avg_ms, flops, bytes_accessed)


def _build_timing_metrics(
    runtime_ms: float,
    flops: float,
    bytes_accessed: float,
) -> MetricSet:
    """Build a MetricSet with timing-only data (other fields zeroed)."""
    runtime_s = runtime_ms / 1000.0
    achieved_tflops = (flops / 1e12 / runtime_s) if runtime_s > 0 and flops > 0 else 0.0
    if runtime_s > 0 and bytes_accessed > 0:
        dram_bw_gbps = bytes_accessed / 1e9 / runtime_s
    else:
        dram_bw_gbps = 0.0
    arithmetic_intensity = (flops / bytes_accessed) if bytes_accessed > 0 else 0.0

    return MetricSet(
        runtime_ms=runtime_ms,
        achieved_tflops=achieved_tflops,
        arithmetic_intensity=arithmetic_intensity,
        dram_bw_gbps=dram_bw_gbps,
        l2_bw_gbps=0.0,
        sm_efficiency=0.0,
        warp_execution_efficiency=0.0,
        eligible_warps_per_cycle=0.0,
        stall_memory_dependency=0.0,
        stall_execution_dependency=0.0,
        stall_sync=0.0,
        occupancy=0.0,
        registers_per_thread=0,
        shared_mem_bytes=0,
        l2_hit_rate=0.0,
    )


def _check_torch() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
