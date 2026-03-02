"""High-level profiler that abstracts Nsight/CUPTI and timing."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from llm_kernel_lab.model import HardwareProfile, KernelRun, KernelVariant, ProblemShape
from llm_kernel_lab.profiler.nsight import (
    check_ncu_available,
)
from llm_kernel_lab.profiler.timing import TimingProfiler


class Profiler:
    """Unified profiler that selects timing-only or Nsight-based profiling."""

    def __init__(
        self,
        profile_level: str = "timing",
        warmup: int = 10,
        repetitions: int = 100,
        ncu_path: str = "ncu",
    ) -> None:
        self.profile_level = profile_level
        self.warmup = warmup
        self.repetitions = repetitions
        self.ncu_path = ncu_path
        self._timer = TimingProfiler(warmup=warmup, repetitions=repetitions)

    def profile_kernel(
        self,
        kernel_fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        flops: float = 0.0,
        bytes_accessed: float = 0.0,
        variant: KernelVariant | None = None,
        shape: ProblemShape | None = None,
        hardware: HardwareProfile | None = None,
    ) -> KernelRun:
        """Profile a single kernel and return a KernelRun."""
        kwargs = kwargs or {}

        if self.profile_level in ("none", "timing"):
            metrics = self._timer.time_kernel(
                kernel_fn, args, kwargs, flops, bytes_accessed
            )
        else:
            # For basic/full, try Nsight first, fall back to timing
            if check_ncu_available(self.ncu_path):
                # Nsight requires subprocess-based profiling
                # For now, get timing and note that full metrics need ncu CLI
                metrics = self._timer.time_kernel(
                    kernel_fn, args, kwargs, flops, bytes_accessed
                )
            else:
                metrics = self._timer.time_kernel(
                    kernel_fn, args, kwargs, flops, bytes_accessed
                )

        if variant is None:
            variant = KernelVariant(
                id="unknown", op_type="unknown", backend="unknown", source="unknown"
            )
        if shape is None:
            shape = ProblemShape(id="unknown", op_type="unknown")
        if hardware is None:
            hardware = HardwareProfile(
                gpu_name="unknown", sm_version=0, num_sms=0,
                memory_bandwidth_gbps=0.0, peak_fp16_tflops=0.0,
                peak_fp32_tflops=0.0, l2_size_mb=0.0,
                driver_version="unknown", cuda_version="unknown",
            )

        return KernelRun(
            id=str(uuid.uuid4()),
            variant=variant,
            shape=shape,
            hardware=hardware,
            metrics=metrics,
            raw_profiler_output_path="",
            timestamp=time.time(),
        )
