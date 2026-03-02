"""Benchmark runner that orchestrates kernel profiling across shapes and variants."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from llm_kernel_lab.config import LabConfig, load_config
from llm_kernel_lab.hardware import detect_hardware
from llm_kernel_lab.model import (
    HardwareProfile,
    KernelRun,
    KernelVariant,
    ProblemShape,
)
from llm_kernel_lab.profiler.metrics import estimate_bytes_accessed, estimate_flops
from llm_kernel_lab.profiler.profiler import Profiler


class BenchmarkRunner:
    """Orchestrates benchmarks across kernel variants and problem shapes."""

    def __init__(
        self,
        backend: str = "triton",
        device: str = "cuda:0",
        config: LabConfig | None = None,
        hardware: HardwareProfile | None = None,
    ) -> None:
        self.backend = backend
        self.device = device
        self.config = config or load_config()
        self.hardware_profile = hardware

    def _get_hardware(self) -> HardwareProfile:
        if self.hardware_profile is None:
            device_idx = 0
            if ":" in self.device:
                try:
                    device_idx = int(self.device.split(":")[1])
                except ValueError:
                    pass
            self.hardware_profile = detect_hardware(device_idx)
        return self.hardware_profile

    def run(
        self,
        kernels: list[KernelVariant],
        shapes: list[ProblemShape],
        repetitions: int | None = None,
        warmup: int | None = None,
        profile_level: str = "timing",
        kernel_fns: dict[str, Callable[..., Any]] | None = None,
    ) -> list[KernelRun]:
        """Run benchmarks for all kernel/shape combinations.

        Args:
            kernels: Kernel variants to benchmark.
            shapes: Problem shapes to test.
            repetitions: Number of timed iterations per benchmark.
            warmup: Number of warmup iterations.
            profile_level: One of "none", "timing", "basic", "full".
            kernel_fns: Mapping of kernel variant IDs to callable functions.

        Returns:
            List of KernelRun results.
        """
        reps = repetitions or self.config.bench.default_repetitions
        wu = warmup or self.config.bench.warmup
        hardware = self._get_hardware()

        profiler = Profiler(
            profile_level=profile_level,
            warmup=wu,
            repetitions=reps,
            ncu_path=self.config.nsight.path,
        )

        results: list[KernelRun] = []
        kernel_fns = kernel_fns or {}

        for variant in kernels:
            for shape in shapes:
                if shape.op_type != variant.op_type:
                    continue

                fn = kernel_fns.get(variant.id)
                if fn is None:
                    fn = _resolve_kernel_fn(variant)
                if fn is None:
                    continue

                flops = estimate_flops(shape.params, shape.op_type)
                bytes_acc = estimate_bytes_accessed(shape.params, shape.op_type)

                run = profiler.profile_kernel(
                    kernel_fn=fn,
                    flops=flops,
                    bytes_accessed=bytes_acc,
                    variant=variant,
                    shape=shape,
                    hardware=hardware,
                )
                results.append(run)

        return results


def _resolve_kernel_fn(variant: KernelVariant) -> Callable | None:
    """Try to import and resolve a kernel function from its source path."""
    try:
        parts = variant.source.rsplit(".", 1)
        if len(parts) == 2:
            import importlib

            module = importlib.import_module(parts[0])
            return getattr(module, parts[1], None)
    except (ImportError, AttributeError):
        pass
    return None
