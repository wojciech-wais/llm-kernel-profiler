"""JSON serialization and deserialization for the data model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_kernel_lab.model import (
    HardwareProfile,
    KernelRun,
    KernelVariant,
    MetricSet,
    ProblemShape,
)


def _serialize_value(v: Any) -> Any:
    if v is None:
        return None
    return v


def kernel_variant_to_dict(variant: KernelVariant) -> dict[str, Any]:
    return asdict(variant)


def kernel_variant_from_dict(d: dict[str, Any]) -> KernelVariant:
    return KernelVariant(**d)


def problem_shape_to_dict(shape: ProblemShape) -> dict[str, Any]:
    return asdict(shape)


def problem_shape_from_dict(d: dict[str, Any]) -> ProblemShape:
    return ProblemShape(**d)


def hardware_profile_to_dict(hw: HardwareProfile) -> dict[str, Any]:
    return asdict(hw)


def hardware_profile_from_dict(d: dict[str, Any]) -> HardwareProfile:
    return HardwareProfile(**d)


def metric_set_to_dict(metrics: MetricSet) -> dict[str, Any]:
    return asdict(metrics)


def metric_set_from_dict(d: dict[str, Any]) -> MetricSet:
    return MetricSet(**d)


def kernel_run_to_dict(run: KernelRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "variant": kernel_variant_to_dict(run.variant),
        "shape": problem_shape_to_dict(run.shape),
        "hardware": hardware_profile_to_dict(run.hardware),
        "metrics": metric_set_to_dict(run.metrics),
        "raw_profiler_output_path": run.raw_profiler_output_path,
        "timestamp": run.timestamp,
    }


def kernel_run_from_dict(d: dict[str, Any]) -> KernelRun:
    return KernelRun(
        id=d["id"],
        variant=kernel_variant_from_dict(d["variant"]),
        shape=problem_shape_from_dict(d["shape"]),
        hardware=hardware_profile_from_dict(d["hardware"]),
        metrics=metric_set_from_dict(d["metrics"]),
        raw_profiler_output_path=d["raw_profiler_output_path"],
        timestamp=d["timestamp"],
    )


def save_results(
    runs: list[KernelRun],
    hardware: HardwareProfile,
    path: str | Path,
) -> None:
    """Save benchmark results to a JSON file."""
    data = {
        "hardware": hardware_profile_to_dict(hardware),
        "runs": [kernel_run_to_dict(r) for r in runs],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_results(path: str | Path) -> tuple[HardwareProfile, list[KernelRun]]:
    """Load benchmark results from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    hardware = hardware_profile_from_dict(data["hardware"])
    runs = [kernel_run_from_dict(r) for r in data["runs"]]
    return hardware, runs
