"""Configuration loading for LLM Kernel Lab."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_CONFIG_FILENAME = "llm-kernel-lab.toml"


@dataclass
class GpuConfig:
    peak_fp16_tflops: float | None = None
    peak_fp32_tflops: float | None = None
    memory_bandwidth_gbps: float | None = None


@dataclass
class NsightConfig:
    path: str = "/usr/local/cuda/bin/ncu"
    metrics: list[str] = field(default_factory=lambda: [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ])


@dataclass
class BenchConfig:
    default_repetitions: int = 50
    warmup: int = 10
    output_dir: str = "./results"


@dataclass
class LabConfig:
    gpu: GpuConfig = field(default_factory=GpuConfig)
    nsight: NsightConfig = field(default_factory=NsightConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)


def load_config(path: str | Path | None = None) -> LabConfig:
    """Load configuration from a TOML file.

    If no path is provided, searches for llm-kernel-lab.toml in the current directory
    and parent directories.
    """
    if path is None:
        path = _find_config()
    if path is None:
        return LabConfig()

    path = Path(path)
    if not path.is_file():
        return LabConfig()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return _parse_config(raw)


def _find_config() -> Path | None:
    cwd = Path.cwd()
    for directory in [cwd, *cwd.parents]:
        candidate = directory / DEFAULT_CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _parse_config(raw: dict[str, Any]) -> LabConfig:
    config = LabConfig()

    if "gpu" in raw:
        gpu = raw["gpu"]
        config.gpu = GpuConfig(
            peak_fp16_tflops=gpu.get("peak_fp16_tflops"),
            peak_fp32_tflops=gpu.get("peak_fp32_tflops"),
            memory_bandwidth_gbps=gpu.get("memory_bandwidth_gbps"),
        )

    if "nsight" in raw:
        ns = raw["nsight"]
        config.nsight = NsightConfig(
            path=ns.get("path", config.nsight.path),
            metrics=ns.get("metrics", config.nsight.metrics),
        )

    if "bench" in raw:
        b = raw["bench"]
        config.bench = BenchConfig(
            default_repetitions=b.get("default_repetitions", config.bench.default_repetitions),
            warmup=b.get("warmup", config.bench.warmup),
            output_dir=b.get("output_dir", config.bench.output_dir),
        )

    return config
