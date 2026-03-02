"""Nsight Compute CLI integration for detailed GPU profiling."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from pathlib import Path
from typing import Any

from llm_kernel_lab.model import MetricSet

# Mapping from Nsight Compute metric names to MetricSet fields
METRIC_MAP: dict[str, str] = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_efficiency",
    "smsp__warps_active.avg.pct_of_peak_sustained_active": "occupancy",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "_dram_throughput_pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct": "_mem_global_ld_pct",
    "l2tex__t_bytes.sum": "_l2_bytes_total",
    "smsp__average_warp_latency_per_inst_issued.ratio": "_warp_latency",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block": "shared_mem_bytes",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
}


def check_ncu_available(ncu_path: str = "ncu") -> bool:
    """Check if Nsight Compute CLI is available."""
    path = shutil.which(ncu_path)
    return path is not None


def run_ncu(
    command: list[str],
    ncu_path: str = "ncu",
    metrics: list[str] | None = None,
    export_path: str | None = None,
    timeout: int = 300,
) -> str:
    """Run Nsight Compute CLI and return CSV output.

    Args:
        command: The kernel binary command to profile.
        ncu_path: Path to ncu binary.
        metrics: List of Nsight metric names to collect.
        export_path: Optional path to save .ncu-rep file.
        timeout: Timeout in seconds.

    Returns:
        CSV output from ncu.
    """
    ncu_cmd = [ncu_path, "--target-processes", "all", "--csv"]

    if metrics:
        ncu_cmd.extend(["--metrics", ",".join(metrics)])
    else:
        ncu_cmd.append("--set=full")

    if export_path:
        ncu_cmd.extend(["--export", export_path])

    ncu_cmd.extend(["--import-source", "yes"])
    ncu_cmd.append("--")
    ncu_cmd.extend(command)

    result = subprocess.run(
        ncu_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ncu failed (exit {result.returncode}): {result.stderr}")

    return result.stdout


def parse_ncu_csv(csv_text: str) -> list[dict[str, Any]]:
    """Parse Nsight Compute CSV output into a list of metric dictionaries.

    Each item in the returned list represents one kernel invocation.
    """
    results: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(csv_text))

    current_kernel: dict[str, Any] = {}
    current_name: str | None = None

    for row in reader:
        kernel_name = row.get("Kernel Name", "")
        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "")

        if kernel_name and kernel_name != current_name:
            if current_kernel:
                results.append(current_kernel)
            current_kernel = {"kernel_name": kernel_name}
            current_name = kernel_name

        if metric_name:
            try:
                current_kernel[metric_name] = float(metric_value)
            except (ValueError, TypeError):
                current_kernel[metric_name] = metric_value

    if current_kernel:
        results.append(current_kernel)

    return results


def ncu_metrics_to_metric_set(
    raw: dict[str, Any],
    runtime_ms: float,
    flops: float,
    bytes_accessed: float,
    peak_bw_gbps: float,
) -> MetricSet:
    """Convert raw Nsight Compute metrics to a MetricSet."""
    runtime_s = runtime_ms / 1000.0

    sm_eff_pct = raw.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    occupancy_pct = raw.get("smsp__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    dram_pct = raw.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0.0)
    l2_bytes = raw.get("l2tex__t_bytes.sum", 0.0)
    l2_hit_pct = raw.get("lts__t_sector_hit_rate.pct", 0.0)
    regs = int(raw.get("launch__registers_per_thread", 0))
    smem = int(raw.get("launch__shared_mem_per_block", 0))

    achieved_tflops = (flops / 1e12 / runtime_s) if runtime_s > 0 and flops > 0 else 0.0
    dram_bw_gbps = (dram_pct / 100.0) * peak_bw_gbps if peak_bw_gbps > 0 else 0.0
    l2_bw_gbps = (l2_bytes / 1e9 / runtime_s) if runtime_s > 0 else 0.0
    arithmetic_intensity = (flops / bytes_accessed) if bytes_accessed > 0 else 0.0

    return MetricSet(
        runtime_ms=runtime_ms,
        achieved_tflops=achieved_tflops,
        arithmetic_intensity=arithmetic_intensity,
        dram_bw_gbps=dram_bw_gbps,
        l2_bw_gbps=l2_bw_gbps,
        sm_efficiency=sm_eff_pct / 100.0,
        warp_execution_efficiency=0.0,  # not directly available from default metrics
        eligible_warps_per_cycle=0.0,
        stall_memory_dependency=0.0,
        stall_execution_dependency=0.0,
        stall_sync=0.0,
        occupancy=occupancy_pct / 100.0,
        registers_per_thread=regs,
        shared_mem_bytes=smem,
        l2_hit_rate=l2_hit_pct / 100.0,
    )
