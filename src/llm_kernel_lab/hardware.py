"""Hardware detection and profiling."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from llm_kernel_lab.model import HardwareProfile

# Known GPU specs for common models (fallback when queries fail)
_KNOWN_GPUS: dict[str, dict[str, Any]] = {
    "NVIDIA A100": {
        "sm_version": 80,
        "peak_fp16_tflops": 312.0,
        "peak_fp32_tflops": 19.5,
        "memory_bandwidth_gbps": 2039.0,
    },
    "NVIDIA H100": {
        "sm_version": 90,
        "peak_fp16_tflops": 989.0,
        "peak_fp32_tflops": 67.0,
        "memory_bandwidth_gbps": 3350.0,
    },
    "NVIDIA L40": {
        "sm_version": 89,
        "peak_fp16_tflops": 181.0,
        "peak_fp32_tflops": 90.5,
        "memory_bandwidth_gbps": 864.0,
    },
}


def detect_hardware(device_index: int = 0) -> HardwareProfile:
    """Detect GPU hardware profile using nvidia-smi and CUDA APIs.

    Falls back to known specs for common GPUs if detailed queries fail.
    """
    smi = _query_nvidia_smi(device_index)

    gpu_name = smi.get("gpu_name", "Unknown GPU")
    driver_version = smi.get("driver_version", "unknown")

    # Try to get specs from known GPUs as defaults
    known = _lookup_known_gpu(gpu_name)

    cuda_version = _detect_cuda_version()

    return HardwareProfile(
        gpu_name=gpu_name,
        sm_version=known.get("sm_version", 0),
        num_sms=smi.get("num_sms", 0),
        memory_bandwidth_gbps=known.get("memory_bandwidth_gbps", 0.0),
        peak_fp16_tflops=known.get("peak_fp16_tflops", 0.0),
        peak_fp32_tflops=known.get("peak_fp32_tflops", 0.0),
        l2_size_mb=smi.get("l2_size_mb", 0.0),
        driver_version=driver_version,
        cuda_version=cuda_version,
    )


def _query_nvidia_smi(device_index: int) -> dict[str, Any]:
    """Query nvidia-smi for GPU information."""
    if not shutil.which("nvidia-smi"):
        return {}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=name,driver_version,memory.total,clocks.max.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 2:
            return {}

        info: dict[str, Any] = {
            "gpu_name": parts[0],
            "driver_version": parts[1],
        }

        # SM count requires deviceQuery or CUDA API; leave as 0 for now
        info["num_sms"] = 0
        info["l2_size_mb"] = 0.0

        return info
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}


def _detect_cuda_version() -> str:
    """Detect CUDA version from nvcc or nvidia-smi."""
    if shutil.which("nvcc"):
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.2, V12.2.140"
                    parts = line.split("release")
                    if len(parts) > 1:
                        return parts[1].strip().rstrip(",").split(",")[0].strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.splitlines():
                if "CUDA Version" in line:
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        return parts[1].strip().split()[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    return "unknown"


def _lookup_known_gpu(gpu_name: str) -> dict[str, Any]:
    """Look up known GPU specs by name prefix matching."""
    for known_name, specs in _KNOWN_GPUS.items():
        if known_name.lower() in gpu_name.lower():
            return specs
    return {}


def hardware_profile_to_json(hw: HardwareProfile) -> str:
    """Serialize hardware profile to JSON string."""
    from llm_kernel_lab.serialization import hardware_profile_to_dict

    return json.dumps(hardware_profile_to_dict(hw), indent=2)
