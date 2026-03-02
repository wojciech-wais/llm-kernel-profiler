"""Core data model for LLM Kernel Lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KernelVariant:
    """A specific implementation of a GPU kernel operation."""

    id: str
    op_type: str  # "attention", "mlp", "matmul", "layernorm", "quant_gemm", ...
    backend: str  # "triton", "cuda", "cublas", "cudnn", "cutlass", ...
    source: str  # module path or binary path
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemShape:
    """A workload configuration (e.g. attention with specific batch/heads/seq len)."""

    id: str
    op_type: str  # matches KernelVariant.op_type
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareProfile:
    """Description of the GPU hardware."""

    gpu_name: str
    sm_version: int
    num_sms: int
    memory_bandwidth_gbps: float
    peak_fp16_tflops: float
    peak_fp32_tflops: float
    l2_size_mb: float
    driver_version: str
    cuda_version: str


@dataclass
class MetricSet:
    """Collected performance metrics for a kernel run."""

    runtime_ms: float
    achieved_tflops: float
    arithmetic_intensity: float  # FLOPs / bytes transferred
    dram_bw_gbps: float
    l2_bw_gbps: float
    sm_efficiency: float  # 0..1
    warp_execution_efficiency: float  # 0..1
    eligible_warps_per_cycle: float
    stall_memory_dependency: float  # 0..1
    stall_execution_dependency: float  # 0..1
    stall_sync: float  # 0..1
    occupancy: float  # 0..1
    registers_per_thread: int
    shared_mem_bytes: int
    l2_hit_rate: float  # 0..1
    power_watts: float | None = None
    temperature_c: float | None = None


@dataclass
class KernelRun:
    """A single profiled kernel execution."""

    id: str
    variant: KernelVariant
    shape: ProblemShape
    hardware: HardwareProfile
    metrics: MetricSet
    raw_profiler_output_path: str
    timestamp: float
