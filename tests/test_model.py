"""Tests for the core data model."""

from llm_kernel_lab.model import (
    HardwareProfile,
    KernelRun,
    KernelVariant,
    MetricSet,
    ProblemShape,
)


def make_hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="NVIDIA A100-PCIE-40GB",
        sm_version=80,
        num_sms=108,
        memory_bandwidth_gbps=1555.0,
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
        l2_size_mb=40.0,
        driver_version="535.129.03",
        cuda_version="12.2",
    )


def make_variant() -> KernelVariant:
    return KernelVariant(
        id="flash_attn_v2_triton",
        op_type="attention",
        backend="triton",
        source="kernels.flash_attn_v2",
        config={"BLOCK_M": 128, "BLOCK_N": 128},
        metadata={"version": "2.0"},
    )


def make_shape() -> ProblemShape:
    return ProblemShape(
        id="gpt3_175b_long",
        op_type="attention",
        params={"B": 8, "H": 16, "S_q": 4096, "S_kv": 4096, "D": 128},
    )


def make_metrics() -> MetricSet:
    return MetricSet(
        runtime_ms=3.21,
        achieved_tflops=225.0,
        arithmetic_intensity=50.0,
        dram_bw_gbps=1200.0,
        l2_bw_gbps=2500.0,
        sm_efficiency=0.78,
        warp_execution_efficiency=0.85,
        eligible_warps_per_cycle=12.0,
        stall_memory_dependency=0.15,
        stall_execution_dependency=0.05,
        stall_sync=0.02,
        occupancy=0.72,
        registers_per_thread=64,
        shared_mem_bytes=32768,
        l2_hit_rate=0.85,
        power_watts=250.0,
        temperature_c=65.0,
    )


def test_kernel_variant_creation():
    v = make_variant()
    assert v.id == "flash_attn_v2_triton"
    assert v.op_type == "attention"
    assert v.backend == "triton"
    assert v.config["BLOCK_M"] == 128


def test_problem_shape_creation():
    s = make_shape()
    assert s.id == "gpt3_175b_long"
    assert s.params["B"] == 8
    assert s.params["S_q"] == 4096


def test_hardware_profile_creation():
    hw = make_hw()
    assert hw.gpu_name == "NVIDIA A100-PCIE-40GB"
    assert hw.sm_version == 80
    assert hw.peak_fp16_tflops == 312.0


def test_metric_set_creation():
    m = make_metrics()
    assert m.runtime_ms == 3.21
    assert m.achieved_tflops == 225.0
    assert m.power_watts == 250.0


def test_metric_set_optional_fields():
    m = MetricSet(
        runtime_ms=1.0,
        achieved_tflops=100.0,
        arithmetic_intensity=10.0,
        dram_bw_gbps=500.0,
        l2_bw_gbps=1000.0,
        sm_efficiency=0.5,
        warp_execution_efficiency=0.5,
        eligible_warps_per_cycle=8.0,
        stall_memory_dependency=0.0,
        stall_execution_dependency=0.0,
        stall_sync=0.0,
        occupancy=0.5,
        registers_per_thread=32,
        shared_mem_bytes=16384,
        l2_hit_rate=0.5,
    )
    assert m.power_watts is None
    assert m.temperature_c is None


def test_kernel_run_creation():
    run = KernelRun(
        id="run_001",
        variant=make_variant(),
        shape=make_shape(),
        hardware=make_hw(),
        metrics=make_metrics(),
        raw_profiler_output_path="results/test.ncu-rep",
        timestamp=1700000000.0,
    )
    assert run.id == "run_001"
    assert run.variant.id == "flash_attn_v2_triton"
    assert run.metrics.runtime_ms == 3.21
