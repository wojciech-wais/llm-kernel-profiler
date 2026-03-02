"""Tests for bottleneck classification."""

from llm_kernel_lab.model import HardwareProfile, MetricSet
from llm_kernel_lab.report.bottleneck import Bottleneck, classify_bottleneck


def _hw() -> HardwareProfile:
    return HardwareProfile(
        gpu_name="Test GPU",
        sm_version=80,
        num_sms=108,
        memory_bandwidth_gbps=1555.0,
        peak_fp16_tflops=312.0,
        peak_fp32_tflops=19.5,
        l2_size_mb=40.0,
        driver_version="535.0",
        cuda_version="12.2",
    )


def _metrics(**overrides) -> MetricSet:
    defaults = dict(
        runtime_ms=1.0,
        achieved_tflops=100.0,
        arithmetic_intensity=10.0,
        dram_bw_gbps=500.0,
        l2_bw_gbps=1000.0,
        sm_efficiency=0.5,
        warp_execution_efficiency=0.5,
        eligible_warps_per_cycle=8.0,
        stall_memory_dependency=0.1,
        stall_execution_dependency=0.1,
        stall_sync=0.05,
        occupancy=0.5,
        registers_per_thread=32,
        shared_mem_bytes=16384,
        l2_hit_rate=0.5,
    )
    defaults.update(overrides)
    return MetricSet(**defaults)


def test_memory_bound():
    m = _metrics(dram_bw_gbps=1400.0, sm_efficiency=0.5)
    assert classify_bottleneck(m, _hw()) == Bottleneck.MEMORY_BOUND


def test_latency_bound():
    m = _metrics(
        dram_bw_gbps=500.0,
        sm_efficiency=0.3,
        stall_execution_dependency=0.5,
    )
    assert classify_bottleneck(m, _hw()) == Bottleneck.LATENCY_BOUND


def test_compute_bound():
    m = _metrics(
        dram_bw_gbps=500.0,
        sm_efficiency=0.9,
        achieved_tflops=250.0,
    )
    assert classify_bottleneck(m, _hw()) == Bottleneck.COMPUTE_BOUND


def test_balanced():
    m = _metrics(dram_bw_gbps=800.0, sm_efficiency=0.6)
    assert classify_bottleneck(m, _hw()) == Bottleneck.BALANCED


def test_unknown_when_no_data():
    m = _metrics(dram_bw_gbps=0.0, sm_efficiency=0.0)
    hw = HardwareProfile(
        gpu_name="Unknown",
        sm_version=0,
        num_sms=0,
        memory_bandwidth_gbps=0.0,
        peak_fp16_tflops=0.0,
        peak_fp32_tflops=0.0,
        l2_size_mb=0.0,
        driver_version="",
        cuda_version="",
    )
    assert classify_bottleneck(m, hw) == Bottleneck.UNKNOWN
