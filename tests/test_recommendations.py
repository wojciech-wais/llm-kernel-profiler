"""Tests for the recommendation engine."""

from llm_kernel_lab.model import HardwareProfile, MetricSet
from llm_kernel_lab.report.recommendations import get_recommendations


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


def test_memory_bound_recommendations():
    m = _metrics(dram_bw_gbps=1400.0, sm_efficiency=0.5)
    recs = get_recommendations(m, _hw())
    assert any("memory-bound" in r.lower() for r in recs)
    assert any("tile" in r.lower() or "cp.async" in r.lower() for r in recs)


def test_memory_bound_low_l2_hit():
    m = _metrics(dram_bw_gbps=1400.0, sm_efficiency=0.5, l2_hit_rate=0.3)
    recs = get_recommendations(m, _hw())
    assert any("l2 hit rate" in r.lower() for r in recs)


def test_latency_bound_recommendations():
    m = _metrics(sm_efficiency=0.3, stall_execution_dependency=0.5)
    recs = get_recommendations(m, _hw())
    assert any("latency-bound" in r.lower() for r in recs)


def test_latency_bound_sync_stalls():
    m = _metrics(sm_efficiency=0.3, stall_execution_dependency=0.5, stall_sync=0.3)
    recs = get_recommendations(m, _hw())
    assert any("synchronization" in r.lower() for r in recs)


def test_low_occupancy_recommendation():
    m = _metrics(occupancy=0.3)
    recs = get_recommendations(m, _hw())
    assert any("occupancy" in r.lower() for r in recs)


def test_high_register_recommendation():
    m = _metrics(registers_per_thread=200)
    recs = get_recommendations(m, _hw())
    assert any("register" in r.lower() for r in recs)


def test_no_data_gives_generic_recommendation():
    m = _metrics(
        sm_efficiency=0.0,
        dram_bw_gbps=0.0,
        occupancy=0.0,
        registers_per_thread=0,
    )
    hw = HardwareProfile(
        gpu_name="Unknown", sm_version=0, num_sms=0,
        memory_bandwidth_gbps=0.0, peak_fp16_tflops=0.0, peak_fp32_tflops=0.0,
        l2_size_mb=0.0, driver_version="", cuda_version="",
    )
    recs = get_recommendations(m, hw)
    assert len(recs) >= 1
