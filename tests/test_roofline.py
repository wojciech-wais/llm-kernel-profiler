"""Tests for roofline computation."""

from llm_kernel_lab.model import HardwareProfile, KernelRun, KernelVariant, MetricSet, ProblemShape
from llm_kernel_lab.report.roofline import compute_roofline_data


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


def _make_run(ai: float, tflops: float) -> KernelRun:
    return KernelRun(
        id="test",
        variant=KernelVariant(
            id="test_kernel", op_type="matmul", backend="triton", source="test"
        ),
        shape=ProblemShape(id="test_shape", op_type="matmul"),
        hardware=_hw(),
        metrics=MetricSet(
            runtime_ms=1.0,
            achieved_tflops=tflops,
            arithmetic_intensity=ai,
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
        ),
        raw_profiler_output_path="",
        timestamp=0.0,
    )


def test_roofline_memory_bound_point():
    hw = _hw()
    # Ridge point = 312 / (1555/1000) = 312 / 1.555 ≈ 200.6
    # AI = 10 is well below ridge -> memory bound
    run = _make_run(ai=10.0, tflops=15.0)
    points = compute_roofline_data([run], hw)
    assert len(points) == 1
    assert points[0].is_memory_bound is True


def test_roofline_compute_bound_point():
    hw = _hw()
    # AI = 500 is well above ridge -> compute bound
    run = _make_run(ai=500.0, tflops=280.0)
    points = compute_roofline_data([run], hw)
    assert len(points) == 1
    assert points[0].is_memory_bound is False


def test_roofline_multiple_runs():
    hw = _hw()
    runs = [
        _make_run(ai=5.0, tflops=7.0),
        _make_run(ai=300.0, tflops=250.0),
    ]
    points = compute_roofline_data(runs, hw)
    assert len(points) == 2
    assert points[0].is_memory_bound is True
    assert points[1].is_memory_bound is False
