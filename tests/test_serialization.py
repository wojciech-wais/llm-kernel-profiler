"""Tests for JSON serialization and deserialization."""

import json
import tempfile
from pathlib import Path

from llm_kernel_lab.model import (
    KernelRun,
)
from llm_kernel_lab.serialization import (
    hardware_profile_from_dict,
    hardware_profile_to_dict,
    kernel_run_from_dict,
    kernel_run_to_dict,
    kernel_variant_from_dict,
    kernel_variant_to_dict,
    load_results,
    metric_set_from_dict,
    metric_set_to_dict,
    problem_shape_from_dict,
    problem_shape_to_dict,
    save_results,
)
from tests.test_model import make_hw, make_metrics, make_shape, make_variant


def test_kernel_variant_roundtrip():
    v = make_variant()
    d = kernel_variant_to_dict(v)
    v2 = kernel_variant_from_dict(d)
    assert v2.id == v.id
    assert v2.config == v.config


def test_problem_shape_roundtrip():
    s = make_shape()
    d = problem_shape_to_dict(s)
    s2 = problem_shape_from_dict(d)
    assert s2.id == s.id
    assert s2.params == s.params


def test_hardware_profile_roundtrip():
    hw = make_hw()
    d = hardware_profile_to_dict(hw)
    hw2 = hardware_profile_from_dict(d)
    assert hw2.gpu_name == hw.gpu_name
    assert hw2.peak_fp16_tflops == hw.peak_fp16_tflops


def test_metric_set_roundtrip():
    m = make_metrics()
    d = metric_set_to_dict(m)
    m2 = metric_set_from_dict(d)
    assert m2.runtime_ms == m.runtime_ms
    assert m2.power_watts == m.power_watts


def test_kernel_run_roundtrip():
    run = KernelRun(
        id="test_run",
        variant=make_variant(),
        shape=make_shape(),
        hardware=make_hw(),
        metrics=make_metrics(),
        raw_profiler_output_path="test.ncu-rep",
        timestamp=1700000000.0,
    )
    d = kernel_run_to_dict(run)
    run2 = kernel_run_from_dict(d)
    assert run2.id == run.id
    assert run2.variant.id == run.variant.id
    assert run2.metrics.runtime_ms == run.metrics.runtime_ms


def test_json_serializable():
    run = KernelRun(
        id="test_run",
        variant=make_variant(),
        shape=make_shape(),
        hardware=make_hw(),
        metrics=make_metrics(),
        raw_profiler_output_path="test.ncu-rep",
        timestamp=1700000000.0,
    )
    d = kernel_run_to_dict(run)
    # Should not raise
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_save_and_load_results():
    hw = make_hw()
    run = KernelRun(
        id="test_run",
        variant=make_variant(),
        shape=make_shape(),
        hardware=hw,
        metrics=make_metrics(),
        raw_profiler_output_path="test.ncu-rep",
        timestamp=1700000000.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "results.json"
        save_results([run], hw, path)
        assert path.exists()

        hw2, runs2 = load_results(path)
        assert hw2.gpu_name == hw.gpu_name
        assert len(runs2) == 1
        assert runs2[0].id == "test_run"
        assert runs2[0].metrics.achieved_tflops == 225.0
