"""Tests for configuration loading."""

import tempfile
from pathlib import Path

from llm_kernel_lab.config import LabConfig, _parse_config, load_config


def test_default_config():
    config = LabConfig()
    assert config.bench.default_repetitions == 50
    assert config.bench.warmup == 10
    assert config.nsight.path == "/usr/local/cuda/bin/ncu"
    assert config.gpu.peak_fp16_tflops is None


def test_parse_config_full():
    raw = {
        "gpu": {
            "peak_fp16_tflops": 312.0,
            "peak_fp32_tflops": 78.0,
            "memory_bandwidth_gbps": 1555.0,
        },
        "nsight": {
            "path": "/opt/cuda/bin/ncu",
            "metrics": ["metric_a", "metric_b"],
        },
        "bench": {
            "default_repetitions": 200,
            "warmup": 20,
            "output_dir": "/tmp/results",
        },
    }
    config = _parse_config(raw)
    assert config.gpu.peak_fp16_tflops == 312.0
    assert config.nsight.path == "/opt/cuda/bin/ncu"
    assert config.nsight.metrics == ["metric_a", "metric_b"]
    assert config.bench.default_repetitions == 200
    assert config.bench.output_dir == "/tmp/results"


def test_parse_config_partial():
    raw = {"bench": {"warmup": 5}}
    config = _parse_config(raw)
    assert config.bench.warmup == 5
    assert config.bench.default_repetitions == 50  # default
    assert config.gpu.peak_fp16_tflops is None


def test_load_config_nonexistent_returns_defaults():
    config = load_config("/nonexistent/path/config.toml")
    # Should not raise, returns defaults
    assert config.bench.default_repetitions == 50


def test_load_config_from_file():
    toml_content = b"""\
[bench]
default_repetitions = 77
warmup = 3
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(f.name)

    assert config.bench.default_repetitions == 77
    assert config.bench.warmup == 3
