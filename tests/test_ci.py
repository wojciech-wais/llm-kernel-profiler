"""Tests for CI regression detection."""

import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from llm_kernel_lab.cli.main import cli
from llm_kernel_lab.model import KernelRun
from llm_kernel_lab.serialization import save_results
from tests.test_model import make_hw, make_metrics, make_shape, make_variant


def _make_run(runtime_ms: float) -> KernelRun:
    metrics = make_metrics()
    # Override runtime
    object.__setattr__(metrics, "runtime_ms", runtime_ms)
    return KernelRun(
        id="test_run",
        variant=make_variant(),
        shape=make_shape(),
        hardware=make_hw(),
        metrics=metrics,
        raw_profiler_output_path="",
        timestamp=1700000000.0,
    )


def test_ci_no_regression():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "baseline.json"
        current_path = Path(tmpdir) / "current.json"

        hw = make_hw()
        save_results([_make_run(3.0)], hw, baseline_path)
        save_results([_make_run(3.1)], hw, current_path)  # ~3% increase, under 10%

        runner = CliRunner()
        result = runner.invoke(cli, [
            "ci",
            "--baseline", str(baseline_path),
            "--current", str(current_path),
            "--max-regression", "0.10",
        ])
        assert result.exit_code == 0
        assert "within regression threshold" in result.output.lower()


def test_ci_with_regression():
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "baseline.json"
        current_path = Path(tmpdir) / "current.json"

        hw = make_hw()
        save_results([_make_run(3.0)], hw, baseline_path)
        save_results([_make_run(4.0)], hw, current_path)  # 33% regression

        runner = CliRunner()
        result = runner.invoke(cli, [
            "ci",
            "--baseline", str(baseline_path),
            "--current", str(current_path),
            "--max-regression", "0.10",
        ])
        assert result.exit_code == 1
        assert "regression" in result.output.lower()
