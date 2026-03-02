"""Tests for CLI argument parsing and commands."""

from click.testing import CliRunner

from llm_kernel_lab.cli.main import cli


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "bench" in result.output
    assert "report" in result.output
    assert "compare" in result.output
    assert "ci" in result.output


def test_bench_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["bench", "--help"])
    assert result.exit_code == 0
    assert "--op" in result.output
    assert "--kernels" in result.output
    assert "--profile-level" in result.output


def test_report_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--format" in result.output


def test_compare_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--inputs" in result.output
    assert "--metric" in result.output


def test_ci_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["ci", "--help"])
    assert result.exit_code == 0
    assert "--baseline" in result.output
    assert "--max-regression" in result.output


def test_record_hw_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["record-hw", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.output


def test_list_shapes():
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "shapes"])
    assert result.exit_code == 0
    assert "attention" in result.output
    assert "matmul" in result.output


def test_init_command(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "llm-kernel-lab.toml").exists()
