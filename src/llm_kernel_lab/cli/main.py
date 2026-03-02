"""Main CLI entry point for llm-kernel-lab."""

from __future__ import annotations

import click

from llm_kernel_lab import __version__
from llm_kernel_lab.cli.bench_cmd import bench
from llm_kernel_lab.cli.ci_cmd import ci
from llm_kernel_lab.cli.compare_cmd import compare
from llm_kernel_lab.cli.hw_cmd import record_hw
from llm_kernel_lab.cli.init_cmd import init
from llm_kernel_lab.cli.list_cmd import list_items
from llm_kernel_lab.cli.report_cmd import report


@click.group()
@click.version_option(version=__version__, prog_name="llm-kernel-lab")
def cli() -> None:
    """LLM Kernel Lab — Profile and optimize GPU kernels for LLM inference."""


cli.add_command(bench)
cli.add_command(report)
cli.add_command(compare)
cli.add_command(record_hw)
cli.add_command(ci)
cli.add_command(init)
cli.add_command(list_items)


if __name__ == "__main__":
    cli()
