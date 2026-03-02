"""Report CLI command — generate reports from benchmark results."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to results JSON file")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["markdown", "html", "json"]),
    default="markdown",
    help="Report format",
)
@click.option("--out", "-o", default=None, help="Output file path")
def report(input_path: str, fmt: str, out: str | None) -> None:
    """Generate reports from existing benchmark results."""
    from llm_kernel_lab.report.renderer import (
        render_html_report,
        render_json_report,
        render_markdown_report,
    )
    from llm_kernel_lab.report.roofline import compute_roofline_data
    from llm_kernel_lab.serialization import load_results

    hardware, runs = load_results(input_path)

    if not runs:
        click.echo("No runs found in input file.")
        return

    ext_map = {"markdown": ".md", "html": ".html", "json": ".json"}
    if out is None:
        out = f"./reports/report{ext_map[fmt]}"

    click.echo(f"Generating {fmt} report from {len(runs)} runs...")

    if fmt == "markdown":
        points = compute_roofline_data(runs, hardware)
        plot_path = Path(out).with_suffix(".png")
        render_markdown_report(runs, hardware, points, out, plot_path)
    elif fmt == "html":
        render_html_report(runs, hardware, out)
    elif fmt == "json":
        render_json_report(runs, hardware, out)

    click.echo(f"Report saved to {out}")
