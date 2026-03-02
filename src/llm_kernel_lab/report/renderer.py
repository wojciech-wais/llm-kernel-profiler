"""Report rendering in Markdown, HTML, and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_kernel_lab.model import HardwareProfile, KernelRun
from llm_kernel_lab.report.bottleneck import classify_bottleneck_from_run
from llm_kernel_lab.report.recommendations import get_recommendations
from llm_kernel_lab.report.roofline import RooflinePoint
from llm_kernel_lab.serialization import kernel_run_to_dict


def render_markdown_report(
    runs: list[KernelRun],
    hardware: HardwareProfile,
    roofline_points: list[RooflinePoint] | None = None,
    outfile: str | Path = "report.md",
    plot_path: str | Path | None = None,
) -> None:
    """Render a Markdown report with summary table, roofline, and recommendations."""
    lines: list[str] = []
    lines.append("# LLM Kernel Lab — Benchmark Report\n")
    lines.append(f"**GPU**: {hardware.gpu_name}  ")
    lines.append(f"**CUDA**: {hardware.cuda_version}  ")
    lines.append(f"**Driver**: {hardware.driver_version}  ")
    lines.append(f"**Peak FP16**: {hardware.peak_fp16_tflops} TFLOP/s  ")
    lines.append(f"**Memory BW**: {hardware.memory_bandwidth_gbps} GB/s\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append(
        "| Kernel Variant | Shape ID | Runtime (ms) | TFLOP/s | DRAM BW (GB/s) "
        "| SM Eff. | Occupancy | Bottleneck |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---|"
    )

    for run in runs:
        bottleneck = classify_bottleneck_from_run(run)
        m = run.metrics
        lines.append(
            f"| {run.variant.id} | {run.shape.id} | {m.runtime_ms:.2f} "
            f"| {m.achieved_tflops:.1f} | {m.dram_bw_gbps:.1f} "
            f"| {m.sm_efficiency:.2f} | {m.occupancy:.2f} | {bottleneck} |"
        )

    # Roofline plot
    if plot_path is not None:
        lines.append("\n## Roofline Analysis\n")
        lines.append(f"![Roofline Plot]({plot_path})\n")

    # Recommendations
    lines.append("\n## Recommendations\n")
    for run in runs:
        recs = get_recommendations(run.metrics, run.hardware)
        lines.append(f"### {run.variant.id} — {run.shape.id}\n")
        for rec in recs:
            lines.append(f"- {rec}")
        lines.append("")

    outpath = Path(outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines))


def render_html_report(
    runs: list[KernelRun],
    hardware: HardwareProfile,
    outfile: str | Path = "report.html",
    plot_path: str | Path | None = None,
) -> None:
    """Render an HTML report."""
    # Convert Markdown to simple HTML
    md_path = Path(outfile).with_suffix(".md")
    render_markdown_report(runs, hardware, outfile=md_path, plot_path=plot_path)

    md_content = md_path.read_text()
    html_lines = [
        "<!DOCTYPE html>",
        "<html><head>",
        '<meta charset="utf-8">',
        "<title>LLM Kernel Lab Report</title>",
        "<style>",
        "body { font-family: sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background: #f5f5f5; }",
        "pre { background: #f8f8f8; padding: 12px; overflow-x: auto; }",
        "</style>",
        "</head><body>",
        f"<pre>{md_content}</pre>",
        "</body></html>",
    ]

    Path(outfile).write_text("\n".join(html_lines))


def render_json_report(
    runs: list[KernelRun],
    hardware: HardwareProfile,
    outfile: str | Path = "report.json",
) -> None:
    """Render a JSON report."""
    from llm_kernel_lab.serialization import hardware_profile_to_dict

    data: dict[str, Any] = {
        "hardware": hardware_profile_to_dict(hardware),
        "runs": [],
    }

    for run in runs:
        run_data = kernel_run_to_dict(run)
        run_data["bottleneck"] = classify_bottleneck_from_run(run)
        run_data["recommendations"] = get_recommendations(run.metrics, run.hardware)
        data["runs"].append(run_data)

    outpath = Path(outfile)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
