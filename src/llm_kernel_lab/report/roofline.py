"""Roofline model computation and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_kernel_lab.model import HardwareProfile, KernelRun


@dataclass
class RooflinePoint:
    """A single point on the roofline plot."""

    kernel_id: str
    shape_id: str
    arithmetic_intensity: float  # FLOPs/byte
    achieved_tflops: float
    is_memory_bound: bool


def compute_roofline_data(
    runs: list[KernelRun],
    hardware: HardwareProfile,
) -> list[RooflinePoint]:
    """Compute roofline points from kernel runs.

    The roofline model shows:
    - X-axis: arithmetic intensity (FLOPs/byte)
    - Y-axis: achieved TFLOP/s
    - Ridge point: where memory roof meets compute roof
    """
    peak_tflops = hardware.peak_fp16_tflops
    peak_bw_tbps = hardware.memory_bandwidth_gbps / 1000.0  # TB/s

    # Ridge point: peak_tflops / peak_bw_tbps
    ridge_intensity = peak_tflops / peak_bw_tbps if peak_bw_tbps > 0 else float("inf")

    points: list[RooflinePoint] = []
    for run in runs:
        ai = run.metrics.arithmetic_intensity
        is_mem_bound = ai < ridge_intensity

        points.append(RooflinePoint(
            kernel_id=run.variant.id,
            shape_id=run.shape.id,
            arithmetic_intensity=ai,
            achieved_tflops=run.metrics.achieved_tflops,
            is_memory_bound=is_mem_bound,
        ))

    return points


def plot_roofline(
    points: list[RooflinePoint],
    hardware: HardwareProfile,
    output_path: str | Path,
    title: str = "Roofline Analysis",
) -> None:
    """Generate a roofline plot and save as PNG."""
    import matplotlib.pyplot as plt
    import numpy as np

    peak_tflops = hardware.peak_fp16_tflops
    peak_bw_tbps = hardware.memory_bandwidth_gbps / 1000.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Memory roof (diagonal line)
    intensities = np.logspace(-2, 4, 200)
    mem_roof = intensities * peak_bw_tbps
    compute_roof = np.full_like(intensities, peak_tflops)
    roofline = np.minimum(mem_roof, compute_roof)

    ax.loglog(intensities, roofline, "k-", linewidth=2, label="Roofline")
    ax.loglog(intensities, mem_roof, "b--", alpha=0.3, label="Memory roof")
    ax.axhline(y=peak_tflops, color="r", linestyle="--", alpha=0.3, label="Compute roof")

    # Plot kernel points
    kernel_ids = sorted(set(p.kernel_id for p in points))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(kernel_ids), 1)))

    for idx, kid in enumerate(kernel_ids):
        kpoints = [p for p in points if p.kernel_id == kid]
        ax.scatter(
            [p.arithmetic_intensity for p in kpoints],
            [p.achieved_tflops for p in kpoints],
            color=colors[idx],
            s=80,
            label=kid,
            zorder=5,
        )

    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Achieved TFLOP/s")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_roofline_report(
    runs: list[KernelRun],
    hardware: HardwareProfile,
    outfile: str | Path,
    plot_path: str | Path | None = None,
) -> None:
    """Generate a complete roofline report (Markdown + plot)."""
    from llm_kernel_lab.report.renderer import render_markdown_report

    points = compute_roofline_data(runs, hardware)

    if plot_path is None:
        outpath = Path(outfile)
        plot_path = outpath.with_suffix(".png")

    try:
        plot_roofline(points, hardware, plot_path)
    except ImportError:
        plot_path = None

    render_markdown_report(runs, hardware, points, outfile, plot_path)
