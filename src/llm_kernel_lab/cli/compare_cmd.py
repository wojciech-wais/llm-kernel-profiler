"""Compare CLI command — compare kernels across results files."""

from __future__ import annotations

import click


@click.command()
@click.option("--inputs", "-i", required=True, help="Comma-separated paths to result JSON files")
@click.option(
    "--metric", "-m",
    default="runtime_ms",
    help="Metric to compare (runtime_ms, achieved_tflops, dram_bw_gbps, etc.)",
)
@click.option("--group-by", default="kernel", type=click.Choice(["kernel", "shape", "gpu"]))
def compare(inputs: str, metric: str, group_by: str) -> None:
    """Compare performance across result files, GPUs, or kernel variants."""
    from llm_kernel_lab.serialization import load_results

    input_paths = [p.strip() for p in inputs.split(",")]

    all_runs = []
    for path in input_paths:
        try:
            hw, runs = load_results(path)
            for run in runs:
                all_runs.append((path, hw, run))
        except FileNotFoundError:
            click.echo(f"Warning: File not found: {path}", err=True)

    if not all_runs:
        click.echo("No results to compare.")
        return

    click.echo(f"Comparing {len(all_runs)} runs by {group_by} on metric: {metric}\n")

    # Group runs
    groups: dict[str, list] = {}
    for path, hw, run in all_runs:
        if group_by == "kernel":
            key = run.variant.id
        elif group_by == "shape":
            key = run.shape.id
        else:
            key = hw.gpu_name
        groups.setdefault(key, []).append((path, hw, run))

    # Print comparison table
    click.echo(f"{'Group':<30} {'Count':>6} {'Min':>12} {'Max':>12} {'Avg':>12}")
    click.echo("-" * 75)

    for group_key, group_runs in sorted(groups.items()):
        values = []
        for _, _, run in group_runs:
            val = getattr(run.metrics, metric, None)
            if val is not None:
                values.append(val)

        if values:
            click.echo(
                f"{group_key:<30} {len(values):>6} "
                f"{min(values):>12.3f} {max(values):>12.3f} "
                f"{sum(values) / len(values):>12.3f}"
            )
