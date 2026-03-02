"""CI CLI command — regression detection for CI pipelines."""

from __future__ import annotations

import sys

import click


@click.command()
@click.option("--baseline", "-b", required=True, help="Path to baseline results JSON")
@click.option("--current", "-c", required=True, help="Path to current results JSON")
@click.option("--metric", "-m", default="runtime_ms", help="Metric to compare")
@click.option("--max-regression", default=0.10, help="Maximum allowed regression ratio (e.g. 0.10 = 10%)")
def ci(baseline: str, current: str, metric: str, max_regression: float) -> None:
    """Run regression detection for CI pipelines.

    Compares current results against a baseline and fails if any kernel
    regresses beyond the threshold.
    """
    from llm_kernel_lab.serialization import load_results

    _, baseline_runs = load_results(baseline)
    _, current_runs = load_results(current)

    # Build lookup by (variant_id, shape_id)
    baseline_map: dict[tuple[str, str], float] = {}
    for run in baseline_runs:
        key = (run.variant.id, run.shape.id)
        val = getattr(run.metrics, metric, None)
        if val is not None:
            baseline_map[key] = val

    current_map: dict[tuple[str, str], float] = {}
    for run in current_runs:
        key = (run.variant.id, run.shape.id)
        val = getattr(run.metrics, metric, None)
        if val is not None:
            current_map[key] = val

    regressions: list[tuple[str, str, float, float, float]] = []
    passed = 0
    skipped = 0

    for key, base_val in baseline_map.items():
        if key not in current_map:
            skipped += 1
            continue

        curr_val = current_map[key]
        if base_val == 0:
            skipped += 1
            continue

        # For runtime_ms: higher is worse (regression)
        # For achieved_tflops: lower is worse (regression)
        if metric in ("runtime_ms",):
            change = (curr_val - base_val) / base_val
        else:
            change = (base_val - curr_val) / base_val

        if change > max_regression:
            regressions.append((*key, base_val, curr_val, change))
        else:
            passed += 1

    click.echo(f"CI Regression Check — metric: {metric}, threshold: {max_regression:.0%}")
    click.echo(f"Passed: {passed}, Regressions: {len(regressions)}, Skipped: {skipped}\n")

    if regressions:
        click.echo("REGRESSIONS DETECTED:")
        for variant_id, shape_id, base_val, curr_val, change in regressions:
            click.echo(
                f"  {variant_id} | {shape_id}: "
                f"{base_val:.3f} -> {curr_val:.3f} ({change:+.1%})"
            )
        sys.exit(1)
    else:
        click.echo("All benchmarks within regression threshold.")
