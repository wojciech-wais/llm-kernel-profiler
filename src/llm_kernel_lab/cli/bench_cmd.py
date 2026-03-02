"""Bench CLI command — run benchmarks for kernels."""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--op", required=True, help="Op type: attention, mlp, matmul, layernorm, quant_gemm, custom")
@click.option("--kernels", required=True, help="Comma-separated kernel variant IDs")
@click.option("--shapes", required=True, help="Comma-separated problem shape IDs")
@click.option("--repetitions", default=100, help="Number of timed iterations per benchmark")
@click.option("--warmup", default=10, help="Number of warmup iterations")
@click.option(
    "--profile-level",
    type=click.Choice(["none", "timing", "basic", "full"]),
    default="timing",
    help="Profiling depth",
)
@click.option("--output", "-o", default="./results/results.json", help="Output path for results")
@click.option("--backend", type=click.Choice(["pytorch", "triton", "cuda"]), default="triton")
@click.option("--device", default="cuda:0", help="CUDA device")
def bench(
    op: str,
    kernels: str,
    shapes: str,
    repetitions: int,
    warmup: int,
    profile_level: str,
    output: str,
    backend: str,
    device: str,
) -> None:
    """Run benchmarks for kernel variants across problem shapes."""
    from llm_kernel_lab.bench.runner import BenchmarkRunner
    from llm_kernel_lab.config import load_config
    from llm_kernel_lab.model import KernelVariant, ProblemShape
    from llm_kernel_lab.serialization import save_results

    config = load_config()
    runner = BenchmarkRunner(backend=backend, device=device, config=config)

    kernel_ids = [k.strip() for k in kernels.split(",")]
    shape_ids = [s.strip() for s in shapes.split(",")]

    kernel_variants = [
        KernelVariant(id=kid, op_type=op, backend=backend, source=kid)
        for kid in kernel_ids
    ]
    problem_shapes = [
        ProblemShape(id=sid, op_type=op)
        for sid in shape_ids
    ]

    click.echo(f"Running benchmarks: {len(kernel_variants)} kernels x {len(problem_shapes)} shapes")
    click.echo(f"Profile level: {profile_level}, Repetitions: {repetitions}")

    results = runner.run(
        kernels=kernel_variants,
        shapes=problem_shapes,
        repetitions=repetitions,
        warmup=warmup,
        profile_level=profile_level,
    )

    hardware = runner._get_hardware()
    save_results(results, hardware, output)

    click.echo(f"\nResults saved to {output}")
    click.echo(f"Total runs: {len(results)}")

    for run in results:
        click.echo(
            f"  {run.variant.id} | {run.shape.id} | "
            f"{run.metrics.runtime_ms:.3f} ms | "
            f"{run.metrics.achieved_tflops:.1f} TFLOP/s"
        )
