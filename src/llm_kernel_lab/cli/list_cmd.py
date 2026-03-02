"""List CLI command — list available kernels and shapes."""

from __future__ import annotations

import click

# Built-in reference shapes for popular LLM models
BUILTIN_SHAPES = {
    "attention": {
        "gpt3_175b_short": {"B": 1, "H": 96, "S_q": 512, "S_kv": 512, "D": 128},
        "gpt3_175b_long": {"B": 1, "H": 96, "S_q": 4096, "S_kv": 4096, "D": 128},
        "llama2_7b": {"B": 8, "H": 32, "S_q": 2048, "S_kv": 2048, "D": 128},
        "llama2_70b": {"B": 1, "H": 64, "S_q": 4096, "S_kv": 4096, "D": 128},
        "mistral_7b": {"B": 8, "H": 32, "S_q": 4096, "S_kv": 4096, "D": 128},
    },
    "matmul": {
        "small": {"M": 1024, "N": 1024, "K": 1024},
        "medium": {"M": 4096, "N": 4096, "K": 4096},
        "large": {"M": 8192, "N": 8192, "K": 8192},
        "llm_proj_7b": {"M": 4096, "N": 11008, "K": 4096},
    },
    "layernorm": {
        "llama2_7b": {"B": 8192, "D": 4096},
        "llama2_70b": {"B": 4096, "D": 8192},
    },
}


@click.command("list")
@click.argument("item_type", type=click.Choice(["kernels", "shapes"]))
@click.option("--op", default=None, help="Filter by op type")
def list_items(item_type: str, op: str | None) -> None:
    """List available kernels or built-in problem shapes."""
    if item_type == "shapes":
        _list_shapes(op)
    else:
        _list_kernels(op)


def _list_shapes(op: str | None) -> None:
    click.echo("Built-in problem shapes:\n")
    for op_type, shapes in BUILTIN_SHAPES.items():
        if op and op != op_type:
            continue
        click.echo(f"  [{op_type}]")
        for name, params in shapes.items():
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            click.echo(f"    {name}: {param_str}")
        click.echo()


def _list_kernels(op: str | None) -> None:
    click.echo("Kernel discovery requires a configured project.")
    click.echo("Use 'llm-kernel-lab init' to create a project configuration.")
    click.echo("\nSupported op types: attention, mlp, matmul, layernorm, quant_gemm, custom")
    click.echo("Supported backends: triton, cuda, cublas, cudnn, cutlass, pytorch")
