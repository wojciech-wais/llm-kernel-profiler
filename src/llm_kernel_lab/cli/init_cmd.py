"""Init CLI command — generate starter configuration."""

from __future__ import annotations

from pathlib import Path

import click

_STARTER_CONFIG = """\
[gpu]
# Optional overrides; otherwise auto-detected
# peak_fp16_tflops = 312.0
# peak_fp32_tflops = 78.0
# memory_bandwidth_gbps = 1555.0

[nsight]
path = "/usr/local/cuda/bin/ncu"
metrics = [
  "sm__throughput.avg.pct_of_peak_sustained_elapsed",
  "smsp__warps_active.avg.pct_of_peak_sustained_active",
  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
  "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
  "l2tex__t_bytes.sum",
]

[bench]
default_repetitions = 50
warmup = 10
output_dir = "./results"
"""


@click.command()
@click.option("--dir", "directory", default=".", help="Directory to create config in")
def init(directory: str) -> None:
    """Generate a starter llm-kernel-lab.toml configuration file."""
    config_path = Path(directory) / "llm-kernel-lab.toml"

    if config_path.exists():
        click.echo(f"Configuration already exists at {config_path}")
        if not click.confirm("Overwrite?"):
            return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_STARTER_CONFIG)
    click.echo(f"Created configuration at {config_path}")
    click.echo("Edit the file to customize GPU specs, Nsight path, and benchmark settings.")
