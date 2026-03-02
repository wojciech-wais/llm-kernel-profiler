"""Record-hw CLI command — capture hardware profile."""

from __future__ import annotations

import click


@click.command("record-hw")
@click.option("--device", default=0, type=int, help="GPU device index")
@click.option("--output", "-o", default=None, help="Output JSON file (prints to stdout if omitted)")
def record_hw(device: int, output: str | None) -> None:
    """Capture hardware profile of the current machine's GPU."""

    from llm_kernel_lab.hardware import detect_hardware, hardware_profile_to_json

    click.echo(f"Detecting GPU hardware (device {device})...")
    hw = detect_hardware(device)

    info = hardware_profile_to_json(hw)

    if output:
        from pathlib import Path

        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(info)
        click.echo(f"Hardware profile saved to {output}")
    else:
        click.echo(info)
