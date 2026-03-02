"""JSON-based storage for benchmark results."""

from __future__ import annotations

from pathlib import Path

from llm_kernel_lab.model import HardwareProfile, KernelRun
from llm_kernel_lab.serialization import load_results, save_results


class JsonStore:
    """Manages reading and writing benchmark results as JSON files."""

    def __init__(self, base_dir: str | Path = "./results") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        runs: list[KernelRun],
        hardware: HardwareProfile,
        name: str = "results",
    ) -> Path:
        """Save results to a JSON file. Returns the path."""
        path = self.base_dir / f"{name}.json"
        save_results(runs, hardware, path)
        return path

    def load(self, name: str = "results") -> tuple[HardwareProfile, list[KernelRun]]:
        """Load results from a JSON file."""
        path = self.base_dir / f"{name}.json"
        return load_results(path)

    def load_from_path(self, path: str | Path) -> tuple[HardwareProfile, list[KernelRun]]:
        """Load results from an arbitrary path."""
        return load_results(path)

    def list_results(self) -> list[Path]:
        """List all JSON result files in the store."""
        return sorted(self.base_dir.glob("*.json"))
