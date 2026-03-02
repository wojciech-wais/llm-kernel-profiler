"""Triton backend harness for benchmarking Triton kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TritonBackend:
    """Harness for running and profiling Triton kernels."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._check_triton()

    def _check_triton(self) -> None:
        try:
            import triton  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def create_launcher(
        self,
        kernel_fn: Any,
        grid: tuple[int, ...] | Callable,
        config: dict[str, Any] | None = None,
    ) -> Callable:
        """Create a launcher function for a Triton kernel.

        Args:
            kernel_fn: A @triton.jit decorated function.
            grid: Grid dimensions or a callable that returns grid dims.
            config: Kernel config overrides (BLOCK_M, num_warps, etc.)

        Returns:
            A callable that launches the kernel with given inputs.
        """
        def launcher(*args: Any, **kwargs: Any) -> None:
            merged_kwargs = {**(config or {}), **kwargs}
            kernel_fn[grid](*args, **merged_kwargs)

        return launcher

    def prepare_tensors(
        self,
        shapes: dict[str, tuple[int, ...]],
        dtype: str = "float16",
    ) -> dict[str, Any]:
        """Allocate tensors on the target device."""
        import torch

        torch_dtype = getattr(torch, dtype, torch.float16)
        tensors = {}
        for name, shape in shapes.items():
            tensors[name] = torch.randn(shape, dtype=torch_dtype, device=self.device)
        return tensors
