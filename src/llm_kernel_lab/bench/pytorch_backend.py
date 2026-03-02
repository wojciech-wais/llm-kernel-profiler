"""PyTorch backend harness using torch.profiler."""

from __future__ import annotations

from typing import Any, Callable


class PytorchBackend:
    """Harness for profiling PyTorch modules and functions."""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device
        self._check_torch()

    def _check_torch(self) -> None:
        try:
            import torch  # noqa: F401
            self._available = True
        except ImportError:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def profile_function(
        self,
        fn: Callable[..., Any],
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        warmup: int = 5,
    ) -> list[dict[str, Any]]:
        """Profile a PyTorch function and return kernel-level info.

        Returns a list of dicts with kernel name, duration, etc.
        """
        import torch
        from torch.profiler import ProfilerActivity, profile

        kwargs = kwargs or {}

        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        kernel_events: list[dict[str, Any]] = []

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            fn(*args, **kwargs)
            torch.cuda.synchronize()

        for event in prof.key_averages():
            if event.device_type is not None and "cuda" in str(event.device_type).lower():
                kernel_events.append({
                    "name": event.key,
                    "cuda_time_ms": event.cuda_time_total / 1000.0,
                    "cpu_time_ms": event.cpu_time_total / 1000.0,
                    "calls": event.count,
                })

        return kernel_events

    def wrap_module(
        self,
        module: Any,
        input_shapes: dict[str, tuple[int, ...]],
        dtype: str = "float16",
    ) -> Callable:
        """Wrap a PyTorch module into a callable for benchmarking."""
        import torch

        torch_dtype = getattr(torch, dtype, torch.float16)
        inputs = {
            name: torch.randn(shape, dtype=torch_dtype, device=self.device)
            for name, shape in input_shapes.items()
        }

        def runner() -> Any:
            return module(**inputs)

        return runner
