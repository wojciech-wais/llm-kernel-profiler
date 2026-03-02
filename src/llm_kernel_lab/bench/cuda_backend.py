"""Raw CUDA backend harness (C++ library bridge)."""

from __future__ import annotations

from typing import Any


class CudaBackend:
    """Harness for running raw CUDA kernels via the C++ library.

    Requires the compiled libllm_kernel_lab_cuda library with pybind11 bindings.
    """

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._lib = None
        self._check_library()

    def _check_library(self) -> None:
        try:
            import llm_kernel_lab_cuda as lib  # type: ignore[import-not-found]

            self._lib = lib
        except ImportError:
            self._lib = None

    @property
    def available(self) -> bool:
        return self._lib is not None

    def run_benchmark(
        self,
        kernel_launcher: Any,
        warmup_iters: int = 10,
        iters: int = 100,
        profile_level: str = "timing",
    ) -> dict[str, Any]:
        """Run a CUDA kernel benchmark through the C++ harness."""
        if not self.available:
            raise RuntimeError(
                "CUDA backend library not available. "
                "Build the C++ harness with CMake first."
            )
        return self._lib.run_kernel_benchmark(
            kernel_launcher, warmup_iters, iters, profile_level
        )
