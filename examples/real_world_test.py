"""Real-world end-to-end test of LLM Kernel Lab.

Benchmarks actual GPU kernels on your hardware:
1. Triton matmul vs PyTorch matmul
2. PyTorch Scaled Dot-Product Attention at various sequence lengths
3. Fused RMSNorm (Triton) vs PyTorch RMSNorm
4. Generates roofline report and compares results

Requires: torch, triton, NVIDIA GPU
"""

import time
import uuid
from pathlib import Path

import torch
import triton
import triton.language as tl

from llm_kernel_lab.bench.runner import BenchmarkRunner
from llm_kernel_lab.hardware import detect_hardware
from llm_kernel_lab.model import HardwareProfile, KernelRun, KernelVariant, MetricSet, ProblemShape
from llm_kernel_lab.profiler.metrics import estimate_attention_flops, estimate_matmul_flops, estimate_layernorm_flops
from llm_kernel_lab.profiler.metrics import estimate_bytes_accessed
from llm_kernel_lab.profiler.timing import TimingProfiler
from llm_kernel_lab.report.bottleneck import classify_bottleneck
from llm_kernel_lab.report.recommendations import get_recommendations
from llm_kernel_lab.report.renderer import render_markdown_report, render_json_report
from llm_kernel_lab.report.roofline import compute_roofline_data, plot_roofline
from llm_kernel_lab.serialization import save_results


# ---------------------------------------------------------------------------
# 1. Triton matmul kernel (from Triton tutorials, simplified)
# ---------------------------------------------------------------------------
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = ((M + BLOCK_M - 1) // BLOCK_M, (N + BLOCK_N - 1) // BLOCK_N)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# ---------------------------------------------------------------------------
# 2. Triton RMSNorm kernel
# ---------------------------------------------------------------------------
@triton.jit
def rmsnorm_kernel(
    x_ptr, weight_ptr, out_ptr,
    N, eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (x * rrms * w).to(tl.float16)
    tl.store(out_ptr + row * N + offs, out, mask=mask)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_kernel[(B,)](x, weight, out, N, eps=eps, BLOCK_N=BLOCK_N)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bench_kernel(name: str, fn, flops: float, bytes_acc: float, hw: HardwareProfile,
                 variant: KernelVariant, shape: ProblemShape,
                 warmup: int = 20, reps: int = 200) -> KernelRun:
    """Benchmark a single kernel function and return a KernelRun."""
    timer = TimingProfiler(warmup=warmup, repetitions=reps)
    metrics = timer.time_kernel(fn, flops=flops, bytes_accessed=bytes_acc)
    return KernelRun(
        id=str(uuid.uuid4()),
        variant=variant,
        shape=shape,
        hardware=hw,
        metrics=metrics,
        raw_profiler_output_path="",
        timestamp=time.time(),
    )


def print_run(run: KernelRun) -> None:
    m = run.metrics
    bottleneck = classify_bottleneck(m, run.hardware)
    print(f"  {run.variant.id:30s} | {run.shape.id:20s} | "
          f"{m.runtime_ms:8.3f} ms | {m.achieved_tflops:7.1f} TFLOP/s | "
          f"{m.dram_bw_gbps:8.1f} GB/s | {bottleneck}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda:0"
    print("=" * 90)
    print("LLM Kernel Lab — Real-World End-to-End Test")
    print("=" * 90)

    # Detect hardware
    print("\n[1/5] Detecting hardware...")
    hw = detect_hardware(0)
    # Fill in known RTX 3060 specs if auto-detect missed them
    if hw.peak_fp16_tflops == 0:
        hw = HardwareProfile(
            gpu_name=hw.gpu_name or "NVIDIA GeForce RTX 3060",
            sm_version=86,
            num_sms=28,
            memory_bandwidth_gbps=360.0,
            peak_fp16_tflops=12.74,
            peak_fp32_tflops=12.74,
            l2_size_mb=3.0,
            driver_version=hw.driver_version,
            cuda_version=hw.cuda_version,
        )
    print(f"  GPU: {hw.gpu_name}")
    print(f"  Peak FP16: {hw.peak_fp16_tflops} TFLOP/s")
    print(f"  Memory BW: {hw.memory_bandwidth_gbps} GB/s")
    print(f"  CUDA: {hw.cuda_version}, Driver: {hw.driver_version}")

    all_runs: list[KernelRun] = []

    # -------------------------------------------------------------------
    # Test 1: Matmul comparison (Triton vs PyTorch)
    # -------------------------------------------------------------------
    print("\n[2/5] Benchmarking matmul kernels (Triton vs PyTorch)...")
    print(f"  {'Kernel':30s} | {'Shape':20s} | {'Runtime':>10s} | {'TFLOP/s':>9s} | {'BW':>10s} | Bottleneck")
    print("  " + "-" * 95)

    for M, N, K, shape_name in [
        (1024, 1024, 1024, "1k_square"),
        (4096, 4096, 4096, "4k_square"),
        (4096, 11008, 4096, "llama7b_mlp"),
    ]:
        a = torch.randn(M, K, device=device, dtype=torch.float16)
        b = torch.randn(K, N, device=device, dtype=torch.float16)
        flops = estimate_matmul_flops(M, N, K)
        bytes_acc = (M * K + K * N + M * N) * 2  # fp16

        shape = ProblemShape(id=shape_name, op_type="matmul", params={"M": M, "N": N, "K": K})

        # Triton matmul
        variant_triton = KernelVariant(
            id="triton_matmul", op_type="matmul", backend="triton",
            source="examples.real_world_test.triton_matmul",
            config={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        )
        run = bench_kernel("triton_matmul", lambda: triton_matmul(a, b),
                           flops, bytes_acc, hw, variant_triton, shape)
        all_runs.append(run)
        print_run(run)

        # PyTorch matmul (uses cuBLAS under the hood)
        variant_torch = KernelVariant(
            id="torch_matmul", op_type="matmul", backend="pytorch",
            source="torch.matmul", config={},
        )
        run = bench_kernel("torch_matmul", lambda: torch.matmul(a, b),
                           flops, bytes_acc, hw, variant_torch, shape)
        all_runs.append(run)
        print_run(run)

    # -------------------------------------------------------------------
    # Test 2: Attention (PyTorch SDPA at different sequence lengths)
    # -------------------------------------------------------------------
    print("\n[3/5] Benchmarking attention (PyTorch SDPA)...")
    print(f"  {'Kernel':30s} | {'Shape':20s} | {'Runtime':>10s} | {'TFLOP/s':>9s} | {'BW':>10s} | Bottleneck")
    print("  " + "-" * 95)

    for B, H, S, D, shape_name in [
        (4, 32, 512, 128, "attn_512"),
        (4, 32, 1024, 128, "attn_1k"),
        (4, 32, 2048, 128, "attn_2k"),
        (2, 32, 4096, 128, "attn_4k"),
    ]:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        flops = estimate_attention_flops(B, H, S, S, D)
        bytes_acc = estimate_bytes_accessed(
            {"B": B, "H": H, "S_q": S, "S_kv": S, "D": D}, "attention", dtype_bytes=2
        )

        shape = ProblemShape(
            id=shape_name, op_type="attention",
            params={"B": B, "H": H, "S_q": S, "S_kv": S, "D": D},
        )
        variant = KernelVariant(
            id="torch_sdpa", op_type="attention", backend="pytorch",
            source="torch.nn.functional.scaled_dot_product_attention", config={},
        )
        run = bench_kernel(
            "torch_sdpa",
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            flops, bytes_acc, hw, variant, shape,
        )
        all_runs.append(run)
        print_run(run)

    # -------------------------------------------------------------------
    # Test 3: RMSNorm (Triton vs PyTorch)
    # -------------------------------------------------------------------
    print("\n[4/5] Benchmarking RMSNorm (Triton vs PyTorch)...")
    print(f"  {'Kernel':30s} | {'Shape':20s} | {'Runtime':>10s} | {'TFLOP/s':>9s} | {'BW':>10s} | Bottleneck")
    print("  " + "-" * 95)

    for tokens, dim, shape_name in [
        (2048, 4096, "norm_7b"),
        (2048, 8192, "norm_70b"),
    ]:
        x = torch.randn(tokens, dim, device=device, dtype=torch.float16)
        w = torch.randn(dim, device=device, dtype=torch.float16)
        flops = estimate_layernorm_flops(tokens, dim)
        bytes_acc = 2 * tokens * dim * 2  # read + write, fp16

        shape = ProblemShape(id=shape_name, op_type="layernorm", params={"B": tokens, "D": dim})

        # Triton RMSNorm
        variant_triton = KernelVariant(
            id="triton_rmsnorm", op_type="layernorm", backend="triton",
            source="examples.real_world_test.triton_rmsnorm", config={},
        )
        run = bench_kernel("triton_rmsnorm", lambda: triton_rmsnorm(x, w),
                           flops, bytes_acc, hw, variant_triton, shape)
        all_runs.append(run)
        print_run(run)

        # PyTorch RMSNorm equivalent
        def torch_rmsnorm():
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + 1e-6)) * w

        variant_torch = KernelVariant(
            id="torch_rmsnorm", op_type="layernorm", backend="pytorch",
            source="manual", config={},
        )
        run = bench_kernel("torch_rmsnorm", torch_rmsnorm,
                           flops, bytes_acc, hw, variant_torch, shape)
        all_runs.append(run)
        print_run(run)

    # -------------------------------------------------------------------
    # Generate reports
    # -------------------------------------------------------------------
    print("\n[5/5] Generating reports...")
    outdir = Path("./results")
    outdir.mkdir(exist_ok=True)
    reportdir = Path("./reports")
    reportdir.mkdir(exist_ok=True)

    # Save JSON results
    save_results(all_runs, hw, outdir / "real_world_test.json")
    print(f"  Saved {len(all_runs)} runs to {outdir / 'real_world_test.json'}")

    # Generate JSON report (with bottleneck + recommendations)
    render_json_report(all_runs, hw, reportdir / "real_world_report.json")
    print(f"  JSON report: {reportdir / 'real_world_report.json'}")

    # Roofline data
    points = compute_roofline_data(all_runs, hw)

    # Roofline plot
    try:
        plot_roofline(points, hw, reportdir / "roofline.png", title=f"Roofline — {hw.gpu_name}")
        print(f"  Roofline plot: {reportdir / 'roofline.png'}")
    except Exception as e:
        print(f"  Roofline plot failed: {e}")

    # Markdown report
    render_markdown_report(all_runs, hw, points, reportdir / "real_world_report.md",
                           plot_path="roofline.png")
    print(f"  Markdown report: {reportdir / 'real_world_report.md'}")

    # -------------------------------------------------------------------
    # Print recommendations for the slowest kernel
    # -------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("Recommendations (sample)")
    print("=" * 90)
    slowest = max(all_runs, key=lambda r: r.metrics.runtime_ms)
    recs = get_recommendations(slowest.metrics, slowest.hardware)
    print(f"\n  Slowest: {slowest.variant.id} on {slowest.shape.id} ({slowest.metrics.runtime_ms:.3f} ms)")
    for rec in recs:
        print(f"    → {rec}")

    print(f"\nDone! {len(all_runs)} kernel benchmarks completed.")
    print(f"Check ./reports/ for the full analysis.")


if __name__ == "__main__":
    main()
