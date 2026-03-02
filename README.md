# LLM Kernel Lab

Profiling and optimization toolkit for GPU kernels used in LLM inference.

LLM Kernel Lab provides a batteries-included workflow for **benchmarking, profiling, and comparing** GPU kernels commonly found in Large Language Model inference stacks — attention (FlashAttention variants), RMSNorm, fused MLPs, quantized matmuls (AWQ, GPTQ, Marlin), and more.

## Why LLM Kernel Lab?

Existing tools like Nsight Compute and Nsight Systems are powerful but not tailored to LLM workloads. LLM Kernel Lab fills the gap:

- **LLM-specific benchmark harness** — predefined shapes for GPT-3, LLaMA, Mistral, and custom models with batch/sequence/head dimension grids.
- **Side-by-side kernel comparison** — compare Triton vs handwritten CUDA vs cuBLAS implementations in a reproducible way.
- **Roofline analysis** — automatic roofline plots showing whether kernels are memory-bound, compute-bound, or latency-bound.
- **Actionable recommendations** — rule-based suggestions for optimization (tile sizes, shared memory usage, occupancy tuning).
- **CI integration** — detect performance regressions in automated pipelines.

## Supported Backends & GPUs

| Backend | Status |
|---------|--------|
| Triton  | Supported |
| Raw CUDA (C++) | Supported |
| PyTorch (torch.profiler) | Supported |
| cuBLAS / cuDNN / CUTLASS | Via CUDA backend |

**Target GPUs**: NVIDIA SM 80+ (A100, H100, L40, RTX 3090/4090)
**Platform**: Linux x86_64

## Installation

```bash
pip install llm-kernel-lab
```

With GPU backends:

```bash
pip install "llm-kernel-lab[all]"   # torch + triton + plotly
pip install "llm-kernel-lab[torch]" # torch only
pip install "llm-kernel-lab[dev]"   # development dependencies
```

For the C++ CUDA harness:

```bash
cd csrc && mkdir build && cd build
cmake .. && make -j$(nproc)
```

## Quickstart

### 1. Record your GPU hardware profile

```bash
llm-kernel-lab record-hw
```

### 2. Run a benchmark

```bash
llm-kernel-lab bench \
  --op attention \
  --kernels flash_attn_v2_triton,sdpa_pytorch \
  --shapes llama2_7b_short,llama2_7b_long \
  --repetitions 100 \
  --profile-level timing \
  --output ./results/attn.json
```

### 3. Generate a report

```bash
llm-kernel-lab report \
  --input ./results/attn.json \
  --format markdown \
  --out ./reports/attn_report.md
```

### 4. Compare across GPUs or kernels

```bash
llm-kernel-lab compare \
  --inputs ./results/attn_a100.json,./results/attn_h100.json \
  --metric runtime_ms \
  --group-by gpu
```

### 5. CI regression detection

```bash
llm-kernel-lab ci \
  --baseline ./baseline/attn.json \
  --current ./results/attn.json \
  --max-regression 0.10
```

## Python API

```python
from llm_kernel_lab.bench import BenchmarkRunner
from llm_kernel_lab.model import KernelVariant, ProblemShape

runner = BenchmarkRunner(backend="triton", device="cuda:0")

kernels = [
    KernelVariant(
        id="flash_attn_v2_triton",
        op_type="attention",
        backend="triton",
        source="project.kernels.flash_attn_v2",
        config={"BLOCK_M": 128, "BLOCK_N": 128},
    ),
]

shapes = [
    ProblemShape(
        id="llama2_7b",
        op_type="attention",
        params={"B": 8, "H": 32, "S_q": 4096, "S_kv": 4096, "D": 128},
    ),
]

results = runner.run(kernels=kernels, shapes=shapes, repetitions=100)
for run in results:
    print(f"{run.variant.id}: {run.metrics.runtime_ms:.2f} ms, {run.metrics.achieved_tflops:.1f} TFLOP/s")
```

### Roofline report generation

```python
from llm_kernel_lab.report import generate_roofline_report

generate_roofline_report(
    runs=results,
    hardware=runner.hardware_profile,
    outfile="./reports/roofline.md",
)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `bench` | Run benchmarks for kernel variants across problem shapes |
| `report` | Generate Markdown/HTML/JSON reports from results |
| `compare` | Compare performance across result files or GPUs |
| `ci` | Regression detection for CI pipelines |
| `record-hw` | Capture GPU hardware profile |
| `list` | List available kernels or built-in shapes |
| `init` | Generate a starter configuration file |

## Profiling Levels

| Level | Metrics | Overhead |
|-------|---------|----------|
| `none` | CUDA event timing only | < 5% |
| `timing` | Runtime, FLOP/s, bandwidth | < 5% |
| `basic` | + SM efficiency, DRAM BW, occupancy | < 20% |
| `full` | + stall breakdown, L2 stats, warp efficiency (via Nsight Compute) | < 3x |

## Report Output

Reports include:

- **Summary table** — runtime, TFLOP/s, bandwidth, SM efficiency, occupancy per kernel/shape
- **Roofline plot** — arithmetic intensity vs achieved performance with memory/compute roofs
- **Bottleneck classification** — memory-bound, compute-bound, latency-bound, or balanced
- **Optimization recommendations** — actionable suggestions based on collected metrics

## Configuration

Create `llm-kernel-lab.toml` in your project root (or run `llm-kernel-lab init`):

```toml
[gpu]
peak_fp16_tflops = 312.0
memory_bandwidth_gbps = 1555.0

[nsight]
path = "/usr/local/cuda/bin/ncu"

[bench]
default_repetitions = 50
warmup = 10
output_dir = "./results"
```

## Project Structure

```
llm-kernel-lab/
├── src/llm_kernel_lab/
│   ├── model.py           # Core data model (KernelVariant, ProblemShape, MetricSet, etc.)
│   ├── config.py           # TOML configuration loading
│   ├── hardware.py         # GPU hardware detection
│   ├── serialization.py    # JSON serialization
│   ├── bench/              # Benchmark orchestrator and backend harnesses
│   ├── profiler/           # Timing, Nsight Compute, and metric collection
│   ├── report/             # Roofline, bottleneck, recommendations, rendering
│   ├── cli/                # Click-based CLI commands
│   └── storage/            # JSON and SQLite storage backends
├── csrc/                   # C++/CUDA benchmark harness with pybind11 bindings
├── tests/                  # pytest unit and integration tests
├── examples/               # Example scripts
└── .github/workflows/      # CI configuration
```

## Security

This tool executes user-provided kernel code. **Never run untrusted kernels.** The profiler invokes `ncu` and CUDA APIs — ensure your environment is secure.

## License

Apache-2.0
