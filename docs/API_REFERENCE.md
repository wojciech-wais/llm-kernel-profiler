# API Reference

## Core Data Model (`llm_kernel_lab.model`)

### `KernelVariant`
Represents a specific kernel implementation.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `op_type` | `str` | Operation type (attention, mlp, matmul, layernorm, quant_gemm) |
| `backend` | `str` | Implementation backend (triton, cuda, cublas, cudnn, cutlass) |
| `source` | `str` | Module path or binary path |
| `config` | `dict` | Kernel parameters (block sizes, num_warps, etc.) |
| `metadata` | `dict` | Free-form metadata (git commit, framework, etc.) |

### `ProblemShape`
Represents a workload configuration.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `op_type` | `str` | Operation type (must match KernelVariant.op_type) |
| `params` | `dict` | Shape parameters (e.g., B, H, S_q, S_kv, D) |

### `HardwareProfile`
GPU hardware description.

| Field | Type | Description |
|-------|------|-------------|
| `gpu_name` | `str` | GPU name |
| `sm_version` | `int` | SM architecture version |
| `num_sms` | `int` | Number of SMs |
| `memory_bandwidth_gbps` | `float` | Peak memory bandwidth |
| `peak_fp16_tflops` | `float` | Peak FP16 throughput |
| `peak_fp32_tflops` | `float` | Peak FP32 throughput |

### `MetricSet`
Collected performance metrics.

Key fields: `runtime_ms`, `achieved_tflops`, `arithmetic_intensity`, `dram_bw_gbps`, `sm_efficiency`, `occupancy`, `stall_memory_dependency`, `stall_execution_dependency`.

### `KernelRun`
A single profiled kernel execution combining variant, shape, hardware, and metrics.

## Benchmark API (`llm_kernel_lab.bench`)

### `BenchmarkRunner`

```python
runner = BenchmarkRunner(backend="triton", device="cuda:0")
results = runner.run(
    kernels=[...],
    shapes=[...],
    repetitions=100,
    profile_level="timing",
    kernel_fns={"kernel_id": callable},
)
```

## Profiler API (`llm_kernel_lab.profiler`)

### `Profiler`

```python
profiler = Profiler(profile_level="full")
run = profiler.profile_kernel(kernel_fn=fn, args=(q, k, v), flops=1e12)
```

### `TimingProfiler`

Low-level timing using CUDA events.

```python
timer = TimingProfiler(warmup=10, repetitions=100)
metrics = timer.time_kernel(fn, flops=1e12, bytes_accessed=1e9)
```

## Report API (`llm_kernel_lab.report`)

### `generate_roofline_report`

```python
generate_roofline_report(runs=results, hardware=hw, outfile="report.md")
```

### `classify_bottleneck`

```python
from llm_kernel_lab.report.bottleneck import classify_bottleneck
bottleneck = classify_bottleneck(metrics, hardware)
```

### `get_recommendations`

```python
from llm_kernel_lab.report.recommendations import get_recommendations
recs = get_recommendations(metrics, hardware)
```

## Storage (`llm_kernel_lab.storage`)

### `JsonStore`

```python
store = JsonStore("./results")
store.save(runs, hardware, name="my_results")
hw, runs = store.load("my_results")
```

### `SqliteStore`

```python
store = SqliteStore("./results/db.sqlite")
store.save(runs, hardware)
all_runs = store.load_all()
```
