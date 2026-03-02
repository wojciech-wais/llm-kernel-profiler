# Getting Started with LLM Kernel Lab

## Prerequisites

- Python 3.10+
- NVIDIA GPU with SM 80+ (A100, H100, L40, RTX 30xx/40xx)
- CUDA Toolkit 11.8+ (for GPU profiling)
- Linux x86_64

## Installation

### Basic install (CPU-only features)

```bash
pip install llm-kernel-lab
```

### Full install (GPU + visualization)

```bash
pip install "llm-kernel-lab[all]"
```

### Development install

```bash
git clone https://github.com/llm-kernel-lab/llm-kernel-lab.git
cd llm-kernel-lab
pip install -e ".[dev,all]"
```

### C++ CUDA harness (optional)

```bash
cd csrc
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## First Benchmark

### Step 1: Initialize a project

```bash
llm-kernel-lab init
```

This creates an `llm-kernel-lab.toml` configuration file.

### Step 2: Check your GPU

```bash
llm-kernel-lab record-hw
```

### Step 3: List available shapes

```bash
llm-kernel-lab list shapes
```

### Step 4: Run a timing benchmark

```bash
llm-kernel-lab bench \
  --op matmul \
  --kernels my_matmul_kernel \
  --shapes small,medium \
  --repetitions 50 \
  --profile-level timing \
  --output ./results/matmul.json
```

### Step 5: Generate a report

```bash
llm-kernel-lab report \
  --input ./results/matmul.json \
  --format markdown \
  --out ./reports/matmul_report.md
```

## Interpreting the Report

### Summary Table

The summary table shows key metrics for each kernel/shape combination:

- **Runtime (ms)**: Average execution time
- **TFLOP/s**: Achieved throughput (higher is better)
- **DRAM BW (GB/s)**: Memory bandwidth utilization
- **SM Eff.**: Streaming multiprocessor efficiency (0-1)
- **Occupancy**: Warp occupancy (0-1)
- **Bottleneck**: Automatic classification

### Roofline Plot

The roofline plot shows where each kernel sits relative to hardware limits:

- Points below the diagonal line are **memory-bound**
- Points near the horizontal line are **compute-bound**
- The ridge point is where memory and compute roofs intersect

### Bottleneck Classification

- **memory-bound**: Kernel is limited by memory bandwidth (DRAM BW > 80% peak, SM efficiency < 70%)
- **compute-bound**: Kernel is using compute resources near peak
- **latency-bound**: Low SM efficiency with high execution dependency stalls
- **balanced**: No single dominant bottleneck
