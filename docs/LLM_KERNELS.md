# LLM Kernels Guide

## Common LLM GPU Kernels

LLM inference relies on several specialized GPU kernel types. This guide describes each and how to plug them into LLM Kernel Lab.

### Attention Kernels

The attention mechanism is the core of transformer-based LLMs. Common variants:

- **FlashAttention v2/v3** — Tiled, IO-aware attention (Triton and CUDA)
- **Scaled Dot-Product Attention (SDPA)** — PyTorch built-in via `torch.nn.functional.scaled_dot_product_attention`
- **PagedAttention** — Used in vLLM for KV cache management

**Typical shapes:**

| Model | B | H | S_q | S_kv | D |
|-------|---|---|-----|------|---|
| LLaMA-2 7B | 8 | 32 | 2048 | 2048 | 128 |
| LLaMA-2 70B | 1 | 64 | 4096 | 4096 | 128 |
| GPT-3 175B | 1 | 96 | 2048 | 2048 | 128 |
| Mistral 7B | 8 | 32 | 4096 | 4096 | 128 |

**FLOPs:** ~4 * B * H * S_q * S_kv * D (QK^T + softmax + attn@V)

### Matrix Multiplication (GEMM)

Used in linear projections (QKV, output, MLP layers).

- **cuBLAS GEMM** — Vendor-optimized
- **CUTLASS** — Template-based CUDA kernels
- **Triton matmul** — Autotunable

### Quantized GEMM

For weight-quantized models (INT4, INT8):

- **Marlin** — INT4 GEMM optimized for NVIDIA GPUs
- **AWQ kernels** — Activation-aware weight quantization
- **GPTQ kernels** — Post-training quantization

### LayerNorm / RMSNorm

Normalization applied at every transformer layer.

- **Triton RMSNorm** — Fused implementation
- **CUDA fused LayerNorm** — Custom implementations

### Fused MLP

Fused gate + up projection + activation + down projection:

- SwiGLU activation (LLaMA, Mistral)
- GeGLU activation

## Plugging in Custom Kernels

### Via Python API

```python
from llm_kernel_lab.model import KernelVariant

my_kernel = KernelVariant(
    id="my_custom_attention",
    op_type="attention",
    backend="triton",
    source="my_project.kernels.custom_attn.forward",
    config={"BLOCK_M": 64, "BLOCK_N": 64},
    metadata={"author": "me", "commit": "abc123"},
)
```

### Via CLI

Register kernels by their Python import path:

```bash
llm-kernel-lab bench \
  --op attention \
  --kernels my_project.kernels.custom_attn.forward \
  --shapes llama2_7b \
  --profile-level full
```

### Via CUDA Backend

For raw CUDA kernels, build against the C++ harness:

```cpp
#include "llm_kernel_lab/benchmark.h"

auto result = llm_kernel_lab::run_kernel_benchmark(
    [](cudaStream_t stream, const LaunchConfig& config) {
        my_kernel<<<config.grid, config.block, config.shared_mem_bytes, stream>>>(args...);
    },
    /*warmup=*/10,
    /*iters=*/100,
    ProfileLevel::Timing
);
```
