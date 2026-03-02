"""FLOP and bandwidth estimation utilities for common LLM operations."""

from __future__ import annotations

from typing import Any

# FLOP estimation functions for common LLM operations


def estimate_attention_flops(
    batch_size: int,
    num_heads: int,
    seq_len_q: int,
    seq_len_kv: int,
    head_dim: int,
) -> float:
    """Estimate FLOPs for attention: Q @ K^T + softmax + attn @ V.

    Approximate: 2 * B * H * S_q * S_kv * D (for QK^T) + 2 * B * H * S_q * S_kv * D (for attn@V)
    """
    qk_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim
    av_flops = 2 * batch_size * num_heads * seq_len_q * seq_len_kv * head_dim
    # Softmax is relatively small
    softmax_flops = 3 * batch_size * num_heads * seq_len_q * seq_len_kv
    return float(qk_flops + av_flops + softmax_flops)


def estimate_matmul_flops(m: int, n: int, k: int) -> float:
    """Estimate FLOPs for matrix multiplication: C[m,n] = A[m,k] @ B[k,n]."""
    return float(2 * m * n * k)


def estimate_layernorm_flops(batch_size: int, hidden_dim: int) -> float:
    """Estimate FLOPs for LayerNorm/RMSNorm."""
    # mean, variance, normalize, scale+bias
    return float(5 * batch_size * hidden_dim)


def estimate_mlp_flops(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    gated: bool = True,
) -> float:
    """Estimate FLOPs for MLP (gate_proj + up_proj + activation + down_proj)."""
    tokens = batch_size * seq_len
    if gated:
        # gate_proj: tokens x hidden -> tokens x intermediate
        # up_proj: same
        # element-wise multiply + activation
        # down_proj: tokens x intermediate -> tokens x hidden
        return float(
            2 * tokens * hidden_dim * intermediate_dim  # gate
            + 2 * tokens * hidden_dim * intermediate_dim  # up
            + tokens * intermediate_dim  # activation
            + 2 * tokens * intermediate_dim * hidden_dim  # down
        )
    return float(
        2 * tokens * hidden_dim * intermediate_dim
        + tokens * intermediate_dim
        + 2 * tokens * intermediate_dim * hidden_dim
    )


def estimate_bytes_accessed(
    params: dict[str, Any],
    op_type: str,
    dtype_bytes: int = 2,
) -> float:
    """Estimate bytes accessed for a given operation and shape."""
    if op_type == "attention":
        B = params.get("B", 1)
        H = params.get("H", 1)
        S_q = params.get("S_q", params.get("S", 1))
        D = params.get("D", 1)
        # Q, K, V reads + output write (K/V use S_kv but same dim for byte estimate)
        qkv_bytes = 3 * B * H * S_q * D * dtype_bytes
        out_bytes = B * H * S_q * D * dtype_bytes
        return float(qkv_bytes + out_bytes)

    if op_type == "matmul":
        M = params.get("M", 1)
        N = params.get("N", 1)
        K = params.get("K", 1)
        return float((M * K + K * N + M * N) * dtype_bytes)

    if op_type == "layernorm":
        B = params.get("B", 1)
        D = params.get("D", 1)
        return float(2 * B * D * dtype_bytes)  # read + write

    if op_type == "mlp":
        B = params.get("B", 1)
        S = params.get("S", 1)
        hidden = params.get("hidden_dim", 1)
        inter = params.get("intermediate_dim", 1)
        # weights + activations
        return float((hidden * inter * 3 + B * S * (hidden + inter * 2 + hidden)) * dtype_bytes)

    return 0.0


def estimate_flops(params: dict[str, Any], op_type: str) -> float:
    """Estimate FLOPs for a given operation type and parameters."""
    if op_type == "attention":
        return estimate_attention_flops(
            batch_size=params.get("B", 1),
            num_heads=params.get("H", 1),
            seq_len_q=params.get("S_q", params.get("S", 1)),
            seq_len_kv=params.get("S_kv", params.get("S", 1)),
            head_dim=params.get("D", 1),
        )
    if op_type == "matmul":
        return estimate_matmul_flops(
            m=params.get("M", 1),
            n=params.get("N", 1),
            k=params.get("K", 1),
        )
    if op_type == "layernorm":
        return estimate_layernorm_flops(
            batch_size=params.get("B", 1),
            hidden_dim=params.get("D", 1),
        )
    if op_type == "mlp":
        return estimate_mlp_flops(
            batch_size=params.get("B", 1),
            seq_len=params.get("S", 1),
            hidden_dim=params.get("hidden_dim", 1),
            intermediate_dim=params.get("intermediate_dim", 1),
            gated=params.get("gated", True),
        )
    return 0.0
