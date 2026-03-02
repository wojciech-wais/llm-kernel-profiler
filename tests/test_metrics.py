"""Tests for FLOP and bandwidth estimation utilities."""

from llm_kernel_lab.profiler.metrics import (
    estimate_attention_flops,
    estimate_bytes_accessed,
    estimate_flops,
    estimate_layernorm_flops,
    estimate_matmul_flops,
    estimate_mlp_flops,
)


def test_attention_flops():
    flops = estimate_attention_flops(batch_size=1, num_heads=32, seq_len_q=2048, seq_len_kv=2048, head_dim=128)
    # 2 * 1 * 32 * 2048 * 2048 * 128 * 2 + softmax
    assert flops > 0
    # QK^T + attn@V ~ 2 * 2 * B*H*Sq*Skv*D
    expected_approx = 2 * 2 * 1 * 32 * 2048 * 2048 * 128
    assert abs(flops - expected_approx) / expected_approx < 0.01


def test_matmul_flops():
    flops = estimate_matmul_flops(m=4096, n=4096, k=4096)
    assert flops == 2 * 4096 * 4096 * 4096


def test_layernorm_flops():
    flops = estimate_layernorm_flops(batch_size=1024, hidden_dim=4096)
    assert flops == 5 * 1024 * 4096


def test_mlp_flops_gated():
    flops = estimate_mlp_flops(
        batch_size=1, seq_len=2048, hidden_dim=4096,
        intermediate_dim=11008, gated=True,
    )
    assert flops > 0


def test_estimate_flops_dispatch():
    assert estimate_flops({"M": 1024, "N": 1024, "K": 1024}, "matmul") == 2 * 1024**3
    assert estimate_flops({"B": 1, "D": 4096}, "layernorm") == 5 * 1 * 4096
    assert estimate_flops({}, "unknown_op") == 0.0


def test_estimate_bytes_accessed_attention():
    params = {"B": 1, "H": 32, "S_q": 2048, "S_kv": 2048, "D": 128}
    b = estimate_bytes_accessed(params, "attention", dtype_bytes=2)
    # 3 * B*H*S*D * 2 (QKV) + B*H*S*D * 2 (output)
    expected = (3 * 1 * 32 * 2048 * 128 + 1 * 32 * 2048 * 128) * 2
    assert b == expected


def test_estimate_bytes_accessed_matmul():
    params = {"M": 1024, "N": 1024, "K": 1024}
    b = estimate_bytes_accessed(params, "matmul", dtype_bytes=2)
    expected = (1024 * 1024 + 1024 * 1024 + 1024 * 1024) * 2
    assert b == expected
