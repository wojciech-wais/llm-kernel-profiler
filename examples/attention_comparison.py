"""Example: Compare attention kernel implementations.

This example demonstrates comparing multiple attention kernel variants
across different problem shapes using llm-kernel-lab.

Requires: torch, triton (for GPU execution)
"""

from llm_kernel_lab.model import KernelVariant, ProblemShape


def define_kernels() -> list[KernelVariant]:
    """Define attention kernel variants to compare."""
    return [
        KernelVariant(
            id="flash_attn_v2_triton",
            op_type="attention",
            backend="triton",
            source="flash_attn.flash_attn_triton.flash_attn_func",
            config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            metadata={"version": "2.0", "framework": "triton"},
        ),
        KernelVariant(
            id="sdpa_pytorch",
            op_type="attention",
            backend="pytorch",
            source="torch.nn.functional.scaled_dot_product_attention",
            config={},
            metadata={"version": "torch_builtin", "framework": "pytorch"},
        ),
    ]


def define_shapes() -> list[ProblemShape]:
    """Define problem shapes representing common LLM configurations."""
    return [
        ProblemShape(
            id="llama2_7b_short",
            op_type="attention",
            params={"B": 8, "H": 32, "S_q": 512, "S_kv": 512, "D": 128},
        ),
        ProblemShape(
            id="llama2_7b_long",
            op_type="attention",
            params={"B": 8, "H": 32, "S_q": 4096, "S_kv": 4096, "D": 128},
        ),
        ProblemShape(
            id="gpt3_175b",
            op_type="attention",
            params={"B": 1, "H": 96, "S_q": 2048, "S_kv": 2048, "D": 128},
        ),
    ]


def main():
    """Run the attention kernel comparison.

    Usage:
        # CLI equivalent:
        llm-kernel-lab bench \\
            --op attention \\
            --kernels flash_attn_v2_triton,sdpa_pytorch \\
            --shapes llama2_7b_short,llama2_7b_long,gpt3_175b \\
            --repetitions 100 \\
            --profile-level timing \\
            --output ./results/attention_comparison.json

        # Then generate report:
        llm-kernel-lab report \\
            --input ./results/attention_comparison.json \\
            --format markdown \\
            --out ./reports/attention_report.md
    """
    kernels = define_kernels()
    shapes = define_shapes()

    print("Attention Kernel Comparison Setup:")
    print(f"  Kernels: {[k.id for k in kernels]}")
    print(f"  Shapes: {[s.id for s in shapes]}")
    print()
    print("To run this comparison, use the CLI command shown in the docstring,")
    print("or use the BenchmarkRunner API with appropriate kernel functions.")


if __name__ == "__main__":
    main()
