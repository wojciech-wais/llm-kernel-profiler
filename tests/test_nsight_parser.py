"""Tests for Nsight Compute output parsing."""

from llm_kernel_lab.profiler.nsight import ncu_metrics_to_metric_set, parse_ncu_csv

SAMPLE_CSV = """\
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","my_kernel","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","78.5"
"0","my_kernel","smsp__warps_active.avg.pct_of_peak_sustained_active","%","72.3"
"0","my_kernel","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","85.2"
"0","my_kernel","l2tex__t_bytes.sum","byte","1073741824"
"0","my_kernel","lts__t_sector_hit_rate.pct","%","65.0"
"0","my_kernel","launch__registers_per_thread","register/thread","64"
"0","my_kernel","launch__shared_mem_per_block","byte/block","32768"
"""


def test_parse_ncu_csv():
    results = parse_ncu_csv(SAMPLE_CSV)
    assert len(results) == 1
    kernel = results[0]
    assert kernel["kernel_name"] == "my_kernel"
    assert kernel["sm__throughput.avg.pct_of_peak_sustained_elapsed"] == 78.5
    assert kernel["smsp__warps_active.avg.pct_of_peak_sustained_active"] == 72.3
    assert kernel["launch__registers_per_thread"] == 64.0


def test_parse_ncu_csv_multiple_kernels():
    csv = """\
"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"
"0","kernel_a","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","70.0"
"1","kernel_b","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","80.0"
"""
    results = parse_ncu_csv(csv)
    assert len(results) == 2
    assert results[0]["kernel_name"] == "kernel_a"
    assert results[1]["kernel_name"] == "kernel_b"


def test_ncu_metrics_to_metric_set():
    raw = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 78.5,
        "smsp__warps_active.avg.pct_of_peak_sustained_active": 72.3,
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": 85.2,
        "l2tex__t_bytes.sum": 1073741824.0,
        "lts__t_sector_hit_rate.pct": 65.0,
        "launch__registers_per_thread": 64,
        "launch__shared_mem_per_block": 32768,
    }

    ms = ncu_metrics_to_metric_set(
        raw=raw,
        runtime_ms=3.0,
        flops=1e12,
        bytes_accessed=1e9,
        peak_bw_gbps=1555.0,
    )

    assert ms.sm_efficiency == 0.785
    assert ms.occupancy == 0.723
    assert abs(ms.dram_bw_gbps - 0.852 * 1555.0) < 1.0
    assert ms.registers_per_thread == 64
    assert ms.shared_mem_bytes == 32768
    assert ms.l2_hit_rate == 0.65
    assert ms.achieved_tflops > 0
