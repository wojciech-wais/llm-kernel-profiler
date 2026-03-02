"""Tests for JSON storage backend."""

import tempfile
from pathlib import Path

from llm_kernel_lab.model import KernelRun
from llm_kernel_lab.storage.json_store import JsonStore
from tests.test_model import make_hw, make_metrics, make_shape, make_variant


def _make_run(run_id: str = "test_run") -> KernelRun:
    return KernelRun(
        id=run_id,
        variant=make_variant(),
        shape=make_shape(),
        hardware=make_hw(),
        metrics=make_metrics(),
        raw_profiler_output_path="",
        timestamp=1700000000.0,
    )


def test_json_store_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonStore(tmpdir)
        hw = make_hw()
        runs = [_make_run("run_1"), _make_run("run_2")]

        path = store.save(runs, hw, name="test_results")
        assert path.exists()

        hw2, loaded = store.load("test_results")
        assert hw2.gpu_name == hw.gpu_name
        assert len(loaded) == 2
        assert loaded[0].id == "run_1"


def test_json_store_list_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonStore(tmpdir)
        hw = make_hw()

        store.save([_make_run()], hw, name="results_a")
        store.save([_make_run()], hw, name="results_b")

        files = store.list_results()
        assert len(files) == 2
        names = [f.stem for f in files]
        assert "results_a" in names
        assert "results_b" in names
