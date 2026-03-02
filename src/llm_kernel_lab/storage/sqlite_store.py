"""SQLite-based storage for benchmark results (optional, for long-term tracking)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from llm_kernel_lab.model import HardwareProfile, KernelRun
from llm_kernel_lab.serialization import (
    hardware_profile_from_dict,
    hardware_profile_to_dict,
    kernel_run_from_dict,
    kernel_run_to_dict,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hardware (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_name TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kernel_runs (
    id TEXT PRIMARY KEY,
    hardware_id INTEGER NOT NULL,
    data TEXT NOT NULL,
    timestamp REAL NOT NULL,
    FOREIGN KEY (hardware_id) REFERENCES hardware(id)
);
"""


class SqliteStore:
    """SQLite storage backend for long-term result tracking."""

    def __init__(self, path: str | Path = "./results/results.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)

    def save(self, runs: list[KernelRun], hardware: HardwareProfile) -> None:
        hw_data = json.dumps(hardware_profile_to_dict(hardware))
        cursor = self._conn.execute(
            "INSERT INTO hardware (gpu_name, data) VALUES (?, ?)",
            (hardware.gpu_name, hw_data),
        )
        hw_id = cursor.lastrowid

        for run in runs:
            run_data = json.dumps(kernel_run_to_dict(run))
            self._conn.execute(
                "INSERT OR REPLACE INTO kernel_runs (id, hardware_id, data, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (run.id, hw_id, run_data, run.timestamp),
            )

        self._conn.commit()

    def load_all(self) -> list[KernelRun]:
        cursor = self._conn.execute("SELECT data FROM kernel_runs ORDER BY timestamp")
        return [kernel_run_from_dict(json.loads(row[0])) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()
