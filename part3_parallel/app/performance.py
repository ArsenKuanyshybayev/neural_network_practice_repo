from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .augmentation import augment_one


def benchmark(arrays: list[np.ndarray], processes: int = 4) -> tuple[np.ndarray, dict]:
    payload = [(array, i) for i, array in enumerate(arrays)]

    t0 = time.perf_counter()
    sequential = [augment_one(item) for item in payload]
    sequential_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    with mp.Pool(processes=processes) as pool:
        parallel = pool.map(augment_one, payload)
    multiprocessing_seconds = time.perf_counter() - t1

    seq = np.stack(sequential)
    par = np.stack(parallel)
    report = {
        "files_processed": len(arrays),
        "processes": processes,
        "sequential_seconds": sequential_seconds,
        "multiprocessing_seconds": multiprocessing_seconds,
        "speedup": sequential_seconds / multiprocessing_seconds if multiprocessing_seconds else None,
        "same_output_shape": seq.shape == par.shape,
    }
    return par, report


def save_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
