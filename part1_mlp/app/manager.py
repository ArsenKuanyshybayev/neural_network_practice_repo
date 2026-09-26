from __future__ import annotations

from pathlib import Path

import numpy as np

from .exceptions import DataNotLoadedError, MismatchedDataError


class DatasetManager:
    """Loads a numerical CSV, standardizes features and creates a train/test split."""

    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def load_csv(self, path: str | Path, has_header: bool = True) -> None:
        data = np.genfromtxt(path, delimiter=",", skip_header=1 if has_header else 0)
        if data.ndim != 2 or data.shape[1] < 2:
            raise MismatchedDataError("CSV must contain at least one feature and one target column")
        self.x = data[:, :-1].astype(float)
        self.y = data[:, -1].astype(float).reshape(-1, 1)

    def standardize(self) -> None:
        if self.x is None:
            raise DataNotLoadedError("Load data before standardization")
        self.mean_ = self.x.mean(axis=0)
        self.std_ = self.x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.x = (self.x - self.mean_) / self.std_

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise DataNotLoadedError("Standardization parameters are unavailable")
        return (np.asarray(x, dtype=float) - self.mean_) / self.std_

    def train_test_split(self, test_ratio: float = 0.25, seed: int = 42):
        if self.x is None or self.y is None:
            raise DataNotLoadedError("Load data before splitting")
        if not 0.0 < test_ratio < 1.0:
            raise ValueError("test_ratio must be between 0 and 1")
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.x))
        n_test = max(1, int(round(len(idx) * test_ratio)))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        return self.x[train_idx], self.x[test_idx], self.y[train_idx], self.y[test_idx]
