from __future__ import annotations

import numpy as np


def augment_one(payload: tuple[np.ndarray, int]) -> np.ndarray:
    image, seed = payload
    rng = np.random.default_rng(seed)
    result = image.copy()
    k = int(rng.integers(0, 4))
    result = np.rot90(result, k=k)
    if rng.random() > 0.5:
        result = np.fliplr(result)
    noise = rng.normal(0.0, 0.025, size=result.shape).astype(np.float32)
    result = np.clip(result + noise, 0.0, 1.0)
    # A few arithmetic passes make the function meaningfully CPU-bound.
    for _ in range(3):
        result = np.clip(np.sqrt(result + 1e-6) ** 2, 0.0, 1.0)
    return result.astype(np.float32)
