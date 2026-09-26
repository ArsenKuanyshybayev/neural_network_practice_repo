from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    log_level: str = os.getenv("NN_LOG_LEVEL", "INFO")
    image_size: int = int(os.getenv("NN_IMAGE_SIZE", "64"))
    default_output: str = os.getenv("NN_OUTPUT", "data/output/prepared.npy")
