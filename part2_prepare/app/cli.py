from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .logging_config import configure_logging
from .scanner import scan_files


def load_as_array(path: Path, image_size: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((image_size, image_size))
            return np.asarray(image, dtype=np.float32) / 255.0
    if suffix == ".csv":
        return np.genfromtxt(path, delimiter=",", skip_header=1).astype(np.float32)
    raise ValueError(f"Unsupported extension: {suffix}")


def prepare(path: str, extensions: list[str], output: str, image_size: int) -> None:
    started = time.perf_counter()
    files = scan_files(path, extensions)
    if not files:
        raise FileNotFoundError("No matching files found")

    arrays = [load_as_array(file_path, image_size=image_size) for file_path in files]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All arrays must have equal shapes; got: {sorted(shapes)}")

    combined = np.stack(arrays)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, combined)
    elapsed = time.perf_counter() - started

    print("PREPARE COMPLETED")
    print(f"Files processed: {len(files)}")
    print(f"Combined shape: {combined.shape}")
    print(f"Dtype: {combined.dtype}")
    print(f"Normalized range: {combined.min():.4f} .. {combined.max():.4f}")
    print(f"Elapsed: {elapsed:.4f} s")
    print(f"Saved: {output_path}")


def doctor() -> None:
    print("ENVIRONMENT DOCTOR")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {Path.cwd()}")
    for package in ["numpy", "PIL", "matplotlib", "torch", "tensorflow"]:
        status = "installed" if importlib.util.find_spec(package) else "not installed"
        print(f"{package:<12}: {status}")

    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first_line = (result.stdout or result.stderr).splitlines()
        print(f"CUDA/GPU: {first_line[0] if first_line else 'nvidia-smi returned no output'}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("CUDA/GPU: nvidia-smi not available")


def build_parser() -> argparse.ArgumentParser:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Dataset preparation utility")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="recursively prepare files as a NumPy array")
    p_prepare.add_argument("--path", required=True)
    p_prepare.add_argument("--ext", nargs="+", default=["png"])
    p_prepare.add_argument("--output", default=settings.default_output)
    p_prepare.add_argument("--image-size", type=int, default=settings.image_size)

    sub.add_parser("doctor", help="print environment diagnostics")
    return parser


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args.path, args.ext, args.output, args.image_size)
    else:
        doctor()


if __name__ == "__main__":
    main()
