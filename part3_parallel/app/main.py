from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .performance import benchmark, save_report
from .producer_consumer import threaded_load


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Threaded loading and multiprocessing augmentation")
    parser.add_argument("--path", required=True)
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--report", default="data/output/performance_report.json")
    parser.add_argument("--output", default="data/output/augmented.npy")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    arrays = threaded_load(args.path, readers=args.readers, image_size=args.image_size)
    if not arrays:
        raise SystemExit("No images found")

    result, report = benchmark(arrays, processes=args.processes)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, result)
    report.update({
        "readers_threads": args.readers,
        "image_size": args.image_size,
        "output_array": str(output),
    })
    save_report(report, args.report)

    print("PARALLEL AUGMENTATION COMPLETED")
    print(f"Files processed: {report['files_processed']}")
    print(f"Sequential: {report['sequential_seconds']:.6f} s")
    print(f"Multiprocessing: {report['multiprocessing_seconds']:.6f} s")
    print(f"Speedup: {report['speedup']:.4f}x")
    print(f"Report saved: {args.report}")


if __name__ == "__main__":
    main()
