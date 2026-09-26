from __future__ import annotations

from pathlib import Path
from typing import Iterable


def normalize_extensions(values: Iterable[str]) -> set[str]:
    result = set()
    for value in values:
        value = value.strip().lower()
        if not value:
            continue
        result.add(value if value.startswith(".") else f".{value}")
    return result


def scan_files(path: str | Path, extensions: Iterable[str]) -> list[Path]:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    allowed = normalize_extensions(extensions)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed)
