from __future__ import annotations

import queue
import threading
from pathlib import Path

import numpy as np
from PIL import Image

SENTINEL = object()


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32) / 255.0


def threaded_load(path: str | Path, readers: int = 4, image_size: int = 96) -> list[np.ndarray]:
    file_queue: queue.Queue[Path | object] = queue.Queue()
    result_queue: queue.Queue[np.ndarray] = queue.Queue()
    lock = threading.Lock()
    processed = {"count": 0}

    def producer() -> None:
        files = sorted(p for p in Path(path).rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        print(f"[producer] found {len(files)} files")
        for file_path in files:
            file_queue.put(file_path)
        for _ in range(readers):
            file_queue.put(SENTINEL)
        print("[producer] scan completed")

    def consumer(index: int) -> None:
        while True:
            item = file_queue.get()
            try:
                if item is SENTINEL:
                    print(f"[reader-{index}] stopped")
                    return
                array = load_image(item, image_size=image_size)
                result_queue.put(array)
                with lock:
                    processed["count"] += 1
                    count = processed["count"]
                if count <= 3 or count % 10 == 0:
                    print(f"[reader-{index}] loaded {item.name}; total={count}")
            finally:
                file_queue.task_done()

    producer_thread = threading.Thread(target=producer, name="producer")
    consumers = [threading.Thread(target=consumer, args=(i + 1,), name=f"reader-{i+1}") for i in range(readers)]

    producer_thread.start()
    for thread in consumers:
        thread.start()

    producer_thread.join()
    file_queue.join()
    for thread in consumers:
        thread.join()

    arrays: list[np.ndarray] = []
    while not result_queue.empty():
        arrays.append(result_queue.get())
    return arrays
