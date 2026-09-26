from pathlib import Path

import matplotlib.pyplot as plt


def save_loss_plot(loss_history: list[float], path: str | Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
