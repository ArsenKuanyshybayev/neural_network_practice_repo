from pathlib import Path

import numpy as np

from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import save_loss_plot


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dataset_path = root / "data" / "xor.csv"
    output_dir = root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager()
    manager.load_csv(dataset_path)
    manager.standardize()
    x_train, x_test, y_train, y_test = manager.train_test_split(test_ratio=0.25, seed=11)

    model = NeuralNetwork(2, [6], 1, activation="sigmoid", seed=7)
    print("PART 1 DEMO - OOP MULTILAYER PERCEPTRON")
    print(f"1) Network created: layers=[2, 6, 1]; activation=sigmoid")
    print(f"2) Dataset loaded: {len(manager.x)} rows; train={len(x_train)}; test={len(x_test)}")
    print("3) Training started")
    model.fit(x_train, y_train, epochs=1400, learning_rate=0.55, batch_size=4, verbose_every=200)

    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)
    accuracy = float((pred == y_test).mean())
    print(f"4) Test accuracy: {accuracy:.3f}")
    for i, (p, cls, expected) in enumerate(zip(proba.ravel(), pred.ravel(), y_test.ravel()), start=1):
        print(f"   sample {i}: probability={p:.4f}; predicted={int(cls)}; expected={int(expected)}")

    weights_path = output_dir / "xor_weights.json"
    plot_path = output_dir / "part1_loss.png"
    model.save_weights(weights_path)
    save_loss_plot(model.loss_history, plot_path)
    restored = NeuralNetwork.load_weights(weights_path)
    restored_prediction = float(restored.predict_proba(x_test[:1])[0, 0])
    print(f"5) Weights saved and loaded: {weights_path}; restored_prediction={restored_prediction:.4f}")
    print(f"6) Loss graph saved: {plot_path}")


if __name__ == "__main__":
    main()
