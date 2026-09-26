from pathlib import Path

import numpy as np

from .exceptions import DataNotLoadedError, ModelNotInitializedError
from .manager import DatasetManager
from .models import NeuralNetwork
from .utils import save_loss_plot


def ask_int(prompt: str) -> int:
    return int(input(prompt).strip())


def main() -> None:
    model: NeuralNetwork | None = None
    data = DatasetManager()
    split = None

    while True:
        print("\n1. Create network")
        print("2. Load CSV and split")
        print("3. Train")
        print("4. Predict")
        print("5. Save/load weights")
        print("6. Save loss plot")
        print("7. Exit")
        choice = input("Select: ").strip()

        try:
            if choice == "1":
                input_size = ask_int("Input size: ")
                hidden = [int(v) for v in input("Hidden sizes, comma-separated: ").split(",") if v.strip()]
                activation = input("Activation (sigmoid/relu): ").strip().lower()
                model = NeuralNetwork(input_size, hidden, 1, activation=activation)
                print("Model created")
            elif choice == "2":
                path = input("CSV path: ").strip()
                data.load_csv(path)
                data.standardize()
                split = data.train_test_split()
                print("Dataset loaded, standardized and split")
            elif choice == "3":
                if model is None:
                    raise ModelNotInitializedError("Create or load a model first")
                if split is None:
                    raise DataNotLoadedError("Load data first")
                x_train, _, y_train, _ = split
                epochs = ask_int("Epochs [e.g. 1400]: ")
                lr = float(input("Learning rate [e.g. 0.55]: "))
                batch = ask_int("Batch size [e.g. 4]: ")
                model.fit(x_train, y_train, epochs=epochs, learning_rate=lr, batch_size=batch, verbose_every=max(1, epochs // 7))
            elif choice == "4":
                if model is None:
                    raise ModelNotInitializedError("Create or load a model first")
                values = np.array([float(v) for v in input("Features, comma-separated: ").split(",")])
                if data.mean_ is not None:
                    values = data.transform(values)
                p = float(model.predict_proba(values)[0, 0])
                print(f"Probability={p:.6f}; class={int(p >= 0.5)}")
            elif choice == "5":
                mode = input("save or load? ").strip().lower()
                path = Path(input("JSON path: ").strip())
                if mode == "save":
                    if model is None:
                        raise ModelNotInitializedError("Create or load a model first")
                    model.save_weights(path)
                    print("Saved")
                elif mode == "load":
                    model = NeuralNetwork.load_weights(path)
                    print("Loaded")
                else:
                    print("Unknown mode")
            elif choice == "6":
                if model is None or not model.loss_history:
                    raise ModelNotInitializedError("Train a model first")
                path = input("PNG path: ").strip()
                save_loss_plot(model.loss_history, path)
                print("Plot saved")
            elif choice == "7":
                print("Bye")
                break
            else:
                print("Unknown menu item")
        except (ValueError, OSError, DataNotLoadedError, ModelNotInitializedError) as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
