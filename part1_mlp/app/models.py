from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .exceptions import InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """Simple fully-connected multilayer perceptron for binary classification."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Iterable[int],
        output_size: int = 1,
        activation: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        sizes = [input_size, *list(hidden_sizes), output_size]
        if any((not isinstance(size, int)) or size <= 0 for size in sizes):
            raise InvalidLayerSizeError("All layer sizes must be positive integers")
        if activation not in {"sigmoid", "relu"}:
            raise ValueError("activation must be 'sigmoid' or 'relu'")

        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.output_size = output_size
        self.activation_name = activation
        self.rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.loss_history: list[float] = []
        self._init_parameters()

    def _init_parameters(self) -> None:
        sizes = [self.input_size, *self.hidden_sizes, self.output_size]
        self.weights.clear()
        self.biases.clear()
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / fan_in) if self.activation_name == "relu" else np.sqrt(1.0 / fan_in)
            self.weights.append(self.rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=float))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _hidden_activation(self, z: np.ndarray) -> np.ndarray:
        return self._sigmoid(z) if self.activation_name == "sigmoid" else np.maximum(0.0, z)

    def _hidden_derivative(self, z: np.ndarray) -> np.ndarray:
        if self.activation_name == "sigmoid":
            s = self._sigmoid(z)
            return s * (1.0 - s)
        return (z > 0.0).astype(float)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.input_size:
            raise MismatchedDataError(
                f"Expected {self.input_size} input features, got {x.shape[1]}"
            )

        activations = [x]
        zs: list[np.ndarray] = []
        a = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = a @ weight + bias
            zs.append(z)
            if index == len(self.weights) - 1:
                a = self._sigmoid(z)
            else:
                a = self._hidden_activation(z)
            activations.append(a)
        return a, zs, activations

    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1, 1)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

    def _backward(
        self,
        y_true: np.ndarray,
        zs: list[np.ndarray],
        activations: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
        m = y_true.shape[0]
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y_true
        grad_w[-1] = activations[-2].T @ delta / m
        grad_b[-1] = np.mean(delta, axis=0, keepdims=True)

        for layer in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[layer + 1].T) * self._hidden_derivative(zs[layer])
            grad_w[layer] = activations[layer].T @ delta / m
            grad_b[layer] = np.mean(delta, axis=0, keepdims=True)
        return grad_w, grad_b

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 1400,
        learning_rate: float = 0.55,
        batch_size: int = 4,
        verbose_every: int = 0,
    ) -> list[float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise MismatchedDataError("X and y must contain the same number of samples")
        if x.shape[1] != self.input_size:
            raise MismatchedDataError(
                f"Expected {self.input_size} features, got {x.shape[1]}"
            )

        self.loss_history.clear()
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            x_shuffled = x[order]
            y_shuffled = y[order]
            for start in range(0, n, batch_size):
                xb = x_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]
                _, zs, activations = self.forward(xb)
                grad_w, grad_b = self._backward(yb, zs, activations)
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * grad_w[i]
                    self.biases[i] -= learning_rate * grad_b[i]

            pred, _, _ = self.forward(x)
            loss = self.binary_cross_entropy(y, pred)
            self.loss_history.append(loss)
            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                print(f"Epoch {epoch:4d}/{epochs}: loss={loss:.6f}")
        return self.loss_history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        pred, _, _ = self.forward(x)
        return pred

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    def save_weights(self, path: str | Path) -> None:
        payload = {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "activation": self.activation_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_weights(cls, path: str | Path) -> "NeuralNetwork":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(
            input_size=payload["input_size"],
            hidden_sizes=payload["hidden_sizes"],
            output_size=payload["output_size"],
            activation=payload["activation"],
        )
        model.weights = [np.asarray(w, dtype=float) for w in payload["weights"]]
        model.biases = [np.asarray(b, dtype=float) for b in payload["biases"]]
        return model
