"""Двошарова мережа X -> Linear(4,8) -> ReLU -> Linear(8,3) на чистому NumPy.

Прямий прохід, стабільна крос-ентропія та ручний backprop. Автоматичного
диференціювання тут немає.
"""

import numpy as np

D_IN, D_HIDDEN, D_OUT = 4, 8, 3


def init_params(seed: int = 0) -> dict[str, np.ndarray]:
    """Новий генератор default_rng(seed): спершу W1 (He), потім W2 (Xavier)."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2.0 / D_IN), size=(D_IN, D_HIDDEN))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (D_HIDDEN + D_OUT)), size=(D_HIDDEN, D_OUT))
    return {
        "W1": W1,
        "b1": np.zeros(D_HIDDEN, dtype=np.float64),
        "W2": W2,
        "b2": np.zeros(D_OUT, dtype=np.float64),
    }


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """log-softmax по рядках зі зсувом на максимум (без overflow в exp)."""
    Z_shift = Z - Z.max(axis=1, keepdims=True)
    return Z_shift - np.log(np.exp(Z_shift).sum(axis=1, keepdims=True))


def cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Середня крос-ентропія. Повертає (loss, log_probs)."""
    log_p = log_softmax(logits)
    N = logits.shape[0]
    loss = -log_p[np.arange(N), y].mean()
    return float(loss), log_p


class NumpyMLP:
    def __init__(self, params: dict[str, np.ndarray], buggy: bool = False):
        # Копія, щоб зовнішні зміни не впливали на модель
        self.params = {k: v.copy() for k, v in params.items()}
        self.buggy = buggy          # True -> навмисна помилка (без ділення на N)
        self.cache: dict | None = None

    def forward(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.params
        Z1 = X @ p["W1"] + p["b1"]          # (N, 8)
        H = np.maximum(Z1, 0.0)             # (N, 8)
        Z2 = H @ p["W2"] + p["b2"]          # (N, 3) логіти
        loss, log_p = cross_entropy(Z2, y)
        P = np.exp(log_p)                   # (N, 3) softmax-ймовірності
        # Проміжні значення, потрібні для backward
        self.cache = {"X": X, "y": y, "Z1": Z1, "H": H, "P": P}
        return loss

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Лише значення втрати (для чисельного диференціювання)."""
        return self.forward(X, y)

    def backward(self) -> dict[str, np.ndarray]:
        if self.cache is None:
            raise RuntimeError("Спершу викличте forward()")
        c, p = self.cache, self.params
        X, y, Z1, H, P = c["X"], c["y"], c["Z1"], c["H"], c["P"]
        N = X.shape[0]

        Y = np.zeros_like(P)
        Y[np.arange(N), y] = 1.0            # one-hot (N, 3)

        dZ2 = P - Y                         # (N, 3)
        if not self.buggy:
            dZ2 = dZ2 / N                   # усереднення за об'єктами

        dW2 = H.T @ dZ2                     # (8, 3)
        db2 = dZ2.sum(axis=0)               # (3,)
        dH = dZ2 @ p["W2"].T                # (N, 8)
        dZ1 = dH * (Z1 > 0)                 # (N, 8) похідна ReLU
        dW1 = X.T @ dZ1                     # (4, 8)
        db1 = dZ1.sum(axis=0)               # (8,)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
