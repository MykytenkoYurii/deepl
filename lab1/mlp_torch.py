"""Еталонна реалізація в PyTorch (autograd) — лише для звірки."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_torch_model(params: dict[str, np.ndarray]) -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)
    ).to(torch.float64)
    with torch.no_grad():
        # nn.Linear зберігає weight як (out, in) -> транспонуємо W з (in, out)
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"]))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"]))
    return model


def torch_loss_and_grads(
    params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray
) -> tuple[float, dict[str, np.ndarray]]:
    model = build_torch_model(params)
    logits = model(torch.from_numpy(X))
    loss = F.cross_entropy(logits, torch.from_numpy(y), reduction="mean")
    loss.backward()
    grads = {
        # Повертаємо градієнти у форматі NumPy-мережі (in, out)
        "W1": model[0].weight.grad.T.numpy().copy(),
        "b1": model[0].bias.grad.numpy().copy(),
        "W2": model[2].weight.grad.T.numpy().copy(),
        "b2": model[2].bias.grad.numpy().copy(),
    }
    return float(loss.item()), grads
