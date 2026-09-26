"""Перевірки: звірка з PyTorch та центральна скінченна різниця."""

import numpy as np

from mlp_numpy import NumpyMLP
from mlp_torch import torch_loss_and_grads

TOL_TORCH = 1e-12
TOL_NUMERIC = 1e-7
EPS = 1e-6
PARAM_NAMES = ("W1", "b1", "W2", "b2")
NUMERIC_TARGETS = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))


def compare_with_torch(model: NumpyMLP, X: np.ndarray, y: np.ndarray) -> dict:
    """Втрата та всі елементи 4 градієнтів NumPy vs PyTorch."""
    loss_np = model.forward(X, y)
    grads_np = model.backward()
    loss_pt, grads_pt = torch_loss_and_grads(model.params, X, y)

    rows = []
    loss_diff = abs(loss_np - loss_pt)
    rows.append({
        "name": "Втрата",
        "max_abs_diff": loss_diff,
        "finite": bool(np.isfinite(loss_np) and np.isfinite(loss_pt)),
    })
    for name in PARAM_NAMES:
        a, b = grads_np[name], grads_pt[name]
        assert a.shape == b.shape, f"{name}: {a.shape} vs {b.shape}"
        rows.append({
            "name": f"Градієнт {name}",
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "finite": bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b))),
        })
    for r in rows:
        # Критерій |a - b| <= 1e-12 для кожної пари скалярів = для максимуму
        r["passed"] = r["finite"] and r["max_abs_diff"] <= TOL_TORCH

    return {"loss_np": loss_np, "loss_pt": loss_pt, "rows": rows,
            "grads_np": grads_np, "grads_pt": grads_pt}


def numerical_derivative(model: NumpyMLP, X, y, name: str, idx: tuple,
                         eps: float = EPS) -> float:
    theta = model.params[name]
    original = theta[idx]                    # 1. зберегти початкове значення
    theta[idx] = original + eps              # 2. +eps
    L_plus = model.loss(X, y)
    theta[idx] = original - eps              # 3. -eps від початкового
    L_minus = model.loss(X, y)
    theta[idx] = original                    # 4. відновити
    return (L_plus - L_minus) / (2.0 * eps)


def numerical_check(model: NumpyMLP, X: np.ndarray, y: np.ndarray) -> list[dict]:
    # Аналітичний градієнт на початкових вагах
    model.forward(X, y)
    grads = model.backward()
    rows = []
    for name, idx in NUMERIC_TARGETS:
        g_manual = float(grads[name][idx])
        g_num = numerical_derivative(model, X, y, name, idx)
        diff = abs(g_num - g_manual)
        finite = bool(np.isfinite(g_manual) and np.isfinite(g_num))
        rows.append({
            "param": f"{name}[{', '.join(map(str, idx))}]",
            "g_manual": g_manual,
            "g_num": g_num,
            "abs_diff": diff,
            "passed": finite and diff <= TOL_NUMERIC,
        })
    return rows
