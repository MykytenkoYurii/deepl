"""Завантаження Iris, стратифікований поділ 70/30 та стандартизація."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35  # 35 train + 15 test у кожному з 3 класів


@dataclass
class Dataset:
    X_train: np.ndarray  # (105, 4), float64, стандартизовано
    y_train: np.ndarray  # (105,), int64
    X_test: np.ndarray   # (45, 4), float64, стандартизовано статистиками train
    y_test: np.ndarray   # (45,), int64
    mean: np.ndarray     # (4,) середнє train
    std: np.ndarray      # (4,) std train, ddof=0


def load_data(seed: int = 0) -> Dataset:
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):                      # класи строго в порядку 0, 1, 2
        idx = np.flatnonzero(y == c)         # індекси класу c (50 шт.)
        rng.shuffle(idx)                     # перемішування in-place
        train_idx.append(idx[:N_TRAIN_PER_CLASS])
        test_idx.append(idx[N_TRAIN_PER_CLASS:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Статистики тільки з навчальної вибірки, ddof=0
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    return Dataset(
        X_train=(X_train - mean) / std,
        y_train=y_train,
        X_test=(X_test - mean) / std,
        y_test=y_test,
        mean=mean,
        std=std,
    )
