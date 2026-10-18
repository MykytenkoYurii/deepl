"""Матриця плутанини, вартість помилок і швидкий пошук порога за O(n log n).

Правило рішення: шахрайство (позитив), якщо s >= t.
Вартість: C(t) = 10 * FN(t) + FP(t).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

COST_FN = 10
COST_FP = 1


def cost(fn: int | np.ndarray, fp: int | np.ndarray) -> int | np.ndarray:
    return COST_FN * fn + COST_FP * fp


def confusion_at(scores: np.ndarray, labels: np.ndarray, t: float) -> dict[str, int]:
    """Пряме застосування правила s >= t до одного порога."""
    pred = scores >= t
    pos = labels == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": int(cost(fn, fp))}


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    """None означає, що метрика не визначена (нульовий знаменник)."""
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    return precision, recall


def threshold_table(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Усі кандидати (унікальні оцінки + +inf) з TP, FP, FN, TN, C.

    Оцінки сортуються за спаданням. Для порога t, що дорівнює деякій оцінці,
    позитивними є всі об'єкти з s >= t, тобто префікс відсортованого масиву
    до ОСТАННЬОГО елемента групи рівних оцінок включно. Тому TP і FP беремо з
    кумулятивних сум саме на кінцях груп: рівні оцінки отримують один прогноз.
    Порядок рядків: +inf (усі негативні), далі пороги за спаданням; останній
    рядок - найменша оцінка (усі позитивні).
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    total_pos = int(np.count_nonzero(labels == 1))
    total_neg = n - total_pos

    order = np.argsort(-scores, kind="stable")  # O(n log n)
    s_sorted = scores[order]
    is_pos = labels[order] == 1
    tp_cum = np.cumsum(is_pos)
    fp_cum = np.cumsum(~is_pos)

    # Індекс останнього елемента кожної групи рівних оцінок.
    group_end = np.r_[np.flatnonzero(s_sorted[1:] != s_sorted[:-1]), n - 1] if n else np.array([], int)

    thresholds = np.r_[np.inf, s_sorted[group_end].astype(np.float64)]
    tp = np.r_[0, tp_cum[group_end]].astype(np.int64)
    fp = np.r_[0, fp_cum[group_end]].astype(np.int64)
    fn = total_pos - tp
    tn = total_neg - fp
    return pd.DataFrame(
        {"threshold": thresholds, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": cost(fn, fp)}
    )


def select_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок з мінімальною C; за кількох мінімумів - найбільший поріг."""
    best = table[table["C"] == table["C"].min()]
    return best.loc[best["threshold"].idxmax()]


def format_threshold(t: float) -> str:
    return "+inf" if math.isinf(t) else repr(float(t))
