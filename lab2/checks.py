"""Незалежна перевірка пошуку порога прямим перебором на наборах A, B, C.

Перевірочна реалізація не використовує threshold_table / select_threshold:
кандидати, лічильники, вартість і вибір порога рахуються окремо, циклами.
"""

from __future__ import annotations

import math

import pandas as pd

from threshold import format_threshold, select_threshold, threshold_table

SMALL_SETS: dict[str, tuple[list[float], list[int]]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0]),
}


def brute_force_table(scores: list[float], labels: list[int]) -> pd.DataFrame:
    """Для кожного кандидата окремо проходить усю вибірку за правилом s >= t."""
    candidates = [math.inf] + sorted(set(scores), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for s, y in zip(scores, labels, strict=True):
            predicted_fraud = s >= t
            if predicted_fraud and y == 1:
                tp += 1
            elif predicted_fraud and y == 0:
                fp += 1
            elif y == 1:
                fn += 1
            else:
                tn += 1
        rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": 10 * fn + fp})
    return pd.DataFrame(rows)


def brute_force_select(rows: pd.DataFrame) -> float:
    best_cost = None
    best_t = None
    for t, c in zip(rows["threshold"], rows["C"], strict=True):
        if best_cost is None or c < best_cost or (c == best_cost and t > best_t):
            best_cost, best_t = c, t
    return best_t


def run_checks() -> tuple[bool, list[dict[str, object]]]:
    """Повертає (усі пройдено, результати по наборах)."""
    results = []
    for name, (scores, labels) in SMALL_SETS.items():
        slow = brute_force_table(scores, labels)
        fast = threshold_table(pd.Series(scores).to_numpy(), pd.Series(labels).to_numpy())
        slow_t = brute_force_select(slow)
        fast_t = float(select_threshold(fast)["threshold"])
        tables_equal = slow.reset_index(drop=True).equals(fast.reset_index(drop=True))
        results.append(
            {
                "set": name,
                "scores": scores,
                "labels": labels,
                "slow": slow,
                "fast": fast,
                "slow_t": slow_t,
                "fast_t": fast_t,
                "tables_equal": tables_equal,
                "threshold_equal": slow_t == fast_t,
            }
        )
    passed = all(r["tables_equal"] and r["threshold_equal"] for r in results)
    return passed, results


def checks_markdown(results: list[dict[str, object]]) -> str:
    lines = ["# Перевірка пошуку порога на малих наборах", ""]
    for r in results:
        lines += [
            f"## Набір {r['set']}",
            "",
            f"s = {r['scores']}",
            "",
            f"y = {r['labels']}",
            "",
            "| Поріг t | TP | FP | FN | TN | C (перебір) | C (кум. суми) | Лічильники збігаються |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s, f in zip(r["slow"].itertuples(), r["fast"].itertuples(), strict=True):
            same = s[1:] == f[1:]
            lines.append(
                f"| {format_threshold(s.threshold)} | {s.TP} | {s.FP} | {s.FN} | "
                f"{s.TN} | {s.C} | {f.C} | {'так' if same else '**ні**'} |"
            )
        lines += [
            "",
            f"- Обраний поріг (перебір): **{format_threshold(r['slow_t'])}**",
            f"- Обраний поріг (кумулятивні суми): **{format_threshold(r['fast_t'])}**",
            f"- Таблиці збігаються точно: {'так' if r['tables_equal'] else '**ні**'}; "
            f"пороги збігаються: {'так' if r['threshold_equal'] else '**ні**'}",
            "",
        ]
    return "\n".join(lines)
