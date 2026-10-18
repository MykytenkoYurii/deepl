"""ЛР2, пункти 2-4: аналіз порогів зі збережених прогнозів, без навчання і без CSV.

    uv run python analysis.py --saved results --out analysis

Порядок: криві й пошук t* на validation -> перевірка пошуку на наборах A/B/C ->
лише після успішних перевірок застосування зафіксованого t* до test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from checks import checks_markdown, run_checks
from lab2_starter import load_predictions, split_summary
from threshold import (
    confusion_at,
    format_threshold,
    precision_recall,
    select_threshold,
    threshold_table,
)

N_ERRORS = 5


def fmt_metric(x: float | None) -> str:
    return "не визначено" if x is None else f"{x:.4f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def plot_curves(y: np.ndarray, s: np.ndarray, marks: dict[str, float], out: Path) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)
    base = y.mean()

    points = {}
    for label, t in marks.items():
        c = confusion_at(s, y, t)
        p, r = precision_recall(c["TP"], c["FP"], c["FN"])
        points[label] = (p, r, c["FP"] / (c["FP"] + c["TN"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=1.5, label=f"PR-крива (AP = {ap:.4f})")
    ax.axhline(base, color="grey", ls="--", lw=1, label=f"частка шахрайства = {base:.5f}")
    for (label, (p, r, _)), m in zip(points.items(), ("o", "s"), strict=True):
        ax.plot(r, p, m, ms=8, label=label)
    ax.set(xlabel="Повнота (recall)", ylabel="Точність (precision)", xlim=(0, 1.01), ylim=(0, 1.02))
    ax.set_title("Validation: крива точність-повнота")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "pr_curve_validation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"ROC-крива (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="випадковий класифікатор")
    for (label, (_, r, f)), m in zip(points.items(), ("o", "s"), strict=True):
        ax.plot(f, r, m, ms=8, label=label)
    ax.set(xlabel="FPR = FP / (FP + TN)", ylabel="TPR = recall", xlim=(-0.01, 1), ylim=(0, 1.02))
    ax.set_title("Validation: ROC-крива")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    # Збільшена ділянка малих FPR: саме тут лежать робочі пороги.
    inset = ax.inset_axes([0.35, 0.3, 0.45, 0.4])
    inset.plot(fpr, tpr, lw=1.5)
    for (_, (_, r, f)), m in zip(points.items(), ("o", "s"), strict=True):
        inset.plot(f, r, m, ms=7)
    inset.set(xlim=(0, 0.002), ylim=(0.6, 0.9), title="FPR ≤ 0.002")
    inset.tick_params(labelsize=7)
    inset.title.set_fontsize(8)
    inset.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_curve_validation.png", dpi=150)
    plt.close(fig)
    return {"roc_auc": float(auc), "average_precision": float(ap)}


def rule_row(name: str, t: float, c: dict[str, int]) -> dict[str, object]:
    p, r = precision_recall(c["TP"], c["FP"], c["FN"])
    return {
        "Правило": name,
        "Поріг": format_threshold(t) if np.isinf(t) else f"{t:.6g}",
        **c,
        "Precision": fmt_metric(p),
        "Recall": fmt_metric(r),
    }


def error_examples(saved: dict[str, np.ndarray], t: float) -> pd.DataFrame:
    s = saved["test_scores"]
    y = saved["test_y"]
    pred = s >= t
    frame = pd.DataFrame(
        {
            "csv_row": saved["test_row_ids"],
            "label": y,
            "score": s.astype(np.float64),
            "amount": saved["test_amount"],
        }
    )
    frame["error"] = np.where(pred & (y == 0), "FP", np.where(~pred & (y == 1), "FN", ""))
    frame["distance_to_threshold"] = frame["score"] - t
    frame = frame[frame["error"] != ""].sort_values("csv_row", kind="stable")
    parts = [frame[frame["error"] == kind].head(N_ERRORS) for kind in ("FP", "FN")]
    return pd.concat(parts, ignore_index=True)[
        ["csv_row", "error", "label", "score", "amount", "distance_to_threshold"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saved", type=Path, default=Path("results"), help="Тека зі збереженими прогнозами")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Тека для результатів аналізу")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    saved = load_predictions(args.saved)
    summary = split_summary(saved)
    print("Поділ:")
    print(summary.to_string(index=False))

    # ---- 2. Validation: криві, повна таблиця кандидатів, t* ----
    y_va = saved["validation_y"]
    s_va = saved["validation_scores"]
    table = threshold_table(s_va, y_va)
    best = select_threshold(table)
    t_star = float(best["threshold"])

    csv_table = table.copy()
    csv_table["threshold"] = [format_threshold(t) for t in table["threshold"]]
    csv_table.to_csv(out / "threshold_candidates_validation.csv", index=False)

    curve_stats = plot_curves(
        y_va, s_va, {"поріг 0.5": 0.5, f"t* = {t_star:.4g}": t_star}, out
    )

    rules = pd.DataFrame(
        [
            rule_row("Стандартний поріг", 0.5, confusion_at(s_va, y_va, 0.5)),
            rule_row("Мінімум вартості", t_star, confusion_at(s_va, y_va, t_star)),
            rule_row("Усі негативні", np.inf, confusion_at(s_va, y_va, np.inf)),
        ]
    )
    # Пряме застосування правила має дати ті самі числа, що й таблиця кандидатів.
    for key in ("TP", "FP", "FN", "TN", "C"):
        assert rules.iloc[1][key] == best[key], key
    n_min = int((table["C"] == table["C"].min()).sum())
    print(f"\nValidation: {len(table)} кандидатів, ROC-AUC = {curve_stats['roc_auc']:.6f}, "
          f"AP = {curve_stats['average_precision']:.6f}")
    print(f"t* = {format_threshold(t_star)} (мінімум C = {int(best['C'])}, порогів із мінімумом: {n_min})")
    print(rules.to_string(index=False))

    # ---- 3. Перевірка пошуку на малих наборах ----
    passed, check_results = run_checks()
    (out / "checks.md").write_text(checks_markdown(check_results) + "\n", encoding="utf-8")
    for r in check_results:
        print(f"Набір {r['set']}: таблиці {'збігаються' if r['tables_equal'] else 'НЕ збігаються'}, "
              f"t* = {format_threshold(r['fast_t'])} / {format_threshold(r['slow_t'])}")
    if not passed:
        raise SystemExit("Перевірки пошуку порога не пройдено - тестову оцінку не виконуємо.")

    threshold_info = {
        "t_star": t_star,
        "t_star_repr": format_threshold(t_star),
        "selected_on": "validation",
        "rule": "fraud if score >= t",
        "cost": "C = 10*FN + FP",
        "validation": {k: int(best[k]) for k in ("TP", "FP", "FN", "TN", "C")},
        "n_candidates": len(table),
        "n_thresholds_with_min_cost": n_min,
        **curve_stats,
        "small_set_checks_passed": passed,
    }
    (out / "threshold.json").write_text(json.dumps(threshold_info, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")

    # ---- 4. Test із зафіксованим t* ----
    y_te = saved["test_y"]
    s_te = saved["test_scores"]
    test_rows = pd.DataFrame(
        [
            rule_row("t* (з validation)", t_star, confusion_at(s_te, y_te, t_star)),
            rule_row("Довідково: поріг 0.5", 0.5, confusion_at(s_te, y_te, 0.5)),
            rule_row("Довідково: усі негативні", np.inf, confusion_at(s_te, y_te, np.inf)),
        ]
    )
    errors = error_examples(saved, t_star)
    errors.to_csv(out / "test_errors.csv", index=False)

    # Довідкові агрегати по всіх помилках test (для обговорення, не для вибору порога).
    pred = s_te >= t_star
    agg = []
    for kind, mask in (("FP", pred & (y_te == 0)), ("FN", ~pred & (y_te == 1)), ("TP", pred & (y_te == 1))):
        amt = saved["test_amount"][mask]
        dist = np.abs(s_te[mask].astype(np.float64) - t_star)
        agg.append({"Група": kind, "Кількість": int(mask.sum()),
                    "Медіана Amount": f"{np.median(amt):.2f}" if mask.any() else "-",
                    "Середнє Amount": f"{np.mean(amt):.2f}" if mask.any() else "-",
                    "Медіана відстані s до t*": f"{np.median(dist):.4g}" if mask.any() else "-"})
    agg = pd.DataFrame(agg)

    print("\nTest (поріг зафіксовано на validation):")
    print(test_rows.to_string(index=False))
    print("\nПерші помилки на test:")
    print(errors.to_string(index=False))
    print(agg.to_string(index=False))

    errors_md = errors.assign(
        score=errors["score"].map(lambda v: f"{v:.6g}"),
        amount=errors["amount"].map(lambda v: f"{v:.2f}"),
        distance_to_threshold=errors["distance_to_threshold"].map(lambda v: f"{v:+.4g}"),
    ).rename(columns={"csv_row": "Рядок CSV", "error": "Тип помилки", "label": "Мітка",
                      "score": "Оцінка s", "amount": "Amount", "distance_to_threshold": "s − t*"})
    report = [
        "# Результати аналізу (згенеровано analysis.py)",
        "",
        "## Поділ",
        "",
        md_table(summary.assign(positive_fraction=summary["positive_fraction"].map(lambda v: f"{v:.6f}"))),
        "",
        "## Validation",
        "",
        f"- ROC-AUC = {curve_stats['roc_auc']:.6f}; average precision = {curve_stats['average_precision']:.6f}",
        f"- Кандидатів: {len(table)}; t* = {format_threshold(t_star)}; порогів із мінімальною C: {n_min}",
        "",
        md_table(rules),
        "",
        "## Перевірки A/B/C",
        "",
        "Пройдено." if passed else "НЕ пройдено.",
        "",
        "## Test",
        "",
        md_table(test_rows),
        "",
        md_table(errors_md),
        "",
        md_table(agg),
        "",
    ]
    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nРезультати збережено: {out.resolve()}")


if __name__ == "__main__":
    main()
