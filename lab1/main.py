"""Точка входу.

    uv run python main.py          # правильна реалізація (за замовчуванням)
    uv run python main.py --bug    # дослід із навмисною помилкою
    uv run python main.py --all    # обидва режими + порівняння

Результати друкуються в консоль і зберігаються в results/*.md.
"""

import argparse
from pathlib import Path

import numpy as np

from checks import PARAM_NAMES, compare_with_torch, numerical_check
from data import load_data
from mlp_numpy import NumpyMLP, init_params

RESULTS_DIR = Path(__file__).parent / "results"


def ok(flag: bool) -> str:
    return "так" if flag else "**ні**"


def run(buggy: bool) -> tuple[str, dict]:
    data = load_data(seed=0)
    X, y = data.X_train, data.y_train
    params = init_params(seed=0)

    lines = []
    title = "з навмисною помилкою (без ділення на N)" if buggy else "правильна"
    lines.append(f"# Результати: реалізація {title}\n")
    lines.append(f"Навчальна вибірка: X {X.shape}, y {y.shape}, "
                 f"класи {np.bincount(y).tolist()}; тест: {data.X_test.shape}.\n")

    # --- Звірка з PyTorch ---
    model = NumpyMLP(params, buggy=buggy)
    cmp = compare_with_torch(model, X, y)
    lines.append("## Звірка з PyTorch\n")
    lines.append(f"- Втрата NumPy:   `{cmp['loss_np']:.17g}`")
    lines.append(f"- Втрата PyTorch: `{cmp['loss_pt']:.17g}`\n")
    lines.append("| Величина | Максимальна абсолютна різниця NumPy / PyTorch "
                 "| Перевірку пройдено |")
    lines.append("|---|---|---|")
    for r in cmp["rows"]:
        lines.append(f"| {r['name']} | {r['max_abs_diff']:.3e} | {ok(r['passed'])} |")
    lines.append("\nКритерій: |a − b| ≤ 1e-12 для кожного елемента, усі значення скінченні.\n")

    # --- Чисельна перевірка ---
    model = NumpyMLP(params, buggy=buggy)
    rows = numerical_check(model, X, y)
    lines.append("## Чисельне диференціювання (ε = 1e-6)\n")
    lines.append("| Параметр | Градієнт backward() | Чисельна похідна "
                 "| Абсолютна різниця | Перевірку пройдено |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['param']} | {r['g_manual']:.12e} | {r['g_num']:.12e} "
                     f"| {r['abs_diff']:.3e} | {ok(r['passed'])} |")
    lines.append("\nКритерій: |g_num − g_manual| ≤ 1e-7.\n")

    summary = {
        "torch_passed": [r["passed"] for r in cmp["rows"]],
        "numeric_passed": [r["passed"] for r in rows],
        "loss": cmp["loss_np"],
        "grads": cmp["grads_np"],
    }
    return "\n".join(lines), summary


def bug_vs_correct(correct: dict, buggy: dict) -> str:
    lines = ["# Порівняння: помилкова vs правильна реалізація\n",
             f"- Втрата правильна: `{correct['loss']:.17g}`",
             f"- Втрата з помилкою: `{buggy['loss']:.17g}`",
             f"- Різниця: `{abs(correct['loss'] - buggy['loss']):.3e}`\n",
             "| Градієнт | К-сть ненульових | Відношення bug / correct (min … max) |",
             "|---|---|---|"]
    for name in PARAM_NAMES:
        g_ok, g_bug = correct["grads"][name], buggy["grads"][name]
        nz = g_ok != 0
        ratio = g_bug[nz] / g_ok[nz]
        lines.append(f"| {name} | {nz.sum()} / {g_ok.size} "
                     f"| {ratio.min():.10g} … {ratio.max():.10g} |")
    lines.append("\nОчікуване відношення: N = 105.\n")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ЛР1: перевірка backprop")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--bug", action="store_true",
                   help="увімкнути навмисну помилку (без ділення на N у dZ2)")
    g.add_argument("--all", action="store_true",
                   help="запустити обидві реалізації та порівняти їх")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    modes = [False, True] if args.all else [args.bug]
    summaries = {}
    for buggy in modes:
        report, summary = run(buggy)
        summaries[buggy] = summary
        fname = RESULTS_DIR / ("bug.md" if buggy else "correct.md")
        fname.write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"[збережено: {fname.relative_to(Path(__file__).parent)}]\n")

    if args.all:
        report = bug_vs_correct(summaries[False], summaries[True])
        fname = RESULTS_DIR / "bug_vs_correct.md"
        fname.write_text(report + "\n", encoding="utf-8")
        print(report)
        print(f"[збережено: {fname.relative_to(Path(__file__).parent)}]")


if __name__ == "__main__":
    main()
