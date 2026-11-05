"""Команда аналізу ЛР3.

    uv run python analyze.py --data /шлях/до/archive

1) перевірка SHA-256 split.csv у семи журналах; 2) таблиця запусків, криві, розклад
ефектів, перевірка вибору — лише зі збережених журналів; 3) одноразова тестова оцінка
lab3_model.pt на CPU, матриця плутанини, п'ять характерних помилок.
Test завантажується тільки тут.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torchvision.models import resnet18

import common as C


def log(*a):
    print(*a, flush=True)


def fmt(x: float, d: int = 4) -> str:
    return f"{x:.{d}f}"


# ---------------------------------------------------------------- журнали
def load_logs(root: Path) -> dict:
    split_hash = C.sha256(root / "split.csv")
    logs, bad = {}, []
    for num, regime, seed in C.RUNS:
        name = f"{regime}_seed{seed}"
        path = root / "runs" / name / "run.json"
        if not path.exists():
            sys.exit(f"ПОМИЛКА: немає журналу {path}")
        lj = C.load_json(path)
        logs[name] = lj
        if lj["split_csv_sha256"] != split_hash:
            bad.append(name)
    hashes = {lj["split_csv_sha256"] for lj in logs.values()}
    if bad or len(hashes) != 1:
        sys.exit(f"ПОМИЛКА: хеш split.csv у журналах {bad or sorted(hashes)} не збігається з "
                 f"поточним {split_hash}. Запуски на різних поділах порівнювати не можна: "
                 f"перенавчіть усі сім.")
    log(f"SHA-256 split.csv однаковий у 7 журналах і збігається з поточним: {split_hash}")
    return logs, split_hash


def check_val_logits(root: Path, logs: dict, val_labels: list[int]) -> None:
    for name, lj in logs.items():
        z = np.load(root / "runs" / name / "val_logits.npy")
        assert z.shape == (1498, 6), (name, z.shape)
        f = C.macro_f1(val_labels, z)
        assert abs(f - lj["final_val_macro_f1"]) < 1e-12, (name, f, lj["final_val_macro_f1"])
    log("val_logits.npy усіх запусків відтворюють записаний macro-F1 епохи 8.")


def runs_table(logs: dict) -> str:
    L = ["| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | "
         "Час навчання, с |", "|---|---|---|---|---|---|"]
    for num, regime, seed in C.RUNS:
        lj = logs[f"{regime}_seed{seed}"]
        L.append(f"| {num} | {regime} | {seed} | {fmt(lj['final_train_loss'])} | "
                 f"{fmt(lj['final_val_macro_f1'])} | {lj['train_time_s']:.1f} |")
    return "\n".join(L)


def plot_curves(logs: dict, path: Path) -> None:
    colors = {"scratch": "#444444", "frozen": "#1f77b4", "unfreeze": "#d62728"}
    styles = {1: "-", 2: "--", 3: ":"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for num, regime, seed in C.RUNS:
        h = logs[f"{regime}_seed{seed}"]["history"]
        ep = [e["epoch"] for e in h]
        kw = dict(color=colors[regime], ls=styles[seed], marker="o", ms=3,
                  label=f"#{num} {regime}, seed {seed}")
        axes[0].plot(ep, [e["train_loss"] for e in h], **kw)
        axes[1].plot(ep, [e["val_macro_f1"] for e in h], **kw)
    axes[0].set(title="Втрата на train", xlabel="епоха", ylabel="CrossEntropy (середнє за епоху)")
    axes[1].set(title="macro-F1 на validation", xlabel="епоха", ylabel="macro-F1", ylim=(0, 1))
    axes[1].axhline(1 / 6, color="gray", lw=0.8, ls="-.", label="1/6 (вгадування)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(range(1, C.EPOCHS + 1))
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def effects(logs: dict) -> dict:
    F = {name: lj["final_val_macro_f1"] for name, lj in logs.items()}
    fr = [F[f"frozen_seed{s}"] for s in (1, 2, 3)]
    un = [F[f"unfreeze_seed{s}"] for s in (1, 2, 3)]
    e = {"F_scratch_1": F["scratch_seed1"],
         "F_frozen_mean": float(np.mean(fr)), "R_frozen": float(max(fr) - min(fr)),
         "F_unfreeze_mean": float(np.mean(un)), "R_unfreeze": float(max(un) - min(un))}
    e["R"] = max(e["R_frozen"], e["R_unfreeze"])
    e["delta_pre"] = e["F_frozen_mean"] - e["F_scratch_1"]
    e["delta_unf"] = e["F_unfreeze_mean"] - e["F_frozen_mean"]
    e["delta_total"] = e["delta_pre"] + e["delta_unf"]
    for k in ("delta_pre", "delta_unf"):
        e[f"{k}_verdict"] = ("вартий перевірки, але не доведений (∣Δ∣ > R)" if abs(e[k]) > e["R"]
                             else "не відрізняється від розкиду (∣Δ∣ ≤ R)")
    return e


def effects_table(e: dict) -> str:
    return "\n".join([
        "| Величина | Значення |", "|---|---|",
        f"| $F_{{\\text{{scratch}},1}}$ | {fmt(e['F_scratch_1'])} |",
        f"| $\\bar F_{{\\text{{frozen}}}},\\ R_{{\\text{{frozen}}}}$ | {fmt(e['F_frozen_mean'])}, {fmt(e['R_frozen'])} |",
        f"| $\\bar F_{{\\text{{unfreeze}}}},\\ R_{{\\text{{unfreeze}}}}$ | {fmt(e['F_unfreeze_mean'])}, {fmt(e['R_unfreeze'])} |",
        f"| $R$ | {fmt(e['R'])} |",
        f"| $\\Delta_{{\\text{{pre}}}}$ | {e['delta_pre']:+.4f} — {e['delta_pre_verdict']} |",
        f"| $\\Delta_{{\\text{{unf}}}}$ | {e['delta_unf']:+.4f} — {e['delta_unf_verdict']} |",
        f"| $\\Delta_{{\\text{{pre}}}} + \\Delta_{{\\text{{unf}}}}$ | {e['delta_total']:+.4f} |",
    ])


# ---------------------------------------------------------------- модель
def load_model(path: Path) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, len(C.CLASSES))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def check_selection(root: Path, logs: dict, ck: dict, model: nn.Module, val_ds) -> dict:
    sel = C.select_run(logs)
    ok_meta = (ck["regime"] == sel["regime"] and ck["seed"] == sel["seed"]
               and list(ck["classes"]) == C.CLASSES and ck["img_size"] == C.IMG_SIZE)
    if not ok_meta:
        sys.exit(f"ПОМИЛКА: lab3_model.pt ({ck['regime']}, seed {ck['seed']}) не відповідає "
                 f"правилу вибору ({sel['regime']}, seed {sel['seed']}).")
    # Додатково: ваги дають на validation той самий результат, що записано в журналі.
    name = f"{sel['regime']}_seed{sel['seed']}"
    z_cpu = C.predict_logits(model, C.eval_loader(val_ds), torch.device("cpu"))
    z_log = np.load(root / "runs" / name / "val_logits.npy")
    agree = float((z_cpu.argmax(1) == z_log.argmax(1)).mean())
    f_cpu = C.macro_f1(val_ds.labels, z_cpu)
    sel.update({"name": name, "val_f1_cpu_recomputed": f_cpu, "val_argmax_agreement": agree,
                "max_abs_logit_diff_cpu_vs_log": float(np.abs(z_cpu - z_log).max())})
    if agree < 0.995:
        sys.exit(f"ПОМИЛКА: ваги lab3_model.pt не відтворюють val_logits.npy запуску {name}")
    log(f"Вибір підтверджено: {name}; F = {sel['val_macro_f1']:.4f} (журнал), "
        f"{f_cpu:.4f} (перерахунок на CPU), збіг argmax {agree:.4f}")
    return sel


# ---------------------------------------------------------------- test
def five_errors(files, y, pred, prob, cm) -> list[dict]:
    pairs = [(int(cm[i, j]), i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    by_pair = {(i, j): sorted((files[k] for k in np.flatnonzero((y == i) & (pred == j))))
               for _, i, j in pairs}
    idx = {f: k for k, f in enumerate(files)}
    picked, rank = [], 0
    while len(picked) < 5 and any(len(v) > rank for v in by_pair.values()):
        for n, i, j in pairs:
            if len(picked) == 5:
                break
            if len(by_pair[i, j]) > rank:
                f = by_pair[i, j][rank]
                picked.append({"file": f, "true": C.CLASSES[i], "pred": C.CLASSES[j],
                               "prob_pred": float(prob[idx[f], j]), "count": n})
        rank += 1
    return picked


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(6), C.CLASSES, rotation=35, ha="right")
    ax.set_yticks(range(6), C.CLASSES)
    ax.set(xlabel="передбачений клас", ylabel="справжній клас", title=title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_errors(errs: list[dict], test_ds, path: Path) -> None:
    pos = {f: k for k, f in enumerate(test_ds.files)}
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, e in zip(axes, errs):
        ax.imshow(test_ds.image(pos[e["file"]]))
        ax.set_title(f"{e['file']}\nсправжній: {e['true']}\nпередбачено: {e['pred']} "
                     f"(p={e['prob_pred']:.2f})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path,
                    help="тека розпакованого архіву з seg_train/ і seg_test/")
    ap.add_argument("--root", type=Path, default=C.HERE,
                    help="тека з split.csv, runs/ і lab3_model.pt")
    ap.add_argument("--out", type=Path, default=None, help="тека результатів (за замовч. <root>/analysis)")
    args = ap.parse_args()
    root = args.root
    out = args.out or root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    cpu = torch.device("cpu")

    # 1. хеш поділу
    logs, split_hash = load_logs(root)
    rows = C.read_split(root / "split.csv")
    val_ds = C.ImageDataset(C.split_rows(rows, "validation"), args.data, C.eval_transform())
    check_val_logits(root, logs, val_ds.labels)

    # 2. таблиця, криві, ефекти, вибір
    table = runs_table(logs)
    plot_curves(logs, root / "figures" / "curves.png")
    eff = effects(logs)
    model, ck = load_model(root / "lab3_model.pt")
    sel = check_selection(root, logs, ck, model, val_ds)
    log("\n" + table + "\n\n" + effects_table(eff))

    # 3. test — один раз, на CPU
    test_ds = C.ImageDataset(C.split_rows(rows, "test"), args.data, C.eval_transform())
    assert len(test_ds) == 1502
    z = C.predict_logits(model, C.eval_loader(test_ds), cpu)
    np.save(out / "test_logits.npy", z.astype(np.float32))
    y = np.array(test_ds.labels)
    pred = z.argmax(1)
    prob = torch.softmax(torch.from_numpy(z), 1).numpy()
    f1 = C.macro_f1(y, z)
    acc = float(accuracy_score(y, pred))
    se = float(np.sqrt(acc * (1 - acc) / len(y)))
    cm = confusion_matrix(y, pred, labels=list(range(6)))
    plot_confusion(cm, root / "figures" / "confusion_test.png",
                   f"Test, {sel['name']}: macro-F1 = {f1:.4f}")
    errs = five_errors(test_ds.files, y, pred, prob, cm)
    plot_errors(errs, test_ds, root / "figures" / "five_errors.png")

    cm_md = ["| справжній \\ передбачений | " + " | ".join(C.CLASSES) + " | разом |",
             "|---|" + "---|" * 7]
    for i, c in enumerate(C.CLASSES):
        cm_md.append(f"| **{c}** | " + " | ".join(
            f"**{v}**" if i == j else str(v) for j, v in enumerate(cm[i])) + f" | {cm[i].sum()} |")
    err_md = ["| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | "
              "Кількість таких помилок у test |", "|---|---|---|---|---|"]
    err_md += [f"| {e['file']} | {e['true']} | {e['pred']} | {e['prob_pred']:.4f} | {e['count']} |"
               for e in errs]
    gap = f1 - sel["val_macro_f1"]
    test_md = "\n".join([
        f"Вибраний запуск: **{sel['name']}**, $F_{{r,s}}$ = {fmt(sel['val_macro_f1'])} на validation.",
        "",
        "| Показник | Значення |", "|---|---|",
        f"| macro-F1 test | {fmt(f1)} |",
        f"| accuracy test, $p$ | {fmt(acc)} ({int((pred == y).sum())} / {len(y)}) |",
        f"| $\\sqrt{{p(1-p)/n}}$, $n={len(y)}$ | {fmt(se)} |",
        f"| macro-F1 test − $F_{{r,s}}$ вибраного запуску | {gap:+.4f} |",
    ])

    summary = "\n\n".join([
        "# Результати команди аналізу",
        f"SHA-256 split.csv (однаковий у 7 журналах і в поточному файлі): `{split_hash}`",
        "## Сім запусків", table, "![криві](../figures/curves.png)",
        "## Розклад ефектів", effects_table(eff),
        "## Вибір і тестова оцінка",
        f"Середні за режимами: " + ", ".join(f"{r}: {fmt(v)}" for r, v in sel["regime_means"].items())
        + f". Перерахунок вибраних ваг на CPU на validation: macro-F1 {fmt(sel['val_f1_cpu_recomputed'])}, "
        f"збіг argmax із журналом {sel['val_argmax_agreement']:.4f}.",
        test_md, "### Матриця плутанини (test)", "\n".join(cm_md),
        "![матриця плутанини](../figures/confusion_test.png)",
        "### П'ять характерних помилок", "\n".join(err_md),
        "![п'ять помилок](../figures/five_errors.png)",
    ]) + "\n"
    (out / "summary.md").write_text(summary)
    C.save_json({"split_sha256": split_hash, "effects": eff, "selection": sel,
                 "test": {"macro_f1": f1, "accuracy": acc, "se": se, "n": len(y),
                          "confusion": cm.tolist(), "test_minus_val": gap},
                 "five_errors": errs}, out / "summary.json")
    log("\n" + test_md + "\n\n" + "\n".join(cm_md) + "\n\n" + "\n".join(err_md))
    log(f"\nРезультати: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
