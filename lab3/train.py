"""Команда навчання ЛР3: поділ, перевірки пунктів 1-2, сім запусків, вибір lab3_model.pt.

    uv run python train.py --data /шлях/до/archive

Тут test не завантажується: ImageDataset будується лише з рядків train і validation.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import common as C


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 1. перевірки до навчання
def check_split(data: Path, out: Path) -> tuple[list[dict], dict]:
    rows = C.make_split(data)
    path = out / "split.csv"
    C.write_split(rows, path)
    rows = C.read_split(path)  # далі все лише з файлу
    ctrl = C.split_control(rows)
    table, all_ok = [], True
    for c in C.CLASSES:
        for p in C.PARTS:
            ok = ctrl[c][p] == C.SPLIT_CONTROL[c][p]
            all_ok &= ok
            table.append({"class": c, "part": p, "count": ctrl[c][p][0], "sum": ctrl[c][p][1],
                          "expected": list(C.SPLIT_CONTROL[c][p]), "match": ok})
    sets = {p: {r["file"] for r in rows if r["part"] == p} for p in C.PARTS}
    overlaps = {f"{a}&{b}": len(sets[a] & sets[b])
                for i, a in enumerate(C.PARTS) for b in C.PARTS[i + 1:]}
    unique = len({r["file"] for r in rows}) == len(rows)
    totals = {p: [len(sets[p]), sum(int(Path(f).stem) for f in sets[p])] for p in C.PARTS}
    log("Поділ:", "усі 18 клітинок збігаються" if all_ok else "Є РОЗБІЖНОСТІ", totals,
        "перетини:", overlaps)
    res = {"cells": table, "all_match": all_ok, "overlaps": overlaps, "unique_files": unique,
           "totals": totals, "split_sha256": C.sha256(path)}
    assert all_ok and unique and not any(overlaps.values()), res
    return rows, res


def save_batch_figure(train_ds, out: Path) -> None:
    torch.manual_seed(0)
    x, y = next(iter(C.train_loader(train_ds)))
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9.6))
    for i, ax in enumerate(axes.flat):
        img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"{C.CLASSES[y[i]]} ({int(y[i])})", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Перші 16 зображень навчального пакета після перетворень "
                 f"(тензор {tuple(x.shape)}, обернене нормування)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "figures" / "train_batch.png", dpi=120)
    plt.close(fig)


def initial_loss(regime: str, train_ds, device) -> dict:
    b = C.build(regime, 1, device)
    loader = C.train_loader(train_ds)
    C.set_train_mode(b.model, regime)
    x, y = next(iter(loader))
    feats = {}
    hook = b.model.avgpool.register_forward_hook(lambda m, i, o: feats.update(f=o.flatten(1)))
    with torch.no_grad():
        z = b.model(x.to(device)).float().cpu()
    hook.remove()
    zm = z.mean(0)
    ce = nn.functional.cross_entropy
    return {"L0": ce(z, y).item(), "ln6": math.log(6), "logit_std": z.std().item(),
            "mean_logit_vector": [round(v, 4) for v in zm.tolist()],
            "L_of_mean_logit_vector": ce(zm.expand_as(z), y).item(),
            "within_sample_logit_std": (z - zm).std().item(),
            "feature_mean": feats["f"].mean().item(),
            "feature_l2_norm_mean": feats["f"].norm(dim=1).mean().item(),
            "batch_label_counts": torch.bincount(y, minlength=6).tolist(),
            "batch_pred_counts": torch.bincount(z.argmax(1), minlength=6).tolist()}


# ---------------------------------------------------------------- 2. перевірка режимів
def param_report(regime: str, device) -> dict:
    b = C.build(regime, 1, device)
    m, opt = b.model, b.optimizer
    total = C.numel(m.parameters())
    trainable = C.numel(p for p in m.parameters() if p.requires_grad)
    in_opt = [p for g in opt.param_groups for p in g["params"]]
    frozen_in_opt = sum(1 for p in in_opt if not p.requires_grad)
    ids_opt = {id(p) for p in in_opt}
    ids_tr = {id(p) for p in m.parameters() if p.requires_grad}
    name_of = {id(p): n.split(".")[0] for n, p in m.named_parameters()}
    groups = [{"modules": sorted({name_of[id(p)] for p in g["params"]},
                                 key=C.TOP_MODULES.index),
               "n_params": C.numel(g["params"]), "lr": g["lr"]} for g in opt.param_groups]
    ok = (total == C.TOTAL_PARAMS and trainable == C.EXPECTED_TRAINABLE[regime]
          and frozen_in_opt == 0 and ids_opt == ids_tr and len(in_opt) == len(ids_opt))
    return {"regime": regime, "total": total, "trainable": trainable, "groups": groups,
            "frozen_params_in_optimizer": frozen_in_opt,
            "optimizer_equals_trainable": ids_opt == ids_tr, "match": ok}


BN_STATS = ("running_mean", "running_var")


def five_steps(regime: str, train_ds, device, mode_fn) -> tuple[dict, dict]:
    b = C.build(regime, 1, device)
    m, opt = b.model, b.optimizer
    before = {k: v.detach().clone() for k, v in m.state_dict().items()}
    mode_fn(m)
    loss_fn = nn.CrossEntropyLoss()
    it = iter(C.train_loader(train_ds))
    for _ in range(5):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(m(x), y).backward()
        opt.step()
    after = m.state_dict()
    params = {n for n, _ in m.named_parameters()}
    rows = {}
    for mod in C.TOP_MODULES:
        keys = [k for k in before if k.split(".")[0] == mod]
        pk = [k for k in keys if k in params]
        sk = [k for k in keys if k.rsplit(".", 1)[-1] in BN_STATS]
        nk = [k for k in keys if k.endswith("num_batches_tracked")]
        p_changed = any(not torch.equal(before[k], after[k]) for k in pk)
        s_changed = None if not sk else any(not torch.equal(before[k], after[k]) for k in sk)
        n_changed = None if not nk else any(not torch.equal(before[k], after[k]) for k in nk)
        rows[mod] = {"params_changed": p_changed, "bn_stats_changed": s_changed,
                     "num_batches_tracked_changed": n_changed}
    k = "layer1.0.bn1"
    delta = {s: float((after[f"{k}.{s}"].float() - before[f"{k}.{s}"].float()).abs().max())
             for s in BN_STATS}
    delta["num_batches_tracked"] = [int(before[f"{k}.num_batches_tracked"]),
                                    int(after[f"{k}.num_batches_tracked"])]
    return rows, delta


def expected_change(regime: str, mod: str) -> tuple[bool, bool | None]:
    has_bn = mod not in ("conv1", "fc")
    frozen = mod in C.EVAL_MODULES[regime]
    return (not frozen), (None if not has_bn else not frozen)


def check_changes(regime: str, train_ds, device, buggy=False) -> dict:
    mode_fn = (lambda m: m.train()) if buggy else (lambda m: C.set_train_mode(m, regime))
    rows, delta = five_steps(regime, train_ds, device, mode_fn)
    table, all_ok = [], True
    for mod in C.TOP_MODULES:
        ep, es = expected_change(regime, mod)
        r = rows[mod]
        ok = r["params_changed"] == ep and r["bn_stats_changed"] == es
        all_ok &= ok
        table.append({"module": mod, **r, "expected_params": ep, "expected_bn": es, "match": ok})
    tag = f"{regime}{' (model.train())' if buggy else ''}"
    log(f"Зміни після 5 кроків, {tag}:",
        "; ".join(f"{t['module']}: p={t['params_changed']} bn={t['bn_stats_changed']}"
                  for t in table), "| збіг з очікуваним:", all_ok)
    return {"regime": regime, "buggy_model_train": buggy, "modules": table, "all_match": all_ok,
            "layer1.0.bn1_max_abs_change": delta}


# ---------------------------------------------------------------- 3. запуски
def run_all(train_ds, val_ds, device, out: Path, split_hash: str) -> dict:
    logs = {}
    git = C.git_info()
    env = C.env_info(device)
    for num, regime, seed in C.RUNS:
        name = f"{regime}_seed{seed}"
        log(f"Запуск {num}/7: {name}")
        res = C.train_run(regime, seed, train_ds, val_ds, device, log=log)
        d = out / "runs" / name
        d.mkdir(parents=True, exist_ok=True)
        log_json = {
            "run": num, "name": name, "seed": seed, "regime": regime,
            "config": C.config_dict(regime, seed, res["optimizer"]),
            "git": git, "split_csv_sha256": split_hash, "env": env,
            "history": res["history"], "optimizer_steps": res["steps"],
            "final_train_loss": res["history"][-1]["train_loss"],
            "final_val_macro_f1": res["history"][-1]["val_macro_f1"],
            "train_time_s": res["train_time_s"],
        }
        C.save_json(log_json, d / "run.json")
        logits = res["val_logits"].astype(np.float32)
        assert logits.shape == (len(val_ds), 6)
        np.save(d / "val_logits.npy", logits)
        torch.save({k: v.cpu() for k, v in res["model"].state_dict().items()}, d / "model.pt")
        logs[name] = log_json
        log(f"  час {res['train_time_s']:.1f} с, F = {log_json['final_val_macro_f1']:.4f}")
    return logs


def save_selected(logs: dict, out: Path) -> dict:
    sel = C.select_run(logs)
    name = f"{sel['regime']}_seed{sel['seed']}"
    sd = torch.load(out / "runs" / name / "model.pt", map_location="cpu", weights_only=True)
    torch.save({"state_dict": sd, "regime": sel["regime"], "seed": sel["seed"],
                "classes": C.CLASSES, "img_size": C.IMG_SIZE}, out / "lab3_model.pt")
    log(f"Вибір за validation: {name} (F = {sel['val_macro_f1']:.4f}); середні: {sel['regime_means']}")
    return sel


# ---------------------------------------------------------------- main
def checks_markdown(chk: dict) -> str:
    yn = lambda v: "немає BN" if v is None else ("так" if v else "ні")
    L = ["# Перевірки до навчання (пункти 1-2)", "", "## Поділ", "",
         "| Клас | train | validation | test |", "|---|---|---|---|"]
    cells = {(c["class"], c["part"]): c for c in chk["split"]["cells"]}
    for c in C.CLASSES:
        L.append(f"| {c} | " + " | ".join(
            f"{cells[c, p]['count']} / {cells[c, p]['sum']:,} {'✓' if cells[c, p]['match'] else '✗'}"
            .replace(",", " ") for p in C.PARTS) + " |")
    L.append("| Разом | " + " | ".join(f"{n} / {s:,}".replace(",", " ")
                                       for n, s in chk["split"]["totals"].values()) + " |")
    L += ["", f"Усі 18 клітинок збігаються: {chk['split']['all_match']}. "
              f"Перетини частин: {chk['split']['overlaps']}. SHA-256 split.csv: "
              f"`{chk['split']['split_sha256']}`", "", "## Початкова втрата", "",
          "| Режим | L0 | ln 6 | L0 для середнього вектора логітів | середній вектор логітів | "
          "‖ознаки avgpool‖ | мітки пакета |", "|---|---|---|---|---|---|---|"]
    for r, v in chk["initial_loss"].items():
        L.append(f"| {r} | {v['L0']:.4f} | {v['ln6']:.4f} | {v['L_of_mean_logit_vector']:.4f} | "
                 f"{v['mean_logit_vector']} | {v['feature_l2_norm_mean']:.2f} | "
                 f"{v['batch_label_counts']} |")
    L += ["", "## Параметри", "",
          "| Режим | Усього | requires_grad=True | Групи AdamW (модулі: параметрів, lr) | "
          "Заморожених в оптимізаторі | Збіг |", "|---|---|---|---|---|---|"]
    for p in chk["params"]:
        g = "; ".join(f"{'+'.join(x['modules']) if len(x['modules']) < 7 else 'уся мережа'}: "
                      f"{x['n_params']:_}, lr = {x['lr']:g}".replace("_", " ") for x in p["groups"])
        L.append(f"| {p['regime']} | {p['total']:_} | {p['trainable']:_} | {g} | "
                 f"{p['frozen_params_in_optimizer']} | {'так' if p['match'] else 'ні'} |"
                 .replace("_", " "))
    for key, title in [("changes_frozen", "frozen, set_train_mode"),
                       ("changes_unfreeze", "unfreeze, set_train_mode"),
                       ("bug_frozen", "frozen, model.train() — навмисна помилка")]:
        c = chk[key]
        L += ["", f"## Зміни після 5 кроків: {title}", "",
              "| Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |",
              "|---|---|---|---|"]
        for t in c["modules"]:
            L.append(f"| {t['module']} | {yn(t['params_changed'])} | {yn(t['bn_stats_changed'])} | "
                     f"{yn(t['match'])} |")
        d = c["layer1.0.bn1_max_abs_change"]
        L.append(f"\n`layer1.0.bn1`: max|Δ running_mean| = {d['running_mean']:.6g}, "
                 f"max|Δ running_var| = {d['running_var']:.6g}, num_batches_tracked "
                 f"{d['num_batches_tracked'][0]} → {d['num_batches_tracked'][1]}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path,
                    help="тека розпакованого архіву з seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=C.HERE, help="куди писати результати")
    ap.add_argument("--checks-only", action="store_true", help="лише пункти 1-2")
    args = ap.parse_args()
    out = args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "checks").mkdir(parents=True, exist_ok=True)
    device = C.pick_device()
    log("Пристрій:", C.device_name(device))

    rows, split_res = check_split(args.data, out)
    split_hash = split_res["split_sha256"]
    train_ds = C.ImageDataset(C.split_rows(rows, "train"), args.data, C.train_transform())
    val_ds = C.ImageDataset(C.split_rows(rows, "validation"), args.data, C.eval_transform())
    assert len(train_ds) == 1200 and len(val_ds) == 1498

    save_batch_figure(train_ds, out)
    chk = {"split": split_res, "device": C.device_name(device), "initial_loss": {}}
    for r in ("scratch", "frozen"):
        chk["initial_loss"][r] = v = initial_loss(r, train_ds, device)
        log(f"L0 {r}:", v)
    chk["params"] = [param_report(r, device) for r in C.REGIMES]
    for p in chk["params"]:
        log(f"Параметри {p['regime']}: {p['total']} / {p['trainable']} / {p['groups']} / "
            f"збіг {p['match']}")
    chk["changes_frozen"] = check_changes("frozen", train_ds, device)
    chk["changes_unfreeze"] = check_changes("unfreeze", train_ds, device)
    chk["bug_frozen"] = check_changes("frozen", train_ds, device, buggy=True)
    log("layer1.0.bn1 у досліді з помилкою:", chk["bug_frozen"]["layer1.0.bn1_max_abs_change"])
    C.save_json(chk, out / "checks" / "checks.json")
    (out / "checks" / "checks.md").write_text(checks_markdown(chk))

    ok = (all(p["match"] for p in chk["params"]) and chk["changes_frozen"]["all_match"]
          and chk["changes_unfreeze"]["all_match"])
    assert ok, "перевірки пункту 2 не збіглися з очікуваним — основний експеримент не запускається"
    log("Усі перевірки пунктів 1-2 збіглися з очікуваним.")
    if args.checks_only:
        return
    logs = run_all(train_ds, val_ds, device, out, split_hash)
    save_selected(logs, out)


if __name__ == "__main__":
    main()
