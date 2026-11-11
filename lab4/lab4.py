"""Команда ЛР4: FP32 проти INT8.

    uv run python lab4.py --model ../lab3/lab3_model.pt --split ../lab3/split.csv --data /шлях/до/archive

Пункти 1–5 по черзі: контроль поділу й калібрувальної вибірки; експорт lab4_fp32.onnx і
перевірка відповідності PyTorch; quant_pre_process і quantize_static (per-channel →
lab4_int8.onnx, per-tensor → lab4_int8_per_tensor.onnx); якість на test із парним
порівнянням; діапазони каналів ваг Conv; затримка за протоколом пункту 5; після нього —
окремий діагностичний замір тим самим протоколом із вимкненими ядрами Arm KleidiAI.
Пише lab4_results.json, results/summary.md, results/test_logits_*.npy, figures/*.png.

`--ranges-only` виконує лише контроль, експорт і діапазони каналів (для прогнозу
пункту 4 до запуску per-tensor квантизації).
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import torch
from onnx import numpy_helper
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quant_pre_process, quantize_static)

import common as C


def log(*a):
    print(*a, flush=True)


def fail(msg: str):
    sys.exit(f"ПОМИЛКА: {msg}")


def f4(x: float) -> str:
    return f"{x:.4f}"


# ---------------------------------------------------------------- контроль
def check_split(rows: list[dict], split_path: Path) -> dict:
    out = {"sha256": C.sha256(split_path), "lab3_sha256": C.LAB3_SPLIT_SHA256}
    out["sha256_matches_lab3"] = out["sha256"] == C.LAB3_SPLIT_SHA256
    for part, (n_exp, s_exp) in C.SPLIT_CONTROL.items():
        n, s = C.count_and_sum(C.part_rows(rows, part))
        out[part] = {"n": n, "name_sum": s, "expected_n": n_exp, "expected_name_sum": s_exp,
                     "ok": (n, s) == (n_exp, s_exp), "per_class": C.per_class(C.part_rows(rows, part))}
        log(f"Поділ, {part}: {n} зображень, сума імен {s:,} — "
            f"{'збігається' if out[part]['ok'] else 'НЕ збігається'} з контролем ({n_exp}, {s_exp:,})")
        if not out[part]["ok"]:
            fail(f"поділ {part} не збігається з контрольним рядком «Разом» ЛР3")
    log(f"SHA-256 split.csv: {out['sha256']} "
        f"({'як у ЛР3' if out['sha256_matches_lab3'] else 'ВІДРІЗНЯЄТЬСЯ від ЛР3'})")
    return out


def check_calibration(calib: list[dict]) -> dict:
    n, s = C.count_and_sum(calib)
    pc = C.per_class(calib)
    exp = C.CALIB_CONTROL
    ok = n == exp["n"] and s == exp["name_sum"] and pc == exp["per_class"]
    log(f"Калібрувальна вибірка: {n} зображень, сума імен {s:,}, за класами {pc} — "
        f"{'збігається' if ok else 'НЕ збігається'} з контролем")
    if not ok:
        fail("калібрувальна вибірка не збігається з контролем")
    return {"n": n, "name_sum": s, "per_class": pc, "ok": ok, "expected": exp,
            "files": [r["file"] for r in calib],
            "procedure": "train rows sorted by file name as strings; "
                         "np.random.default_rng(0).permutation(1200)[:200]"}


# ---------------------------------------------------------------- експорт і діапазони
def export(model, img_size: int, path: Path) -> float:
    dummy = torch.zeros(1, 3, img_size, img_size)
    with warnings.catch_warnings():
        # torch 2.8 попереджає, що з 2.9 за замовчуванням буде dynamo=True; умова фіксує dynamo=False
        warnings.simplefilter("ignore", DeprecationWarning)
        t0 = time.perf_counter()
        torch.onnx.export(model, dummy, str(path), dynamo=False, opset_version=17,
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
        return time.perf_counter() - t0


def op_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in onnx.load(str(path)).graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return dict(sorted(counts.items()))


def channel_ranges(path: Path) -> dict:
    """α_c = max|w| за вихідним каналом; r = max_c α_c / median_c α_c для кожного Conv."""
    g = onnx.load(str(path)).graph
    inits = {t.name: t for t in g.initializer}
    convs = []
    for node in g.node:
        if node.op_type != "Conv":
            continue
        w = numpy_helper.to_array(inits[node.input[1]])
        assert w.ndim == 4, (node.name, w.shape)
        alpha = np.abs(w).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        convs.append({"name": node.name, "weight": node.input[1], "shape": list(w.shape),
                      "alpha_max": float(alpha.max()), "alpha_median": float(np.median(alpha)),
                      "alpha_min": float(alpha.min()), "r": r, "log2_r": float(np.log2(r))})
    rs = [c["r"] for c in convs]
    med = float(np.median(rs))
    top3 = sorted(convs, key=lambda c: -c["r"])[:3]
    return {"n_conv": len(convs), "convs": convs, "top3": [c["name"] for c in top3],
            "median_r": med, "median_r_bits": float(np.log2(med))}


def ranges_table(cr: dict) -> str:
    L = ["| # | Згортка | Форма ваг | $\\max_c\\alpha_c$ | $\\mathrm{median}_c\\,\\alpha_c$ | $r$ | $\\log_2 r$, біт |",
         "|---|---|---|---|---|---|---|"]
    top = set(cr["top3"])
    for i, c in enumerate(cr["convs"], 1):
        name = f"**`{c['name']}`**" if c["name"] in top else f"`{c['name']}`"
        L.append(f"| {i} | {name} | {'×'.join(map(str, c['shape']))} | {c['alpha_max']:.4f} | "
                 f"{c['alpha_median']:.4f} | {c['r']:.2f} | {c['log2_r']:.2f} |")
    return "\n".join(L)


# ---------------------------------------------------------------- квантизація
class Reader(CalibrationDataReader):
    """Калібрувальні зображення по одному, форма 1×3×H×W."""

    def __init__(self, x: np.ndarray):
        self.x = x
        self.i = 0

    def get_next(self):
        if self.i >= len(self.x):
            return None
        item = {"input": self.x[self.i:self.i + 1]}
        self.i += 1
        return item

    def rewind(self):
        self.i = 0


def quantize(prep: Path, out: Path, calib_x: np.ndarray, per_channel: bool) -> float:
    reader = Reader(calib_x)
    t0 = time.perf_counter()
    quantize_static(str(prep), str(out), reader,
                    quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8,
                    per_channel=per_channel,
                    calibrate_method=CalibrationMethod.MinMax)
    dt = time.perf_counter() - t0
    assert reader.i == len(calib_x), reader.i
    return dt


# ---------------------------------------------------------------- затримка
def latency(sessions: dict, x0: np.ndarray) -> dict:
    """Протокол пункту 5: 3 раунди; у кожному FP32, потім INT8; 20 прогрівів, 100 вимірів.
    Вимір — лише один sess.run між двома зчитуваннями time.perf_counter()."""
    raw = {k: [] for k in sessions}
    feeds = {k: {s.get_inputs()[0].name: x0} for k, s in sessions.items()}
    perf = time.perf_counter
    for rnd in range(C.LAT_ROUNDS):
        for k, sess in sessions.items():
            feed = feeds[k]
            for _ in range(C.LAT_WARMUP):
                sess.run(None, feed)
            ts = []
            for _ in range(C.LAT_MEASURE):
                t0 = perf()
                sess.run(None, feed)
                t1 = perf()
                ts.append(t1 - t0)
            raw[k].append([t * 1e3 for t in ts])
            log(f"  раунд {rnd + 1}, {k}: p50 {np.percentile(ts, 50) * 1e3:.3f} мс")
    out = {}
    for k, rounds in raw.items():
        allv = np.concatenate(rounds)
        rp50 = [float(np.percentile(r, 50)) for r in rounds]
        out[k] = {"p50_ms": float(np.percentile(allv, 50)), "p95_ms": float(np.percentile(allv, 95)),
                  "mean_ms": float(allv.mean()), "min_ms": float(allv.min()), "max_ms": float(allv.max()),
                  "p50_by_round_ms": rp50, "p95_by_round_ms": [float(np.percentile(r, 95)) for r in rounds],
                  "p50_round_interval_ms": [min(rp50), max(rp50)], "raw_ms": rounds}
    return out


# ---------------------------------------------------------------- графіки
def plot_latency(lat: dict, path: Path) -> None:
    colors = {"FP32": "#1f77b4", "INT8": "#d62728"}
    fig, axes = plt.subplots(1, len(lat), figsize=(12, 4))
    for ax, (k, v) in zip(axes, lat.items()):
        n = len(v["raw_ms"][0])
        for r, vals in enumerate(v["raw_ms"]):
            x = np.arange(n) + r * (n + 10)
            ax.plot(x, vals, ".", ms=3, color=colors[k])
            ax.hlines(v["p50_by_round_ms"][r], x[0], x[-1], color="black", lw=1.2,
                      label="p50 раунду" if r == 0 else None)
        ax.axhline(v["p95_ms"], color="gray", ls="--", lw=1, label=f"p95 за 300 = {v['p95_ms']:.2f} мс")
        ax.set_xticks([n / 2 + r * (n + 10) for r in range(len(v["raw_ms"]))],
                      [f"раунд {r + 1}" for r in range(len(v["raw_ms"]))])
        ax.set(title=f"{k}: p50 = {v['p50_ms']:.2f} мс", ylabel="затримка sess.run, мс")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_ranges(cr: dict, path: Path) -> None:
    names = [c["name"].replace("/Conv", "").lstrip("/") for c in cr["convs"]]
    rs = [c["r"] for c in cr["convs"]]
    top = set(cr["top3"])
    colors = ["#d62728" if c["name"] in top else "#1f77b4" for c in cr["convs"]]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(rs)), rs, color=colors)
    ax.axhline(cr["median_r"], color="gray", ls="--", lw=1,
               label=f"медіана r = {cr['median_r']:.2f} ({cr['median_r_bits']:.2f} біта)")
    ax.set_xticks(range(len(rs)), names, rotation=60, ha="right", fontsize=8)
    ax.set(ylabel="r = max α_c / median α_c", title="Розкид діапазонів вихідних каналів ваг Conv")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- головна
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=C.HERE.parent / "lab3" / "lab3_model.pt",
                    help="lab3_model.pt із ЛР3")
    ap.add_argument("--split", type=Path, default=C.HERE.parent / "lab3" / "split.csv",
                    help="split.csv із ЛР3")
    ap.add_argument("--data", type=Path, required=True,
                    help="тека розпакованого архіву з seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=C.HERE, help="тека для .onnx і результатів")
    ap.add_argument("--ranges-only", action="store_true",
                    help="лише контроль, експорт і діапазони каналів (для прогнозу пункту 4)")
    args = ap.parse_args()
    out = args.out
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    fp32_path, prep_path = out / "lab4_fp32.onnx", out / "lab4_fp32_prep.onnx"
    int8_path, pt_path = out / "lab4_int8.onnx", out / "lab4_int8_per_tensor.onnx"
    env = C.env_info()
    log(f"Середовище: {env['cpu']}, {env['os']}, Python {env['python']}, PyTorch {env['torch']}, "
        f"ONNX {env['onnx']}, ONNX Runtime {env['onnxruntime']}; живлення: {env['power_source']}")

    # ---- дані, модель, контроль
    rows = C.read_split(args.split)
    split = check_split(rows, args.split)
    model, ck = C.load_torch_model(args.model)
    img = int(ck["img_size"])
    meta = {"regime": ck["regime"], "seed": int(ck["seed"]), "classes": list(ck["classes"]),
            "img_size": img, "sha256": C.sha256(args.model)}
    if meta["classes"] != C.CLASSES:
        fail(f"порядок класів у моделі {meta['classes']} не збігається з {C.CLASSES}")
    log(f"Модель: {meta['regime']}, seed {meta['seed']}, img_size {img}, класи {meta['classes']}")
    calib_rows = C.calibration_rows(rows)
    calib = check_calibration(calib_rows)
    test_rows = C.part_rows(rows, "test")
    y = C.labels(test_rows)

    # ---- 1. експорт
    t_export = export(model, img, fp32_path)
    ops = op_counts(fp32_path)
    log(f"Експорт lab4_fp32.onnx: {t_export:.2f} с; вузлів Conv {ops.get('Conv', 0)}, "
        f"BatchNormalization {ops.get('BatchNormalization', 0)}")
    cr = channel_ranges(fp32_path)
    log("\nДіапазони каналів ваг Conv:\n" + ranges_table(cr))
    log(f"Три найбільші r: " + ", ".join(
        f"{n} ({next(c['r'] for c in cr['convs'] if c['name'] == n):.2f})" for n in cr["top3"])
        + f"; медіана r = {cr['median_r']:.2f}, log2 = {cr['median_r_bits']:.2f} біта")
    if args.ranges_only:
        return

    t0 = time.perf_counter()
    x_test = C.load_images(test_rows, args.data, img)
    log(f"\nTest: {len(x_test)} зображень завантажено за {time.perf_counter() - t0:.1f} с")
    with torch.no_grad():
        z_pt = np.concatenate([model(torch.from_numpy(x_test[i:i + C.EVAL_BS])).numpy()
                               for i in range(0, len(x_test), C.EVAL_BS)])
    sess_fp32 = C.make_session(fp32_path)
    z_fp32 = C.onnx_logits(sess_fp32, x_test)
    pred_pt, pred_fp32 = z_pt.argmax(1), z_fp32.argmax(1)
    exp_chk = {"max_abs_logit_diff": float(np.abs(z_fp32 - z_pt).max()),
               "argmax_agreement": float((pred_pt == pred_fp32).mean()),
               "macro_f1_pytorch": C.macro_f1(y, pred_pt), "macro_f1_onnx_fp32": C.macro_f1(y, pred_fp32),
               "n_conv": ops.get("Conv", 0), "n_batchnorm": ops.get("BatchNormalization", 0),
               "op_counts": ops, "control": C.EXPORT_CONTROL}
    exp_chk["ok"] = (exp_chk["max_abs_logit_diff"] < 1e-3 and exp_chk["argmax_agreement"] == 1.0
                     and exp_chk["macro_f1_pytorch"] == exp_chk["macro_f1_onnx_fp32"]
                     and exp_chk["n_conv"] == 20 and exp_chk["n_batchnorm"] == 0)
    log(f"Перевірка експорту: max|z_ONNX − z_PyTorch| = {exp_chk['max_abs_logit_diff']:.3e}, "
        f"збіг argmax {exp_chk['argmax_agreement']:.4f}, macro-F1 PyTorch {exp_chk['macro_f1_pytorch']:.4f}, "
        f"ONNX FP32 {exp_chk['macro_f1_onnx_fp32']:.4f}")
    if not exp_chk["ok"]:
        fail("перевірка експорту не пройшла контроль")

    # ---- 2. калібрування й квантизація
    t0 = time.perf_counter()
    x_calib = C.load_images(calib_rows, args.data, img)
    t_calib_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)
    t_prep = time.perf_counter() - t0
    t_q_pc = quantize(prep_path, int8_path, x_calib, per_channel=True)
    log(f"\nquant_pre_process: {t_prep:.2f} с; quantize_static per-channel (з калібруванням на "
        f"{len(x_calib)} зображеннях): {t_q_pc:.2f} с")

    # ---- 3. якість
    sess_int8 = C.make_session(int8_path)
    z_int8 = C.onnx_logits(sess_int8, x_test)
    q = {"FP32": C.quality_row(y, pred_fp32, pred_fp32),
         "INT8": C.quality_row(y, z_int8.argmax(1), pred_fp32)}
    p = q["FP32"]["accuracy"]
    se_single = float(np.sqrt(p * (1 - p) / len(y)))

    # ---- 4. per-tensor
    t_q_pt = quantize(prep_path, pt_path, x_calib, per_channel=False)
    sess_pt = C.make_session(pt_path)
    z_ptq = C.onnx_logits(sess_pt, x_test)
    q["INT8 per-tensor"] = C.quality_row(y, z_ptq.argmax(1), pred_fp32)
    log(f"quantize_static per-tensor: {t_q_pt:.2f} с")
    for name, z in (("fp32", z_fp32), ("int8", z_int8), ("int8_per_tensor", z_ptq), ("pytorch", z_pt)):
        np.save(out / "results" / f"test_logits_{name}.npy", z.astype(np.float32))

    q_md = ["| Варіант | macro-F1 | accuracy | Згода з FP32 | $b$ | $c$ | $\\Delta_{\\text{acc}} \\pm \\mathrm{SE}$ | $2\\sqrt{b+c}$ | Вердикт |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, v in q.items():
        d = "—" if k == "FP32" else f"{v['delta_acc']:+.4f} ± {v['se_delta_acc']:.4f}"
        t = "—" if k == "FP32" else f"{v['two_sqrt_b_plus_c']:.2f}"
        verdict = "—" if k == "FP32" else v["verdict"]
        q_md.append(f"| {k} | {f4(v['macro_f1'])} | {f4(v['accuracy'])} ({v['n_correct']}/{len(y)}) | "
                    f"{f4(v['agreement'])} | {v['b']} | {v['c']} | {d} | {t} | {verdict} |")
    q_md = "\n".join(q_md)
    log("\n" + q_md)
    log(f"SE однієї моделі sqrt(p(1-p)/n) = {se_single:.4f} (p = {p:.4f}); "
        f"SE(Δacc) INT8 = {q['INT8']['se_delta_acc']:.4f}")

    # ---- 5. затримка
    x0 = np.ascontiguousarray(x_test[0:1])
    assert x0.shape == (1, 3, img, img) and x0.dtype == np.float32
    log(f"\nЗатримка: вхід — перше зображення test ({test_rows[0]['file']}), "
        f"{C.LAT_ROUNDS} раунди × ({C.LAT_WARMUP} прогрівів + {C.LAT_MEASURE} вимірів)")
    env["power_source_at_latency"] = C.power_source()
    lat = latency({"FP32": sess_fp32, "INT8": sess_int8}, x0)
    sizes = {k: p_.stat().st_size for k, p_ in (("FP32", fp32_path), ("INT8", int8_path),
                                                ("INT8 per-tensor", pt_path), ("FP32 prep", prep_path))}
    a, b_ = lat["FP32"]["p50_round_interval_ms"], lat["INT8"]["p50_round_interval_ms"]
    overlap = not (a[1] < b_[0] or b_[1] < a[0])
    summ = {"speedup_p50": lat["FP32"]["p50_ms"] / lat["INT8"]["p50_ms"],
            "speedup_p95": lat["FP32"]["p95_ms"] / lat["INT8"]["p95_ms"],
            "size_ratio": sizes["FP32"] / sizes["INT8"],
            "p50_round_intervals_overlap": overlap,
            "latency_change_measured": not overlap,
            "input_file": test_rows[0]["file"]}
    lat_md = ["| Варіант | Файл, МБ | p50, мс | p95, мс | p50 за раундами, мс |", "|---|---|---|---|---|"]
    for k in ("FP32", "INT8"):
        v = lat[k]
        lat_md.append(f"| {k} | {sizes[k] / 1e6:.2f} | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | "
                      + " / ".join(f"{x:.2f}" for x in v["p50_by_round_ms"]) + " |")
    lat_md = "\n".join(lat_md)
    log("\n" + lat_md)
    log(f"Прискорення p50 {summ['speedup_p50']:.2f}, p95 {summ['speedup_p95']:.2f}; "
        f"розмір файлу менший у {summ['size_ratio']:.2f} раза; відрізки p50 за раундами "
        f"{'перетинаються — зміна не виміряна' if overlap else 'не перетинаються — зміна виміряна'}")
    plot_latency(lat, out / "figures" / "latency.png")
    plot_ranges(cr, out / "figures" / "channel_ranges.png")

    # ---- 5b. діагностика (поза протоколом пункту 5): ті самі файли й протокол,
    # але з вимкненими ядрами Arm KleidiAI (mlas.disable_kleidiai=1)
    log("\nДіагностика: той самий протокол із mlas.disable_kleidiai=1")
    lat_nk = latency({"FP32": C.make_session(fp32_path, disable_kleidiai=True),
                      "INT8": C.make_session(int8_path, disable_kleidiai=True)}, x0)
    diag = {"config": {"mlas.disable_kleidiai": "1"},
            "p50_ms": {k: v["p50_ms"] for k, v in lat_nk.items()},
            "p95_ms": {k: v["p95_ms"] for k, v in lat_nk.items()},
            "p50_by_round_ms": {k: v["p50_by_round_ms"] for k, v in lat_nk.items()},
            "speedup_p50": lat_nk["FP32"]["p50_ms"] / lat_nk["INT8"]["p50_ms"],
            "fp32_slowdown_vs_default": lat_nk["FP32"]["p50_ms"] / lat["FP32"]["p50_ms"],
            "int8_slowdown_vs_default": lat_nk["INT8"]["p50_ms"] / lat["INT8"]["p50_ms"]}
    diag_md = "\n".join(
        ["| Конфігурація MLAS | FP32 p50, мс | INT8 p50, мс | Прискорення p50 |", "|---|---|---|---|",
         f"| за замовчуванням (KleidiAI увімкнено) — основний протокол | {lat['FP32']['p50_ms']:.2f} | "
         f"{lat['INT8']['p50_ms']:.2f} | {summ['speedup_p50']:.2f} |",
         f"| `mlas.disable_kleidiai=1` — діагностика | {diag['p50_ms']['FP32']:.2f} | "
         f"{diag['p50_ms']['INT8']:.2f} | {diag['speedup_p50']:.2f} |"])
    log("\n" + diag_md)

    # ---- запис
    results = {
        "environment": env, "git": C.git_info(),
        "inputs": {"model": str(args.model), "split": str(args.split), "data": str(args.data),
                   "model_meta": meta},
        "split_control": split,
        "settings": {
            "transform": f"Resize(({img}, {img})) -> ToTensor() -> Normalize(mean={C.MEAN}, std={C.STD})",
            "export": {"dynamo": False, "opset_version": 17, "input_names": ["input"],
                       "output_names": ["logits"], "dynamic_axes": {"input": {"0": "batch"},
                                                                    "logits": {"0": "batch"}},
                       "dummy_input": f"zeros(1, 3, {img}, {img})", "model_mode": "eval"},
            "pre_process": "quant_pre_process(fp32, prep, skip_symbolic_shape=True)",
            "quantize_static": {"quant_format": "QDQ", "activation_type": "QUInt8",
                                "weight_type": "QInt8", "per_channel": [True, False],
                                "calibrate_method": "MinMax", "other": "defaults"},
            "session": {"providers": ["CPUExecutionProvider"], "intra_op_num_threads": C.THREADS,
                        "inter_op_num_threads": C.THREADS, "graph_optimization_level": "default"},
            "quality": {"n_test": len(y), "batch_size": C.EVAL_BS, "order": "split.csv"},
            "latency": {"batch_size": 1, "threads": C.THREADS, "rounds": C.LAT_ROUNDS,
                        "warmup": C.LAT_WARMUP, "measure": C.LAT_MEASURE,
                        "order_in_round": ["FP32", "INT8"], "timer": "time.perf_counter"},
        },
        "calibration": calib,
        "timings_s": {"export": t_export, "quant_pre_process": t_prep,
                      "quantize_static_per_channel": t_q_pc, "quantize_static_per_tensor": t_q_pt,
                      "calibration_images_load": t_calib_load},
        "1_export_check": exp_chk,
        "3_quality": {"n": len(y), "variants": q, "se_single_model_fp32": se_single,
                      "se_ratio_single_over_paired_int8": se_single / q["INT8"]["se_delta_acc"]
                      if q["INT8"]["se_delta_acc"] > 0 else None},
        "4_per_tensor": {"channel_ranges": cr, "quality": q["INT8 per-tensor"]},
        "5_latency": {"file_size_bytes": sizes, "file_size_mb": {k: v / 1e6 for k, v in sizes.items()},
                      "variants": lat, "summary": summ},
        "5b_diagnostic_kleidiai_disabled": diag,
    }
    C.save_json(results, out / "lab4_results.json")

    t_md = "\n".join(["| Етап | Час, с |", "|---|---|",
                      f"| Експорт `torch.onnx.export` → `lab4_fp32.onnx` | {t_export:.2f} |",
                      f"| `quant_pre_process` | {t_prep:.2f} |",
                      f"| `quantize_static` per-channel (калібрування + квантизація) → `lab4_int8.onnx` | {t_q_pc:.2f} |",
                      f"| `quantize_static` per-tensor → `lab4_int8_per_tensor.onnx` | {t_q_pt:.2f} |"])
    e_md = "\n".join(["| Показник | Значення |", "|---|---|",
                      f"| $\\max\\lvert z_{{\\text{{ONNX}}}} - z_{{\\text{{PyTorch}}}}\\rvert$ | {exp_chk['max_abs_logit_diff']:.3e} |",
                      f"| Збіг argmax | {exp_chk['argmax_agreement']:.4f} |",
                      f"| macro-F1 PyTorch | {f4(exp_chk['macro_f1_pytorch'])} |",
                      f"| macro-F1 ONNX FP32 | {f4(exp_chk['macro_f1_onnx_fp32'])} |",
                      f"| Вузлів `Conv` / `BatchNormalization` | {exp_chk['n_conv']} / {exp_chk['n_batchnorm']} |"])
    summary = "\n\n".join([
        "# Результати команди ЛР4",
        f"Середовище: {env['cpu']}, {env['os']}, Python {env['python']}, PyTorch {env['torch']}, "
        f"ONNX {env['onnx']}, ONNX Runtime {env['onnxruntime']}, {C.THREADS} потік, пакет 1. "
        f"Живлення під час вимірювання: {env['power_source_at_latency']}.",
        f"Поділ: train {split['train']['n']} / {split['train']['name_sum']:,}, test {split['test']['n']} / "
        f"{split['test']['name_sum']:,}. Калібрувальна вибірка: {calib['n']} / {calib['name_sum']:,}, {calib['per_class']}.",
        "## 1. Експорт", e_md, "## 2. Час експорту й квантизації", t_md,
        "## 3–4. Якість", q_md,
        f"SE однієї моделі $\\sqrt{{p(1-p)/n}}$ = {se_single:.4f} ($p$ = {p:.4f}).",
        "## 4. Діапазони каналів", ranges_table(cr),
        f"Медіана $r$ = {cr['median_r']:.2f}, $\\log_2$ = {cr['median_r_bits']:.2f} біта.",
        "## 5. Затримка", lat_md,
        f"Прискорення p50 {summ['speedup_p50']:.2f}, p95 {summ['speedup_p95']:.2f}; розмір у "
        f"{summ['size_ratio']:.2f} раза менший; відрізки p50 за раундами "
        f"{'перетинаються' if overlap else 'не перетинаються'}.",
        "## 5b. Діагностика: ядра Arm KleidiAI (поза протоколом)", diag_md,
    ]) + "\n"
    (out / "results" / "summary.md").write_text(summary)
    log("\nРезультати: lab4_results.json, results/summary.md, results/test_logits_*.npy, figures/*.png")


if __name__ == "__main__":
    main()
