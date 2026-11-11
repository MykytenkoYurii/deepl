"""Спільні частини ЛР4: константи, поділ і його контроль, калібрувальна вибірка, дані,
модель із ЛР3, сесії ONNX Runtime, метрики, параметри середовища."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

EVAL_BS = 64                 # якість: пакети по 64
N_CALIB = 200                # калібрувальна вибірка
LAT_ROUNDS, LAT_WARMUP, LAT_MEASURE = 3, 20, 100
THREADS = 1                  # intra_op і inter_op

# Контроль поділу ЛР3 (рядок «Разом»): (кількість, сума числових імен).
SPLIT_CONTROL = {"train": (1200, 11_993_084), "test": (1502, 33_351_444)}
# SHA-256 split.csv із ЛР3 (README ЛР3, розділ 5).
LAB3_SPLIT_SHA256 = "d3a611aec41f6e335ab58287f10a443f3e0e35aa9adfefae662d48b15af567f9"
# Контроль калібрувальної вибірки.
CALIB_CONTROL = {"n": 200, "name_sum": 1_921_470,
                 "per_class": {"buildings": 40, "forest": 35, "glacier": 36,
                               "mountain": 33, "sea": 29, "street": 27}}
# Контроль експорту.
EXPORT_CONTROL = {"max_abs_diff_lt": 1e-3, "n_conv": 20, "n_bn": 0}


# ---------------------------------------------------------------- поділ і калібрування
def read_split(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def part_rows(rows: list[dict], part: str) -> list[dict]:
    return [r for r in rows if r["part"] == part]


def name_num(r: dict) -> int:
    return int(Path(r["file"]).stem)


def count_and_sum(rows: list[dict]) -> tuple[int, int]:
    return len(rows), sum(name_num(r) for r in rows)


def per_class(rows: list[dict]) -> dict[str, int]:
    return {c: sum(r["label"] == c for r in rows) for c in CLASSES}


def calibration_rows(rows: list[dict]) -> list[dict]:
    """Кроки 1–2 пункту 2: train, відсортований за іменем файлу як рядком, і
    перші 200 індексів перестановки default_rng(0) (у порядку перестановки)."""
    train = sorted(part_rows(rows, "train"), key=lambda r: r["file"])
    assert len(train) == 1200, len(train)
    idx = np.random.default_rng(0).permutation(len(train))[:N_CALIB]
    return [train[i] for i in idx]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- дані
def eval_transform(img_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def image_path(data: Path, r: dict) -> Path:
    sub = ("seg_train", "seg_train") if r["part"] == "train" else ("seg_test", "seg_test")
    return Path(data).joinpath(*sub, r["label"], r["file"])


def load_images(rows: list[dict], data: Path, img_size: int) -> np.ndarray:
    """Масив float32 (N, 3, H, W) у порядку рядків; перетворення як для test у ЛР3."""
    tf = eval_transform(img_size)
    out = np.empty((len(rows), 3, img_size, img_size), dtype=np.float32)
    for i, r in enumerate(rows):
        with Image.open(image_path(data, r)) as im:
            out[i] = tf(im.convert("RGB")).numpy()
    return out


def labels(rows: list[dict]) -> np.ndarray:
    return np.array([CLASS_TO_IDX[r["label"]] for r in rows], dtype=np.int64)


# ---------------------------------------------------------------- модель і сесії
def load_torch_model(path: Path):
    """Завантаження за умовою: resnet18(weights=None), fc = Linear(512, 6), eval()."""
    import torch
    from torch import nn
    from torchvision.models import resnet18
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def make_session(path: Path, disable_kleidiai: bool = False):
    """CPUExecutionProvider, 1 потік intra/inter, рівень оптимізації графа за замовчуванням.

    disable_kleidiai=True — лише для діагностичного заміру (вимикає ядра Arm KleidiAI у MLAS);
    в основному протоколі не використовується."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = THREADS
    if disable_kleidiai:
        so.add_session_config_entry("mlas.disable_kleidiai", "1")
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def onnx_logits(sess, x: np.ndarray, bs: int = EVAL_BS) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i:i + bs]})[0] for i in range(0, len(x), bs)])


# ---------------------------------------------------------------- метрики
def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, average="macro", labels=list(range(len(CLASSES))),
                          zero_division=0))


def quality_row(y: np.ndarray, pred: np.ndarray, pred_ref: np.ndarray) -> dict:
    """macro-F1, accuracy, згода з FP32 і парне порівняння з FP32 (пункт 3)."""
    n = len(y)
    ok, ok_ref = pred == y, pred_ref == y
    b = int((ok & ~ok_ref).sum())      # варіант правий, FP32 помиляється
    c = int((~ok & ok_ref).sum())      # навпаки
    d = (b - c) / n
    se = float(np.sqrt(b + c) / n)
    noise = abs(b - c) <= 2 * np.sqrt(b + c)
    if noise:
        verdict = "не відрізняється від шуму (∣b−c∣ ≤ 2√(b+c))"
    else:
        verdict = ("відрізняється від шуму: варіант " + ("кращий" if b > c else "гірший")
                   + " за FP32 (∣b−c∣ > 2√(b+c))")
    return {"macro_f1": macro_f1(y, pred), "accuracy": float(accuracy_score(y, pred)),
            "n_correct": int(ok.sum()), "agreement": float((pred == pred_ref).mean()),
            "n_disagree": int((pred != pred_ref).sum()),
            "b": b, "c": c, "delta_acc": d, "se_delta_acc": se,
            "two_sqrt_b_plus_c": float(2 * np.sqrt(b + c)), "within_noise": bool(noise),
            "verdict": verdict}


# ---------------------------------------------------------------- середовище
def _sh(*args: str) -> str | None:
    try:
        r = subprocess.run(list(args), capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model() -> str:
    if sys.platform == "darwin":
        name = _sh("sysctl", "-n", "machdep.cpu.brand_string")
        if name:
            return name
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith(("model name", "hardware")):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def cpu_features() -> dict:
    """Наявність цілочисельних інструкцій для INT8 (dot product / i8mm / VNNI)."""
    out: dict = {"machine": platform.machine()}
    if sys.platform == "darwin":
        for key in ("hw.optional.arm.FEAT_DotProd", "hw.optional.arm.FEAT_I8MM",
                    "hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME2"):
            v = _sh("sysctl", "-n", key)
            out[key.rsplit(".", 1)[1]] = None if v is None else v == "1"
        out["perf_cores"] = _sh("sysctl", "-n", "hw.perflevel0.physicalcpu")
        out["eff_cores"] = _sh("sysctl", "-n", "hw.perflevel1.physicalcpu")
    elif sys.platform.startswith("linux"):
        try:
            flags = Path("/proc/cpuinfo").read_text()
            for f in ("avx2", "avx512_vnni", "avx_vnni", "amx_int8", "asimddp", "i8mm"):
                out[f] = f" {f}" in flags
        except OSError:
            pass
    return out


def power_source() -> str | None:
    """Джерело живлення і режим енергоспоживання macOS (powermode: 0 — автоматичний,
    1 — Low Power Mode, 2 — High Power)."""
    if sys.platform == "darwin":
        s = _sh("pmset", "-g", "batt")
        if s:
            src = s.splitlines()[0].replace("Now drawing from ", "").strip("'")
            mode = next((ln.split()[-1] for ln in (_sh("pmset", "-g") or "").splitlines()
                         if ln.strip().startswith("powermode")), None)
            return f"{src}, powermode {mode}" if mode is not None else src
    return None


def os_name() -> str:
    if sys.platform == "darwin":
        return f"macOS {platform.mac_ver()[0]} ({platform.platform()})"
    return platform.platform()


def env_info() -> dict:
    import onnx
    import onnxruntime as ort
    import sklearn
    import torch
    import torchvision
    return {"cpu": cpu_model(), "cpu_features": cpu_features(), "logical_cpus": _cpu_count(),
            "os": os_name(), "python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "onnx": onnx.__version__,
            "onnxruntime": ort.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "ort_available_providers": ort.get_available_providers(),
            "provider": "CPUExecutionProvider", "intra_op_num_threads": THREADS,
            "inter_op_num_threads": THREADS, "graph_optimization_level": "default (ORT_ENABLE_ALL)",
            "batch_size_latency": 1, "batch_size_quality": EVAL_BS,
            "power_source": power_source()}


def _cpu_count() -> int | None:
    import os
    return os.cpu_count()


def git_info() -> dict:
    commit = _sh("git", "-C", str(HERE), "rev-parse", "HEAD")
    status = _sh("git", "-C", str(HERE), "status", "--porcelain", "--", ".")
    return {"commit": commit, "uncommitted_changes": bool(status) if status is not None else None}


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
