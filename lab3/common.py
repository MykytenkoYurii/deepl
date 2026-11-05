"""Спільні частини ЛР3: поділ, дані, перетворення, модель, режими, цикл навчання.

Модуль не завантажує test сам по собі: `ImageDataset` будується з переданих рядків
split.csv, а рядки test передає лише analyze.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

HERE = Path(__file__).resolve().parent

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PARTS = ["train", "validation", "test"]

IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

N_TRAIN_PER_CLASS = 200
EPOCHS = 8
TRAIN_BS = 32
EVAL_BS = 64
WEIGHT_DECAY = 1e-4
HEAD_LR = 3e-3
BACKBONE_LR = 3e-4

REGIMES = ["scratch", "frozen", "unfreeze"]
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
# Модулі, які під час навчання переводяться в eval() (стовпець «У eval()»).
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}
# Модулі, параметри яких навчаються (requires_grad=True).
TRAINABLE = {
    "scratch": TOP_MODULES,
    "frozen": ["fc"],
    "unfreeze": ["layer4", "fc"],
}
EXPECTED_TRAINABLE = {"scratch": 11_179_590, "frozen": 3_078, "unfreeze": 8_396_806}
TOTAL_PARAMS = 11_179_590

# Сім запусків протоколу: (номер, режим, зерно).
RUNS = [(1, "scratch", 1), (2, "frozen", 1), (3, "frozen", 2), (4, "frozen", 3),
        (5, "unfreeze", 1), (6, "unfreeze", 2), (7, "unfreeze", 3)]

# Контрольна таблиця поділу: (кількість, сума числових імен).
SPLIT_CONTROL = {
    "buildings": {"train": (200, 1_996_761), "validation": (218, 4_837_323), "test": (219, 4_837_559)},
    "forest":    {"train": (200, 2_032_625), "validation": (237, 5_276_962), "test": (237, 5_291_326)},
    "glacier":   {"train": (200, 2_183_923), "validation": (276, 6_135_652), "test": (277, 6_157_250)},
    "mountain":  {"train": (200, 1_950_345), "validation": (262, 5_789_535), "test": (263, 5_849_765)},
    "sea":       {"train": (200, 1_987_613), "validation": (255, 5_655_068), "test": (255, 5_668_011)},
    "street":    {"train": (200, 1_841_817), "validation": (250, 5_541_845), "test": (251, 5_547_533)},
}


def train_dir(data: Path) -> Path:
    return Path(data) / "seg_train" / "seg_train"


def test_dir(data: Path) -> Path:
    return Path(data) / "seg_test" / "seg_test"


# ---------------------------------------------------------------- поділ
def make_split(data: Path) -> list[dict]:
    """Поділ за процедурою завдання. Повертає рядки {file, label, part}."""
    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for c in CLASSES:
        names = sorted(p.name for p in (train_dir(data) / c).iterdir() if p.suffix == ".jpg")
        idx = np.arange(len(names))
        rng.shuffle(idx)
        for i in sorted(idx[:N_TRAIN_PER_CLASS], key=lambda j: names[j]):
            rows.append({"file": names[i], "label": c, "part": "train"})
    rng = np.random.default_rng(0)
    for c in CLASSES:
        names = sorted(p.name for p in (test_dir(data) / c).iterdir() if p.suffix == ".jpg")
        idx = np.arange(len(names))
        rng.shuffle(idx)
        cut = len(names) // 2
        val = sorted((names[i] for i in idx[:cut]))
        tst = sorted((names[i] for i in idx[cut:]))
        rows += [{"file": f, "label": c, "part": "validation"} for f in val]
        rows += [{"file": f, "label": c, "part": "test"} for f in tst]
    order = {p: i for i, p in enumerate(PARTS)}
    rows.sort(key=lambda r: (order[r["part"]], CLASS_TO_IDX[r["label"]], r["file"]))
    return rows


def write_split(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "part"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_split(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_rows(rows: list[dict], part: str) -> list[dict]:
    return [r for r in rows if r["part"] == part]


def split_control(rows: list[dict]) -> dict:
    """{клас: {частина: (кількість, сума числових імен)}}."""
    out = {c: {p: [0, 0] for p in PARTS} for c in CLASSES}
    for r in rows:
        cell = out[r["label"]][r["part"]]
        cell[0] += 1
        cell[1] += int(Path(r["file"]).stem)
    return {c: {p: tuple(v) for p, v in d.items()} for c, d in out.items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- дані
def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageDataset(Dataset):
    """Зображення з рядків split.csv у їхньому порядку.

    Декодовані RGB-зображення кешуються в пам'яті: це не змінює результат, лише
    прибирає повторне читання JPEG на кожній епосі. Випадкові перетворення
    застосовуються заново при кожному зверненні.
    """

    def __init__(self, rows: list[dict], data: Path, transform):
        parts = {r["part"] for r in rows}
        assert len(parts) == 1, parts
        base = train_dir(data) if parts == {"train"} else test_dir(data)
        self.files = [r["file"] for r in rows]
        self.labels = [CLASS_TO_IDX[r["label"]] for r in rows]
        self.paths = [base / r["label"] / r["file"] for r in rows]
        self.transform = transform
        self._cache: list[Image.Image | None] = [None] * len(rows)

    def __len__(self) -> int:
        return len(self.paths)

    def image(self, i: int) -> Image.Image:
        if self._cache[i] is None:
            with Image.open(self.paths[i]) as im:
                self._cache[i] = im.convert("RGB")
        return self._cache[i]

    def __getitem__(self, i: int):
        return self.transform(self.image(i)), self.labels[i]


def train_loader(ds: Dataset) -> DataLoader:
    # Без власного generator/sampler: перемішування з глобального генератора PyTorch.
    return DataLoader(ds, batch_size=TRAIN_BS, shuffle=True, drop_last=False, num_workers=0)


def eval_loader(ds: Dataset) -> DataLoader:
    return DataLoader(ds, batch_size=EVAL_BS, shuffle=False, num_workers=0)


# ---------------------------------------------------------------- модель і режими
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda: {torch.cuda.get_device_name(device)}"
    if device.type == "mps":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
        return f"mps: {chip or platform.machine()}"
    return f"cpu: {platform.processor() or platform.machine()}"


@dataclass
class Built:
    model: nn.Module
    optimizer: torch.optim.Optimizer


def build(regime: str, seed: int, device: torch.device) -> Built:
    """Кроки 1-5 побудови моделі із завдання."""
    assert regime in REGIMES
    torch.manual_seed(seed)                                         # 1
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)                               # 2
    model.fc = nn.Linear(512, len(CLASSES))                         # 3
    for name, p in model.named_parameters():                        # 4
        p.requires_grad = name.split(".")[0] in TRAINABLE[regime]
    model.to(device)                                                # 5
    if regime == "unfreeze":
        groups = [{"params": list(model.fc.parameters()), "lr": HEAD_LR},
                  {"params": list(model.layer4.parameters()), "lr": BACKBONE_LR}]
    elif regime == "frozen":
        groups = [{"params": list(model.fc.parameters()), "lr": HEAD_LR}]
    else:
        groups = [{"params": list(model.parameters()), "lr": HEAD_LR}]
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    return Built(model, opt)


def set_train_mode(model: nn.Module, regime: str) -> None:
    """model.train(), потім eval() для заморожених модулів режиму."""
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def numel(params) -> int:
    return sum(p.numel() for p in params)


# ---------------------------------------------------------------- навчання й оцінка
@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(y_true, logits: np.ndarray) -> float:
    return float(f1_score(y_true, logits.argmax(1), average="macro",
                          labels=list(range(len(CLASSES))), zero_division=0))


def train_run(regime: str, seed: int, train_ds: Dataset, val_ds: ImageDataset,
              device: torch.device, log=print) -> dict:
    b = build(regime, seed, device)
    model, opt = b.model, b.optimizer
    loader = train_loader(train_ds)
    vloader = eval_loader(val_ds)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    steps = 0
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            steps += 1
            total += loss.item() * len(y)
            n += len(y)
        logits = predict_logits(model, vloader, device)
        f1 = macro_f1(val_ds.labels, logits)
        history.append({"epoch": epoch, "train_loss": total / n, "val_macro_f1": f1,
                        "last_batch_size": len(y)})
        log(f"  {regime} seed={seed} epoch {epoch}: train loss {total / n:.4f}, val macro-F1 {f1:.4f}")
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    assert n == len(train_ds) and steps == EPOCHS * len(loader)
    return {"model": model, "optimizer": opt, "history": history, "steps": steps,
            "train_time_s": elapsed, "val_logits": logits}


# ---------------------------------------------------------------- журнал
def git_info() -> dict:
    def run(*args):
        r = subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--", ".")
    return {"commit": commit, "uncommitted_changes": bool(status) if status is not None else None}


def config_dict(regime: str, seed: int, opt: torch.optim.Optimizer) -> dict:
    """Усі налаштування, включно з тими, що лишилися за замовчуванням."""
    groups = [{k: v for k, v in g.items() if k != "params"} | {"n_params": numel(g["params"])}
              for g in opt.param_groups]
    return {
        "seed": seed,
        "regime": regime,
        "classes": CLASSES,
        "transforms": {
            "train": [
                {"name": "RandomResizedCrop", "size": IMG_SIZE, "scale": [0.7, 1.0],
                 "ratio": [3 / 4, 4 / 3], "interpolation": "bilinear", "antialias": True},
                {"name": "RandomHorizontalFlip", "p": 0.5},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": MEAN, "std": STD},
            ],
            "validation_test": [
                {"name": "Resize", "size": [IMG_SIZE, IMG_SIZE], "interpolation": "bilinear",
                 "antialias": True},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": MEAN, "std": STD},
            ],
            "image_open": "PIL.Image.open(...).convert('RGB')",
        },
        "model": {
            "arch": "torchvision.models.resnet18",
            "weights": None if regime == "scratch" else "ResNet18_Weights.IMAGENET1K_V1",
            "fc": "torch.nn.Linear(512, 6), default init",
            "trainable_modules": TRAINABLE[regime],
            "eval_modules_during_training": EVAL_MODULES[regime],
            "total_params": TOTAL_PARAMS,
            "trainable_params": EXPECTED_TRAINABLE[regime],
            "batchnorm": {"momentum": 0.1, "eps": 1e-5},
            "build_order": ["torch.manual_seed(seed); np.random.seed(seed)", "resnet18(weights=W)",
                            "fc = Linear(512, 6)", "requires_grad", "to(device); AdamW"],
        },
        "optimizer": {"name": "torch.optim.AdamW", "param_groups": groups},
        "loss": {"name": "torch.nn.CrossEntropyLoss", "reduction": "mean",
                 "label_smoothing": 0.0, "weight": None},
        "training": {
            "epochs": EPOCHS, "steps_per_epoch": 38, "total_steps": 304,
            "train_loader": {"batch_size": TRAIN_BS, "shuffle": True, "drop_last": False,
                             "num_workers": 0, "generator": None, "sampler": None},
            "eval_loader": {"batch_size": EVAL_BS, "shuffle": False, "num_workers": 0},
            "epoch_loss": "sum_b L_b*|b| / 1200 (loss before optimizer step)",
            "val_metric": "sklearn f1_score(average='macro'), after every epoch",
            "lr_schedule": None, "grad_clip": None, "amp": False, "dtype": "float32",
            "early_stopping": None, "checkpoint_selection": "weights after epoch 8",
        },
    }


def env_info(device: torch.device) -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "device": device_name(device), "platform": platform.platform()}


def save_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def select_run(logs: dict[str, dict]) -> dict:
    """Правило вибору: режим із найбільшим середнім F, у ньому запуск із найбільшим F.

    logs: {"<regime>_seed<s>": run.json}. За рівності береться перший у порядку RUNS.
    """
    by_regime: dict[str, list[tuple[int, float]]] = {}
    for _, regime, seed in RUNS:
        f = logs[f"{regime}_seed{seed}"]["final_val_macro_f1"]
        by_regime.setdefault(regime, []).append((seed, f))
    means = {r: float(np.mean([f for _, f in v])) for r, v in by_regime.items()}
    best_regime = max(REGIMES, key=lambda r: means[r])
    best_seed, best_f = max(by_regime[best_regime], key=lambda t: t[1])
    return {"regime": best_regime, "seed": best_seed, "val_macro_f1": best_f, "regime_means": means}
