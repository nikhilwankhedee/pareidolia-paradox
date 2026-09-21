"""Dataset discovery (Kaggle + local) and validation of structure."""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from skimage import io as skio
except Exception:
    skio = None


def find_local_dataset_root():
    """Local fallback: env FINAL_CAMPAIGN_REPO, cwd, parents."""
    envp = os.environ.get("FINAL_CAMPAIGN_REPO")
    cands = []
    if envp:
        cands.append(Path(envp))
    for base in [Path.cwd(), *Path.cwd().parents]:
        cands.append(base)
        cands.append(base / "The Pareidolia Paradox Dataset")
    for c in cands:
        if (c / "Train" / "train_metadata.csv").exists():
            return c
    return None


def _find_image_dir(lookup_ids, sample=64):
    """Scan /kaggle/input for a directory densely populated with images whose
    filenames match a sample of the requested ids."""
    ids = set(sorted(list(lookup_ids))[:sample])
    hits = []
    for base in sorted(Path("/kaggle/input").iterdir()):
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not any(
                k in d.lower() for k in ("prev", "old", "cache", "__"))]
            if len(filenames) < 50:
                continue
            names = os.listdir(dirpath)
            n_hit = sum(1 for n in names if n in ids)
            if n_hit >= 50:
                hits.append((None, n_hit, Path(dirpath)))  # (skip, n, path)
    if not hits:
        return None
    hits.sort(key=lambda t: -t[1])
    return hits[0][2]


def discover_dataset():
    is_kaggle = Path("/kaggle/working").exists()
    if is_kaggle:
        roots = []
        for base in os.walk("/kaggle/input", followlinks=True):
            if "train_metadata.csv" in base[2]:
                roots.append(Path(base[0]))
        if not roots:
            raise RuntimeError("train_metadata.csv not found under /kaggle/input")
        root = roots[0]
        df = pd.read_csv(root / "train_metadata.csv")
        tmeta = None
        for r in roots + [p for p in sorted(Path("/kaggle/input").iterdir()) if p.is_dir()]:
            if (r / "test_metadata.csv").exists():
                tmeta = pd.read_csv(r / "test_metadata.csv")
                break
        if tmeta is None:
            raise RuntimeError("test_metadata.csv not found under /kaggle/input")
        img_train = _find_image_dir(df["image_id"].tolist())
        img_test = _find_image_dir(tmeta["image_id"].tolist())
        note = f"kaggle root={root}"
    else:
        root = find_local_dataset_root()
        if root is None:
            raise RuntimeError("local dataset root not found (set FINAL_CAMPAIGN_REPO)")
        df = pd.read_csv(root / "Train" / "train_metadata.csv")
        tmeta = pd.read_csv(root / "Test" / "test_metadata.csv")
        img_train = root / "Train" / "images" / "train_images"
        img_test = root / "Test" / "images" / "eval_images"
        note = f"local root={root}"
    for d in (df, tmeta):
        if "sun_azimuth_angle" in d.columns:
            d.rename(columns={"sun_azimuth_angle": "azimuth"}, inplace=True)
        if "azimuth" not in d.columns:
            raise RuntimeError("no azimuth column in metadata")
    df["label"] = df["label"].astype(int)
    df["image_id"] = df["image_id"].astype(str)
    tmeta["image_id"] = tmeta["image_id"].astype(str)
    return {"df": df, "tmeta": tmeta, "img_train": img_train, "img_test": img_test,
            "note": note, "is_kaggle": is_kaggle}


def verify_structure(data, expect_train=7854, expect_test=2000):
    checks = {}
    df, tmeta = data["df"], data["tmeta"]
    checks["train_rows"] = len(df)
    checks["test_rows"] = len(tmeta)
    checks["train_unique"] = df["image_id"].nunique()
    checks["test_unique"] = tmeta["image_id"].nunique()
    checks["labels"] = sorted(df["label"].unique().tolist())
    checks["train_counts_ok"] = len(df) == expect_train and df["image_id"].nunique() == expect_train
    checks["test_counts_ok"] = len(tmeta) == expect_test and tmeta["image_id"].nunique() == expect_test
    return checks


def open_image(path):
    """Return float32 grayscale 2D array in [0,1]."""
    if skio is not None:
        try:
            arr = skio.imread(str(path))
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            return (arr / 255.0).astype(np.float32)
        except Exception:
            pass
    import PIL.Image
    with PIL.Image.open(str(path)) as im:
        arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    return arr