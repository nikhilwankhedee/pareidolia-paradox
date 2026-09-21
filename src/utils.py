"""Shared utilities: seeds, progress, CSV writing, caching, env record, zipping."""
import hashlib
import json
import os
import pickle
import time
import zipfile
from pathlib import Path


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import lightgbm as lgb
        _ = lgb  # lightgbm is deterministic given random_state params
    except Exception:
        pass
    return seed


_T0 = None

OUTDIR = None  # set to the campaign `OUT` dir by the setup cell


def tick():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0


def pt(phase, msg):
    print(f"[{phase}] {time.strftime('%H:%M:%S')} ({tick():7.1f}s) {msg}", flush=True)


def save_csv(df, path, root=None):
    path = Path(path)
    root = Path(root) if root is not None else (Path(OUTDIR) if OUTDIR else None)
    if root is not None and not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def cprint(dct):
    for k, v in dct.items():
        print(f"  {k:<28} {v}")


def env_record(out_dir, seed, extra=None):
    import numpy as np
    import pandas as pd
    import sklearn
    try:
        import lightgbm as lgb
        lgbv = lgb.__version__
    except Exception:
        lgbv = None
    try:
        import matplotlib
        mpv = matplotlib.__version__
    except Exception:
        mpv = None
    rec = {
        "python": __import__("sys").version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lgbv,
        "matplotlib": mpv,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        rec.update(extra)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment_record.json").write_text(json.dumps(rec, indent=2))
    return rec


# ---------------------------------------------------------------- caching ----
def _key_hash(key):
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]


def cache_load(cache_dir, key):
    path = Path(cache_dir) / f"{_key_hash(key)}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def cache_save(cache_dir, key, value):
    path = Path(cache_dir) / f"{_key_hash(key)}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(value, f)
    return path


def big_zip(src_root, zip_path):
    src_root = Path(src_root)
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_root.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(src_root)))
    return zip_path