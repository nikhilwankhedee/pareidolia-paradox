"""Grouped-stratified cross-validation by exact image hash.

Leakage-free validation design for the Pareidolia Paradox dataset.

Grouping rule
-------------
Each exact image hash is one group. A group may NOT span two folds, so no
exact pixel duplicate ever appears in both the training and validation parts
of a fold.

Stratification rule (deterministic, seed-fixed) given the dataset's structure:
* 4,938 singleton groups (one row, label 0 or 1)
* 1,458 two-row groups that are ALL mixed-label (labels {0,1}) -- pixel-
  identical images labeled both Depth and Rise.
Because a mixed 2-row group carries exactly one class-0 and one class-1 row,
distributing the mixed groups evenly across folds preserves global class
balance:
* singletons are stratified by their own label via StratifiedKFold(5);
* mixed groups are shuffled with the seed and assigned round-robin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DEFAULT_SEED = 42
DEFAULT_N_FOLDS = 5


def construct_folds(frame: pd.DataFrame, hash_col: str = "hash",
                    label_col: str = "label", n_folds: int = DEFAULT_N_FOLDS,
                    seed: int = DEFAULT_SEED, foo=None) -> pd.DataFrame:
    """Return a copy of `frame` with a 'fold' column (0..n_folds-1).

    `frame` must already contain the exact-image hash per row (`hash_col`).
    """
    frame = frame.copy()
    comp = group_composition(frame, hash_col, label_col)
    singl_hashes = [h for h, c in comp.items() if c != "mixed"]
    mixed_hashes = [h for h, c in comp.items() if c == "mixed"]

    # singletons: stratified by their own label (exact at row level)
    singl_meta = frame[frame[hash_col].isin(singl_hashes)]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of_singl = {}
    for fold, (_, val_idx) in enumerate(skf.split(singl_meta, singl_meta[label_col])):
        for i in val_idx:
            fold_of_singl[singl_meta.iloc[i][hash_col]] = fold

    # mixed groups: seeded shuffle + round-robin to even the 0/1 spread
    rng = np.random.default_rng(seed)
    order = mixed_hashes.copy()
    rng.shuffle(order)
    fold_of_mixed = {h: i % n_folds for i, h in enumerate(order)}

    fold_of_hash = {**fold_of_singl, **fold_of_mixed}
    frame["fold"] = [fold_of_hash[h] for h in frame[hash_col]]
    return frame


def group_composition(frame, hash_col, label_col):
    gc = frame.groupby(hash_col)[label_col].agg(lambda s: tuple(sorted(s)))
    return gc.map(lambda t: "mixed" if len(t) > 1 else str(t[0])).to_dict()


def validate_folds(frame: pd.DataFrame, hash_col: str = "hash",
                   fold_col: str = "fold", verbose: bool = True) -> dict:
    """Verify no exact hash crosses folds; return per-fold statistics.

    Raises SystemExit if any hash spans multiple folds (leakage).
    """
    usage = {}
    for h, f in zip(frame[hash_col], frame[fold_col]):
        usage.setdefault(h, set()).add(int(f))
    n_cross = sum(1 for s in usage.values() if len(s) > 1)
    if verbose:
        print(f"unique hashes/groups    : {len(usage)}")
        print(f"max folds per hash      : {max(len(s) for s in usage.values())}")
        print(f"cross-fold dup groups   : {n_cross}")
    if n_cross:
        raise SystemExit("ABORT: a hash spans multiple folds (leakage).")

    rows = []
    for f in sorted(frame[fold_col].unique()):
        sub = frame[frame[fold_col] == f]
        c0 = int((sub[label_col] == 0).sum())
        c1 = int((sub[label_col] == 1).sum())
        rows.append({
            "fold": f, "rows": len(sub), "unique_hashes": sub[hash_col].nunique(),
            "class_0_count": c0, "class_1_count": c1,
            "class_0_prop": round(c0 / len(sub), 4), "class_1_prop": round(c1 / len(sub), 4),
        })
        if verbose:
            print(f"  fold {f}: rows={len(sub):5d} groups={sub[hash_col].nunique():5d} "
                  f"c0={c0} ({c0/len(sub):.4f}) c1={c1} ({c1/len(sub):.4f})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import hashlib
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]  # repo root
    train_dir = root / "Train" / "images" / "train_images"
    meta = pd.read_csv(root / "Train" / "train_metadata.csv")
    print("Loading metadata + computing sha256 hashes (raw data must be present)...")
    meta["hash"] = [hashlib.sha256((lambda p: open(p, "rb").read())(train_dir / img)).hexdigest()
                    for img in meta["image_id"]]
    folded = construct_folds(meta)
    stats = validate_folds(folded)
    out = folded[["image_id", "hash", "label", "sun_azimuth_angle", "fold"]]
    out_path = root / "results" / "fold_assignments.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} rows)")
