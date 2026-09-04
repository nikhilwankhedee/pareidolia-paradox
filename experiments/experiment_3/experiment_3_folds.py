"""
EXPERIMENT 3, PART 1 (LOCAL): GROUPED-STRATIFIED FOLD CONSTRUCTION + MODEL A
=============================================================================
The Pareidolia Paradox Dataset

Builds the deterministic grouped-stratified fold assignment (by exact image
hash) that ALL five Experiment-3 models will share, verifies it is leakage-
free, and runs Model A (azimuth-only logistic regression) on those identical
folds. This local script runs the cheap parts (folding + Model A + data
pipeline producer). The CNN models B-E run on Kaggle GPU via the notebook
experiment_3_signal_decomposition.ipynb using the SAME fold assignment.

CRITICAL GROUPING FACT (verified):
  * 6,396 unique image hashes across 7,854 train rows.
  * 4,938 singleton groups; 1,458 multi-image groups.
  * EVERY one of the 1,458 multi groups has exactly 2 rows whose labels are
    (0, 1): pixel-identical images labeled both Depth(0) and Rise(1).
  * Therefore group-level label composition is: "0", "1", or "mixed"(0+1).

Grouped stratification strategy (deterministic, seed-fixed):
  * Singletons are stratified by their own label via StratifiedKFold(5),
    which is exact because a singleton group == a single row.
  * Mixed 2-row groups carry one class-0 and one class-1 row, so distributing
    them evenly across folds preserves global class balance. They are
    assigned round-robin over folds after a seeded shuffle.
  * Fold assignment is saved once and reused by every model.
"""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    recall_score,
)

# ============================================================
# CONFIG
# ============================================================
def _repo_root():
    _p = Path(__file__).resolve().parent
    while not (_p / "README.md").exists():
        _p = _p.parent
    return _p

ROOT = _repo_root()
TRAIN_META = ROOT / "Train" / "train_metadata.csv"
TRAIN_IMG_DIR = ROOT / "Train" / "images" / "train_images"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
RNG = np.random.default_rng(SEED)

FOLD_ASSIGN = OUT_DIR / "fold_assignments.csv"
OOF_CSV = OUT_DIR / "signal_decomposition_oof.csv"


def section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def hash_img(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_hash_table():
    """Map image_id -> exact sha256 hash of its raw PNG bytes."""
    img_ids = pd.read_csv(TRAIN_META)["image_id"].tolist()
    table = {}
    t = time.time()
    for i, img in enumerate(img_ids):
        table[img] = hash_img(TRAIN_IMG_DIR / img)
        if (i + 1) % 2000 == 0:
            print(f"  hashed {i+1}/{len(img_ids)} ({time.time()-t:.1f}s)")
    return table


# ============================================================
# FOLD CONSTRUCTION (grouped-stratified deterministic)
# ============================================================
def construct_folds(meta, hash_of):
    """Return DataFrame with columns: image_id, hash, label, azimuth, fold.

    Groups are defined by exact image hash. Group label composition is
    '0' (singleton class 0), '1' (singleton class 1), or 'm' (mixed 0+1).
    """
    meta = meta.copy()
    meta["hash"] = [hash_of[i] for i in meta["image_id"]]

    # group label composition
    group_lab = meta.groupby("hash")["label"].agg(lambda s: tuple(sorted(s)))
    comp = group_lab.map(lambda t: "m" if len(t) > 1 else str(t[0]))
    comp = comp.to_dict()

    singl_hashes = [h for h in comp if comp[h] != "m"]
    mixed_hashes = [h for h in comp if comp[h] == "m"]

    # --- singletons: StratifiedKFold on each singleton's own label ---
    singl_meta = meta[meta["hash"].isin(singl_hashes)].copy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of_singl = {}
    for fold, (_, val_idx) in enumerate(skf.split(singl_meta, singl_meta["label"])):
        for i in val_idx:
            fold_of_singl[singl_meta.iloc[i]["hash"]] = fold

    # --- mixed groups: seeded shuffle + round-robin to even the 0/1 spread ---
    rng = np.random.default_rng(SEED)
    mixed_order = mixed_hashes.copy()
    rng.shuffle(mixed_order)
    fold_of_mixed = {h: i % N_FOLDS for i, h in enumerate(mixed_order)}

    fold_of_hash = {**fold_of_singl, **fold_of_mixed}
    meta["fold"] = [fold_of_hash[h] for h in meta["hash"]]
    return meta


def validate_folds(meta):
    """Verify leakage: no hash crosses folds; report fold stats."""
    section("FOLD VALIDATION CHECKS")
    # group -> set of folds
    usage = defaultdict(set)
    for h, f in zip(meta["hash"], meta["fold"]):
        usage[h].add(int(f))
    n_cross = sum(1 for u in usage.values() if len(u) > 1)
    print(f"total unique hashes/groups : {len(usage)}")
    print(f"max folds occupied by a hash: {max(len(u) for u in usage.values())}")
    print(f"cross-fold duplicate groups: {n_cross}")
    if n_cross != 0:
        raise SystemExit("ABORT: hash crosses folds — grouped stratification failed.")
    print("OK: no exact image hash spans multiple folds.")

    rows = []
    for f in range(N_FOLDS):
        sub = meta[meta["fold"] == f]
        c0 = int((sub["label"] == 0).sum())
        c1 = int((sub["label"] == 1).sum())
        rows.append({
            "fold": f,
            "rows": len(sub),
            "unique_hashes": sub["hash"].nunique(),
            "class_0_count": c0,
            "class_1_count": c1,
            "class_0_prop": round(c0 / len(sub), 4),
            "class_1_prop": round(c1 / len(sub), 4),
        })
        print(f"  fold {f}: rows={len(sub):5d} groups={sub['hash'].nunique():5d} "
              f"c0={c0} ({c0/len(sub):.4f}) c1={c1} ({c1/len(sub):.4f})")
    stats = pd.DataFrame(rows)
    stats.to_csv(OUT_DIR / "fold_statistics.csv", index=False)
    return stats


# ============================================================
# MODEL A: azimuth-only logistic regression
# ============================================================
def run_model_A(meta):
    section("MODEL A — AZIMUTH ONLY (sin/cos Logistic Regression)")
    OOF = np.zeros(len(meta))
    t0 = time.time()
    for f in range(N_FOLDS):
        tr = meta[meta["fold"] != f]
        va = meta[meta["fold"] == f]
        theta = np.deg2rad(tr["sun_azimuth_angle"].values)
        X_tr = np.stack([np.sin(theta), np.cos(theta)], axis=1)
        va_theta = np.deg2rad(va["sun_azimuth_angle"].values)
        X_va = np.stack([np.sin(va_theta), np.cos(va_theta)], axis=1)
        clf = LogisticRegression(max_iter=2000, random_state=SEED)
        clf.fit(X_tr, tr["label"].values)
        OOF[va.index] = clf.predict_proba(X_va)[:, 1]
    meta = meta.copy()
    meta["prob_A"] = OOF
    y = meta["label"].values
    fold_rows = []
    for f in range(N_FOLDS):
        va = meta[meta["fold"] == f]
        p = va["prob_A"].values
        yv = va["label"].values
        ba = balanced_accuracy_score(yv, (p >= 0.5).astype(int))
        r0 = recall_score(yv, (p >= 0.5).astype(int), pos_label=0)
        r1 = recall_score(yv, (p >= 0.5).astype(int), pos_label=1)
        auc = roc_auc_score(yv, p)
        fold_rows.append([f, ba, r0, r1, auc])
        print(f"  fold {f}: BA={ba:.4f} R0={r0:.4f} R1={r1:.4f} AUC={auc:.4f}")
    fr = np.array(fold_rows)
    print(f"\n  mean BA={fr[:,1].mean():.4f} std BA={fr[:,1].std():.4f}")
    oof_ba = balanced_accuracy_score(y, (OOF >= 0.5).astype(int))
    oof_r0 = recall_score(y, (OOF >= 0.5).astype(int), pos_label=0)
    oof_r1 = recall_score(y, (OOF >= 0.5).astype(int), pos_label=1)
    oof_auc = roc_auc_score(y, OOF)
    print(f"  OOF BA={oof_ba:.4f} R0={oof_r0:.4f} R1={oof_r1:.4f} AUC={oof_auc:.4f}")
    print(f"  OOF confusion (t=0.5):\n{confusion_matrix(y, (OOF>=0.5).astype(int))}")
    # OOF-optimized global threshold
    best_t, best_ba = 0.5, oof_ba
    for t in np.linspace(0.0, 1.0, 201):
        ba = balanced_accuracy_score(y, (OOF >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    print(f"  OOF-optimal threshold={best_t:.3f} BA={best_ba:.4f}")
    print(f"  training time (A, all folds)={time.time()-t0:.2f}s")
    return meta, {
        "model": "A_azimuth",
        "mean_BA": float(fr[:,1].mean()),
        "std_BA": float(fr[:,1].std()),
        "OOF_BA": float(oof_ba),
        "OOF_recall_0": float(oof_r0),
        "OOF_recall_1": float(oof_r1),
        "OOF_ROC_AUC": float(oof_auc),
        "optimal_OOF_threshold": float(best_t),
        "optimal_OOF_BA": float(best_ba),
        "training_time_seconds": float(time.time() - t0),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-fold-assignment", action="store_true",
                    help="build + save fold_assignments.csv and fold_statistics.csv, then exit")
    args = ap.parse_args()

    meta = pd.read_csv(TRAIN_META)
    print(f"Loaded {len(meta)} train rows, {meta['label'].nunique()} classes.")

    hash_of = build_hash_table()
    print(f"Computed hashes for {len(hash_of)} images; {len(set(hash_of.values()))} unique.")

    meta = construct_folds(meta, hash_of)
    stats = validate_folds(meta)

    if args.save_fold_assignment:
        assign = meta.copy()
        assign["sun_azimuth_angle"] = meta["sun_azimuth_angle"].values
        out = pd.DataFrame({
            "image_id": meta["image_id"],
            "hash": meta["hash"],
            "label": meta["label"],
            "azimuth": meta["sun_azimuth_angle"],
            "fold": meta["fold"],
        })
        out.to_csv(FOLD_ASSIGN, index=False)
        print(f"\nSaved fold assignment -> {FOLD_ASSIGN} ({len(out)} rows)")
        print(f"Saved fold statistics  -> {OUT_DIR/'fold_statistics.csv'}")
        cfg = {"n_folds": N_FOLDS, "seed": SEED, "n_rows": int(len(meta)),
               "n_unique_hashes": int(meta['hash'].nunique()),
               "grouping": "exact sha256 image hash",
               "stratification": "singletons by label, mixed groups round-robin"}
        with open(OUT_DIR / "training_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        return

    # Model A + OOF
    meta_withA, resA = run_model_A(meta)

    # Save OOF with at least model A probabilities
    oof = pd.DataFrame({
        "image_id": meta["image_id"],
        "hash": meta["hash"],
        "fold": meta["fold"],
        "label": meta["label"],
        "azimuth": meta["sun_azimuth_angle"],
        "prob_A_azimuth": meta_withA["prob_A"],
    })
    oof.to_csv(OOF_CSV, index=False)
    print(f"\nSaved OOF (partial, Model A) -> {OOF_CSV} ({len(oof)} rows)")

    res = resA
    results = pd.DataFrame([res])
    results.to_csv(OUT_DIR / "experiment_results.csv", index=False)
    print("Saved experiment_results.csv (Model A).")
    print("\nMODEL A DONE. CNNs (B-E) run on Kaggle GPU via notebook.")


if __name__ == "__main__":
    main()
