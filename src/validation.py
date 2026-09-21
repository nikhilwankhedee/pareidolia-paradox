"""Leakage-safe validation: canonical folds, OOF, distribution-matched splits."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_canonical_folds(df, hash_col=None, seed=42, n_folds=5):
    """Replicates the campaign's canonical grouped-stratified folds.

    - singleton images (no duplicate hash): StratifiedKFold by label (seed 42)
    - multi-member hash groups: shuffled with RNG(42), assigned round-robin i % n_folds
    Guarantees: groups never cross folds (hash-cross-fold == 0).

    df must contain columns image_id, label and (optionally) hash.
    """
    lab = dict(zip(df["image_id"], df["label"]))
    fold = {}

    sing = df[~df["image_id"].isin(hash_members(df, hash_col))]["image_id"].tolist()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(sing, [lab[i] for i in sing])):
        for j in val_idx:
            fold[sing[j]] = k

    rng = np.random.default_rng(seed)
    groups = df[df["image_id"].isin(hash_members(df, hash_col))].groupby(hash_col)["image_id"] \
        .apply(list) if hash_col else {}
    order = sorted(groups.keys())
    rng.shuffle(order)
    for i, h in enumerate(order):
        for m in groups[h]:
            fold[m] = i % n_folds

    out = df.copy()
    out["fold"] = out["image_id"].map(fold).astype(int)
    return out


def hash_members(df, hash_col):
    if hash_col is None:
        return set()
    vc = df[hash_col].value_counts()
    return set(vc[vc >= 2].index)


def grouped_generator(df, n_folds=5):
    """Yield (train_idx, val_idx) fold splits that keep hash groups intact."""
    for k in range(n_folds):
        tr_idx = np.where(df["fold"].values != k)[0]
        va_idx = np.where(df["fold"].values == k)[0]
        yield tr_idx, va_idx


def oof_predict_az(df, seed=42):
    """Grouped OOF predictions for the canonical order-1 azimuth LGBM."""
    from .azimuth import az_harm
    from .models import make_az_model
    y = df["label"].values.astype(int)
    X = az_harm(df["azimuth"].values, 1)
    po = np.zeros(len(df))
    for tr_idx, va_idx in grouped_generator(df):
        clf = make_az_model(seed=seed)
        clf.fit(X[tr_idx], y[tr_idx])
        po[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
    return po


def test_az_distribution(taz, bins=(0, 45, 90, 135, 180, 225, 270, 315, 360)):
    from .azimuth import bin_counts
    return bin_counts(np.asarray(taz, dtype=float), np.array(bins, dtype=float))


def sample_match_az_distribution(single_ids, az_by_id, target_counts, seed):
    """Subsample singleton pool so az-bin proportions match a target histogram."""
    from .azimuth import bin_index
    rng = np.random.default_rng(seed)
    target = np.asarray(target_counts, dtype=float)
    w = target / target.sum()
    total = min(len(single_ids), int(target.sum()))
    bins = {}
    for i in single_ids:
        k = int(bin_index([az_by_id[i]])[0])
        bins.setdefault(k, []).append(i)
    picks = []
    for k in range(8):
        n = int(round(total * w[k]))
        pool = bins.get(k, [])
        rng.shuffle(pool)
        picks.extend(pool[:n])
    rng.shuffle(picks)
    return picks, w, total