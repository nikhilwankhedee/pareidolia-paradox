"""Lunar expert/pretrained-embedding experiments.

Campaign rule: an experiment runs ONLY if part 14 produced a specific,
concrete reason to run it (a hypothesis with a falsifiable expected result).
Otherwise this module builds the documented SKIP entry that is written to
lunar_embedding_results.csv.
"""
import numpy as np
import pandas as pd

from .geometry import _grads  # reuse internal helpers


def decide(hypothesis):
    """hypothesis: dict with keys name, expected_effect, mechanism.
    Returns 'RUN' if all three are concrete, else ('SKIP', reason)."""
    if not isinstance(hypothesis, dict):
        return "SKIP", "no hypothesis provided"
    ok = all(hypothesis.get(k) for k in ("name", "expected_effect", "mechanism"))
    if not ok:
        return "SKIP", "hypothesis not concrete (name/expected_effect/mechanism missing)"
    return "RUN", ""


def skip_entry(hypothesis, reason):
    return pd.DataFrame([{
        "experiment": hypothesis.get("name", "lunar_embedding"),
        "decision": "SKIP",
        "reason": reason,
        "validation_BA": np.nan,
    }])


def run_lunar_experiment(df, img_dir, cache_dir=None):
    """Executes the geometry-on-lunar specialist check if decided RUN."""
    from .geometry import build_geometry_matrix
    from .metrics import add_metrics_row
    from .models import fit_lgb
    from .validation import build_canonical_folds, grouped_generator
    from .azimuth import az_harm

    df = build_canonical_folds(df, hash_col="hash" if "hash" in df.columns else None)
    Xg = build_geometry_matrix(img_dir, list(df["image_id"]),
                               dict(zip(df["image_id"], df["azimuth"].astype(float))),
                               cache_dir=cache_dir)
    Xaz = az_harm(df["azimuth"].values, 1)
    X = np.hstack([Xaz, Xg])
    y = df["label"].values
    po = np.zeros(len(df))
    for tr_idx, va_idx in grouped_generator(df):
        tree = fit_lgb(X[tr_idx], y[tr_idx], seed=42)
        po[va_idx] = tree(X[va_idx])
    rows = []
    add_metrics_row(rows, "azimuth+geometry(OOF)", y, po)
    trees = [f"per-fold BA placeholder seeded by focus {df['label'].mean():.3f}"]
    return rows, (po, y), trees