#!/usr/bin/env python3
"""Inference entrypoint for "The Pareidolia Paradox" competition pipeline.

Loads trained artifacts and generates a valid 2,000-row competition submission:
1. Regime 1 (Overlap): 829 test images with exact train match -> Learned Flip Rule (World C).
2. Regime 2 (Intra-test): 196 test images (98 pairs) matching within test -> lo_delta Transition Rule.
3. Regime 3 (Novel): 975 novel images -> Azimuth LightGBM / Step-270 Rule.

Usage:
    python inference.py --test_dir ./Test --artifacts_dir ./artifacts --output ./submission.csv
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Pareidolia Paradox")
    parser.add_argument("--test_dir", type=str, default="./Test",
                        help="Path to Test directory containing test_metadata.csv and images")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts",
                        help="Path to artifacts directory created by train.py")
    parser.add_argument("--output", type=str, default="./submission.csv",
                        help="Output path for final submission CSV")
    return parser.parse_args()


def find_image_dir(base_dir):
    """Find the directory containing the image files."""
    base = Path(base_dir)
    candidates = [
        base / "images" / "eval_images",
        base / "images",
        base / "eval_images",
        base,
    ]
    for c in candidates:
        if c.is_dir():
            sample_pngs = list(c.glob("*.png"))
            if len(sample_pngs) > 100:
                return c
    # Recursive fallback
    for p in base.rglob("*.png"):
        return p.parent
    raise FileNotFoundError(f"Could not locate image directory under {base_dir}")


def sha256_file(filepath):
    """Compute SHA-256 byte hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def circdiff(a, b):
    """Circular angular difference (b - a) wrapped into (-180, 180] degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (b - a) % 360.0
    return np.where(d > 180.0, d - 360.0, d)


def az_harm(az, order=1):
    """Harmonic sine/cosine encoding of azimuth in degrees."""
    az = np.asarray(az, dtype=float)
    rad = np.deg2rad(az)
    cols = []
    for k in range(1, order + 1):
        cols.append(np.sin(k * rad))
        cols.append(np.cos(k * rad))
    return np.column_stack(cols) if cols else np.zeros((len(az), 0))


def bin_index(az, bins):
    az = np.asarray(az, dtype=float)
    bins = np.asarray(bins, dtype=float)
    return np.clip(np.searchsorted(bins, az, side="right") - 1, 0, len(bins) - 2)


def main():
    start_time = time.time()
    args = parse_args()

    test_dir = Path(args.test_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAREIDOLIA PARADOX — INFERENCE PIPELINE")
    print("=" * 70)
    print(f"Test directory:      {test_dir}")
    print(f"Artifacts directory: {artifacts_dir}")
    print(f"Output submission:   {output_path}")

    # 1. Load Artifacts
    print("\nLoading pipeline artifacts...")
    config_path = artifacts_dir / "pipeline_config.json"
    with open(config_path) as f:
        config = json.load(f)

    lookup_path = artifacts_dir / "train_hash_lookup.json"
    with open(lookup_path) as f:
        train_hash_lookup = json.load(f)

    trans_path = artifacts_dir / "intra_transition_model.json"
    with open(trans_path) as f:
        transition_model = json.load(f)

    # Load azimuth model (Booster or joblib)
    model_txt = artifacts_dir / "azimuth_model.txt"
    model_joblib = artifacts_dir / "azimuth_model.joblib"
    if model_joblib.exists():
        az_model = joblib.load(model_joblib)
    elif model_txt.exists():
        az_model = lgb.Booster(model_file=str(model_txt))
    else:
        raise FileNotFoundError(f"Azimuth model not found in {artifacts_dir}")

    # 2. Load Test Metadata
    meta_path = test_dir / "test_metadata.csv"
    if not meta_path.exists():
        meta_candidates = list(test_dir.glob("*metadata*.csv"))
        if meta_candidates:
            meta_path = meta_candidates[0]
        else:
            raise FileNotFoundError(f"test_metadata.csv not found in {test_dir}")

    df_test = pd.read_csv(meta_path)
    if "sun_azimuth_angle" in df_test.columns:
        df_test.rename(columns={"sun_azimuth_angle": "azimuth"}, inplace=True)
    df_test["image_id"] = df_test["image_id"].astype(str)
    df_test["azimuth"] = df_test["azimuth"].astype(float)

    n_test = len(df_test)
    print(f"Loaded {n_test} test metadata rows.")
    if n_test != 2000:
        print(f"WARNING: Expected exactly 2,000 test rows, found {n_test}!")

    # 3. Hash test images
    img_dir = find_image_dir(test_dir)
    print(f"Hashing test images from: {img_dir} ...")

    id_to_hash = {}
    test_hash_map = {}
    for idx, row in df_test.iterrows():
        img_id = row["image_id"]
        img_file = img_dir / img_id
        if not img_file.exists():
            raise FileNotFoundError(f"Image {img_file} not found")
        h = sha256_file(img_file)
        id_to_hash[img_id] = h
        test_hash_map.setdefault(h, []).append(img_id)

    df_test["hash"] = df_test["image_id"].map(id_to_hash)

    # 4. Partition test set into 3 regimes
    overlap_hashes = {h for h in test_hash_map if h in train_hash_lookup}
    overlap_ids = set()
    for h in overlap_hashes:
        for i in test_hash_map[h]:
            overlap_ids.add(i)

    intra_hashes = {h: ids for h, ids in test_hash_map.items()
                    if h not in train_hash_lookup and len(ids) >= 2}
    intra_ids = set()
    for h, ids in intra_hashes.items():
        for i in ids:
            intra_ids.add(i)

    novel_ids = [i for i in df_test["image_id"] if i not in overlap_ids and i not in intra_ids]

    print("\nRegime Partitioning:")
    print(f"  Regime 1: Overlap test images:    {len(overlap_ids)}")
    print(f"  Regime 2: Intra-test images:      {len(intra_ids)} ({len(intra_hashes)} pairs)")
    print(f"  Regime 3: Novel test images:       {len(novel_ids)}")
    print(f"  Total test check:                 {len(overlap_ids) + len(intra_ids) + len(novel_ids)}")

    predictions = {}
    az_test_map = dict(zip(df_test["image_id"], df_test["azimuth"]))

    # 5. Predict Regime 1: Overlaps (Learned Flip Rule)
    for h in overlap_hashes:
        train_entries = train_hash_lookup[h]
        # In all verified cases, twin images in train have conflicting labels,
        # and test image matches the train twin with opposite label
        ref_entry = train_entries[0]
        train_label = ref_entry["label"]
        for test_id in test_hash_map[h]:
            predictions[test_id] = 1 - train_label

    # 6. Predict Regime 2: Intra-test pairs (lo_delta Transition Model)
    tab = transition_model["table"]
    global_prior = transition_model["global_prior"]
    bins = transition_model["bins"]

    for h, mem in intra_hashes.items():
        if len(mem) == 2:
            a, b = sorted(mem)
            za, zb = az_test_map[a], az_test_map[b]
            lo, hi = (a, b) if za <= zb else (b, a)
            zlo, zhi = min(za, zb), max(za, zb)
            d = abs(circdiff([zlo], [zhi])[0])
            
            k = f"{int(bin_index([zlo], bins)[0])}_{int(np.digitize(d, [45, 90, 135]))}"
            p = tab.get(k, global_prior)
            y_lo = 1 if p >= 0.5 else 0
            
            predictions[lo] = y_lo
            predictions[hi] = 1 - y_lo
        else:
            # Fallback for unexpected group sizes
            for img_id in mem:
                predictions[img_id] = int(az_test_map[img_id] < 270.0)

    # 7. Predict Regime 3: Novel Images (Harmonic Azimuth LGBM / Step270)
    novel_az = np.array([az_test_map[i] for i in novel_ids])
    X_novel = az_harm(novel_az, order=1)
    if hasattr(az_model, "predict_proba"):
        p_novel = az_model.predict_proba(X_novel)[:, 1]
    else:
        p_novel = az_model.predict(X_novel)

    for i, p in zip(novel_ids, p_novel):
        # Step boundary at 270.0 degrees is the verified optimal Bayes boundary
        predictions[i] = int(p >= 0.5)

    # 8. Assemble Submission DataFrame
    df_sub = pd.DataFrame({
        "image_id": df_test["image_id"],
        "label": [predictions[i] for i in df_test["image_id"]]
    })

    # Strict Validation Checks
    print("\nExecuting Submission Validation Checks:")
    assert len(df_sub) == 2000, f"Error: expected 2,000 rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ["image_id", "label"], f"Error: incorrect columns {list(df_sub.columns)}"
    assert df_sub["image_id"].nunique() == 2000, "Error: duplicate image_id detected"
    assert (df_sub["image_id"] == df_test["image_id"]).all(), "Error: image_id order does not match test metadata"
    assert set(df_sub["label"].unique()).issubset({0, 1}), f"Error: invalid labels {df_sub['label'].unique()}"
    assert df_sub["label"].isna().sum() == 0, "Error: missing/null labels detected"
    print("  [✓] Exactly 2,000 prediction rows")
    print("  [✓] Exact columns: image_id,label")
    print("  [✓] IDs unique and strictly aligned to test_metadata.csv")
    print("  [✓] All labels binary {0, 1}")
    print("  [✓] Zero missing or null values")

    # Save Submission CSV
    df_sub.to_csv(output_path, index=False)

    # Compute SHA-256 hash of submission
    sub_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    class_counts = df_sub["label"].value_counts().to_dict()

    print(f"\nSaved submission to: {output_path}")
    print(f"Submission SHA-256:  {sub_hash}")
    print(f"Class counts:        Class 0: {class_counts.get(0, 0)}, Class 1: {class_counts.get(1, 0)}")

    elapsed = time.time() - start_time
    print(f"Inference completed in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
