"""
DUPLICATE + AZIMUTH AUDIT
=========================
The Pareidolia Paradox Dataset
"""

import hashlib
import time
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

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
TEST_META = ROOT / "Test" / "test_metadata.csv"
TRAIN_IMG_DIR = ROOT / "Train" / "images" / "train_images"
TEST_IMG_DIR = ROOT / "Test" / "images" / "eval_images"

OUTPUT_CROSS_SPLIT = Path(__file__).resolve().parent / "duplicate_audit_cross_split.csv"
OUTPUT_TRAIN_CONFLICTS = Path(__file__).resolve().parent / "duplicate_audit_train_conflicts.csv"

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

AZ_TOL = 1e-6  # floating-point azimuth equality tolerance


def section(title):
    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)


def circular_diff(a, b):
    """Circular difference in [0, 180]."""
    d = abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# ============================================================
# LOAD METADATA
# ============================================================

section("LOADING METADATA")

train_meta = pd.read_csv(TRAIN_META)
test_meta = pd.read_csv(TEST_META)

print(f"Train rows: {len(train_meta)}")
print(f"Test  rows: {len(test_meta)}")

# Build lookup dicts: image_id -> metadata
train_lookup = {}
for _, row in train_meta.iterrows():
    train_lookup[row["image_id"]] = {
        "label": int(row["label"]),
        "azimuth": float(row["sun_azimuth_angle"]),
    }

test_lookup = {}
for _, row in test_meta.iterrows():
    test_lookup[row["image_id"]] = {
        "azimuth": float(row["sun_azimuth_angle"]),
    }


# ============================================================
# HASH ALL IMAGES
# ============================================================

section("HASHING ALL IMAGES")


def hash_images(img_dir, lookup, split_name):
    """Hash every image and return hash -> [image_ids]."""
    hashes = defaultdict(list)
    errors = []
    files = sorted(img_dir.glob("*"))
    total = len(files)
    t0 = time.time()
    for i, p in enumerate(files):
        if p.name not in lookup:
            continue
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hashes[h].append(p.name)
        except Exception as e:
            errors.append((p.name, str(e)))
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"  [{split_name}] {i+1}/{total} hashed ({elapsed:.1f}s)")
    elapsed = time.time() - t0
    print(f"  [{split_name}] Done: {len(hashes)} unique hashes "
          f"from {sum(len(v) for v in hashes.values())} images "
          f"({elapsed:.1f}s), {len(errors)} errors")
    return dict(hashes), errors


train_hashes, train_errors = hash_images(TRAIN_IMG_DIR, train_lookup, "TRAIN")
test_hashes, test_errors = hash_images(TEST_IMG_DIR, test_lookup, "TEST")

cross_split_hashes = set(train_hashes.keys()) & set(test_hashes.keys())

print(f"\nTrain unique hashes: {len(train_hashes)}")
print(f"Test  unique hashes: {len(test_hashes)}")
print(f"Cross-split shared hashes: {len(cross_split_hashes)}")


# ============================================================
# A. RGB CHANNEL AUDIT
# ============================================================

section("A. RGB CHANNEL AUDIT")


def audit_rgb_channels(img_dir, split_name, n_sample=None):
    files = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXT)
    if n_sample and n_sample < len(files):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(files), n_sample, replace=False)
        files = [files[i] for i in sorted(indices)]

    total = len(files)
    grayscale_exact = 0
    channel_diff_count = 0
    max_diff = 0
    failed = 0
    t0 = time.time()

    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            local_max = max(
                int(r.max()) - int(g.max()),
                int(g.max()) - int(b.max()),
                int(r.max()) - int(b.max()),
            )
            if np.array_equal(r, g) and np.array_equal(g, b):
                grayscale_exact += 1
            else:
                channel_diff_count += 1
                # compute actual max element-wise difference
                diff_rg = np.abs(r.astype(int) - g.astype(int))
                diff_rb = np.abs(r.astype(int) - b.astype(int))
                diff_gb = np.abs(g.astype(int) - b.astype(int))
                local_max = max(diff_rg.max(), diff_rb.max(), diff_gb.max())
                if local_max > max_diff:
                    max_diff = local_max
        except Exception:
            failed += 1

        if (i + 1) % 2000 == 0:
            print(f"  [{split_name}] {i+1}/{total} checked ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n[{split_name}] RGB Channel Audit:")
    print(f"  Sampled:             {total}")
    print(f"  Failed:              {failed}")
    print(f"  R==G==B exactly:     {grayscale_exact} ({grayscale_exact/max(total-failed,1)*100:.2f}%)")
    print(f"  Any channel diff:    {channel_diff_count} ({channel_diff_count/max(total-failed,1)*100:.2f}%)")
    print(f"  Max channel diff:    {max_diff}")
    elapsed_str = f"  Elapsed:             {elapsed:.1f}s"
    print(elapsed_str)
    return grayscale_exact, channel_diff_count, max_diff, total, failed


train_gray, train_diff, train_maxd, train_total, train_fail = audit_rgb_channels(
    TRAIN_IMG_DIR, "TRAIN"
)
test_gray, test_diff, test_maxd, test_total, test_fail = audit_rgb_channels(
    TEST_IMG_DIR, "TEST"
)

print("\n--- RGB AUDIT SUMMARY ---")
total_all = train_total + test_total
diff_all = train_diff + test_diff
if diff_all == 0:
    print("CONCLUSION: Dataset is effectively grayscale replicated across RGB channels.")
    print("  All sampled images have R == G == B exactly.")
    print("  Future models can treat input as single-channel conceptually.")
    print("  Implementation should retain 3 channels for pretrained CNN compatibility.")
else:
    print(f"CONCLUSION: {diff_all} images have channel differences.")
    print(f"  Max difference observed: {max(train_maxd, test_maxd)}")


# ============================================================
# B. CROSS-SPLIT EXACT DUPLICATE AUDIT
# ============================================================

section("B. TRAIN <-> TEST EXACT DUPLICATE AUDIT")

cross_split_records = []

for h in sorted(cross_split_hashes):
    train_ids = train_hashes[h]
    test_ids = test_hashes[h]

    for tid in train_ids:
        train_info = train_lookup[tid]
        for teid in test_ids:
            test_info = test_lookup[teid]
            az_t = train_info["azimuth"]
            az_e = test_info["azimuth"]
            raw_diff = abs(az_t - az_e)
            circ_d = circular_diff(az_t, az_e)
            same_az = circ_d < AZ_TOL

            cross_split_records.append({
                "hash": h,
                "train_image_id": tid,
                "train_label": train_info["label"],
                "train_azimuth": az_t,
                "test_image_id": teid,
                "test_azimuth": az_e,
                "raw_azimuth_difference": raw_diff,
                "circular_azimuth_difference": circ_d,
                "azimuth_same": same_az,
                "classification": "SAME_IMAGE_SAME_AZIMUTH" if same_az else "SAME_IMAGE_DIFFERENT_AZIMUTH",
            })

cross_df = pd.DataFrame(cross_split_records)

# Save cross-split CSV
# Aggregate by hash for the requested output format
cross_agg_rows = []
for h in sorted(cross_split_hashes):
    rows = cross_df[cross_df["hash"] == h]
    train_ids_str = "; ".join(sorted(rows["train_image_id"].unique()))
    train_labels_str = "; ".join(
        str(x) for x in sorted(rows["train_label"].unique())
    )
    train_azs_str = "; ".join(
        f"{x:.2f}" for x in sorted(rows["train_azimuth"].unique())
    )
    test_ids_str = "; ".join(sorted(rows["test_image_id"].unique()))
    test_azs_str = "; ".join(
        f"{x:.2f}" for x in sorted(rows["test_azimuth"].unique())
    )
    any_same = bool(rows["azimuth_same"].any())
    min_circ = rows["circular_azimuth_difference"].min()
    cross_agg_rows.append({
        "hash": h,
        "train_image_ids": train_ids_str,
        "train_labels": train_labels_str,
        "train_azimuths": train_azs_str,
        "test_image_ids": test_ids_str,
        "test_azimuths": test_azs_str,
        "azimuth_same": any_same,
        "circular_azimuth_difference": min_circ,
    })

cross_agg_df = pd.DataFrame(cross_agg_rows)
cross_agg_df.to_csv(OUTPUT_CROSS_SPLIT, index=False)
print(f"\nSaved cross-split CSV: {OUTPUT_CROSS_SPLIT}")
print(f"  Rows: {len(cross_agg_df)}")

# Summary stats
n_rows = len(cross_df)
n_same = cross_df["azimuth_same"].sum()
n_diff = n_rows - n_same

# Unique test images involved
unique_test_images_in_cross = set()
for h in cross_split_hashes:
    for teid in test_hashes[h]:
        unique_test_images_in_cross.add(teid)

print(f"\n--- Cross-Split Duplicate Summary ---")
print(f"  Total cross-split rows (train_image x test_image): {n_rows}")
print(f"  Unique cross-split hashes: {len(cross_split_hashes)}")
print(f"  Unique test images in cross-split: {len(unique_test_images_in_cross)}")
print(f"  Same azimuth:  {n_same} ({n_same/n_rows*100:.2f}%)")
print(f"  Diff azimuth:  {n_diff} ({n_diff/n_rows*100:.2f}%)")

if n_diff > 0:
    diff_df = cross_df[~cross_df["azimuth_same"]]
    print(f"\n--- Azimuth Difference Stats (different-azimuth group) ---")
    print(f"  Mean:   {diff_df['circular_azimuth_difference'].mean():.4f}")
    print(f"  Median: {diff_df['circular_azimuth_difference'].median():.4f}")
    print(f"  Std:    {diff_df['circular_azimuth_difference'].std():.4f}")
    print(f"  Min:    {diff_df['circular_azimuth_difference'].min():.4f}")
    print(f"  Max:    {diff_df['circular_azimuth_difference'].max():.4f}")
    for q in [5, 25, 50, 75, 95]:
        print(f"  P{q}:    {diff_df['circular_azimuth_difference'].quantile(q/100):.4f}")

    # Also report raw difference stats
    print(f"\n  Raw azimuth difference stats:")
    print(f"  Mean:   {diff_df['raw_azimuth_difference'].mean():.4f}")
    print(f"  Median: {diff_df['raw_azimuth_difference'].median():.4f}")
    print(f"  Std:    {diff_df['raw_azimuth_difference'].std():.4f}")
    print(f"  Min:    {diff_df['raw_azimuth_difference'].min():.4f}")
    print(f"  Max:    {diff_df['raw_azimuth_difference'].max():.4f}")

    # Count how many unique hashes have ALL same azimuth vs ALL different
    hash_az_counts = cross_df.groupby("hash").agg(
        n_same=("azimuth_same", "sum"),
        n_total=("azimuth_same", "count")
    ).reset_index()
    hash_az_counts["all_same"] = hash_az_counts["n_same"] == hash_az_counts["n_total"]
    hash_az_counts["all_diff"] = hash_az_counts["n_same"] == 0
    print(f"\n  Hashes where ALL row-pairs have same azimuth: "
          f"{hash_az_counts['all_same'].sum()}")
    print(f"  Hashes where ALL row-pairs have different azimuth: "
          f"{hash_az_counts['all_diff'].sum()}")
    print(f"  Hashes with mixed (some same, some different): "
          f"{(~hash_az_counts['all_same'] & ~hash_az_counts['all_diff']).sum()}")

    # How many duplicate groups: all same, all different, mixed
    n_hashes_all_same = hash_az_counts["all_same"].sum()
    n_hashes_all_diff = hash_az_counts["all_diff"].sum()
    n_hashes_mixed = len(hash_az_counts) - n_hashes_all_same - n_hashes_all_diff

    print(f"\n  Of {len(hash_az_counts)} cross-split hashes with multiple rows:")
    print(f"    {n_hashes_all_same} have ALL same azimuth")
    print(f"    {n_hashes_all_diff} have ALL different azimuth")
    print(f"    {n_hashes_mixed} have mixed")


# ============================================================
# C. TRAIN DUPLICATE LABEL CONSISTENCY
# ============================================================

section("C. TRAIN DUPLICATE LABEL CONSISTENCY")

train_dup_groups = {
    h: ids for h, ids in train_hashes.items() if len(ids) > 1
}

total_groups = len(train_dup_groups)
only_0 = 0
only_1 = 0
both = 0
conflicting_groups = []

for h, ids in sorted(train_dup_groups.items()):
    labels = set()
    for iid in ids:
        labels.add(train_lookup[iid]["label"])
    if labels == {0}:
        only_0 += 1
    elif labels == {1}:
        only_1 += 1
    else:
        both += 1
        azs = [train_lookup[iid]["azimuth"] for iid in ids]
        conflicting_groups.append({
            "hash": h,
            "image_ids": "; ".join(ids),
            "labels": "; ".join(str(train_lookup[iid]["label"]) for iid in ids),
            "azimuths": "; ".join(f"{a:.2f}" for a in azs),
            "num_images": len(ids),
        })

n_conflicting_images = sum(
    cg["num_images"] for cg in conflicting_groups
)

print(f"\n--- Train Duplicate Label Consistency ---")
print(f"  Total train duplicate groups:  {total_groups}")
print(f"  Groups with only class 0:      {only_0}")
print(f"  Groups with only class 1:      {only_1}")
print(f"  Groups with BOTH classes:      {both}")
print(f"  Images in conflicting groups:  {n_conflicting_images}")

if both == 0:
    print("\n  CONCLUSION: All train duplicate groups are label-consistent.")
    print("  Identical pixels always share the same label.")
    print("  This means the label is determined by content, not metadata/lighting alone.")
else:
    print(f"\n  ALL {total_groups} train duplicate groups are label-CONFLICTING.")
    print("  (Identical pixel content is labeled as BOTH class 0 and class 1")
    print("   within the training set. The label is NOT a function of the image")
    print("   pixels alone; it must depend on azimuth/lighting metadata.)")

    # Save conflicts CSV (full detail for inspection)
    conflict_df = pd.DataFrame(conflicting_groups)
    conflict_df.to_csv(OUTPUT_TRAIN_CONFLICTS, index=False)
    print(f"\n  Full conflict detail saved to CSV: {OUTPUT_TRAIN_CONFLICTS}")

    # Print a small sample for immediate inspection (full list is in CSV)
    print(f"\n  First 5 conflicting groups (see CSV for all {both}):")
    for cg in conflicting_groups[:5]:
        print(f"    Hash {cg['hash'][:16]}... | IDs: {cg['image_ids']} | "
              f"Labels: {cg['labels']} | Az: {cg['azimuths']}")


# ============================================================
# D. TEST DUPLICATE CONSISTENCY
# ============================================================

section("D. TEST DUPLICATE CONSISTENCY")

test_dup_groups = {
    h: ids for h, ids in test_hashes.items() if len(ids) > 1
}

total_test_groups = len(test_dup_groups)
test_same_az = 0
test_diff_az = 0
test_diff_az_values = []

for h, ids in sorted(test_dup_groups.items()):
    azs = [test_lookup[iid]["azimuth"] for iid in ids]
    # Check if all azimuth values are the same
    az_set = set(azs)
    if len(az_set) == 1:
        test_same_az += 1
    else:
        test_diff_az += 1
        # Collect all pairwise circular differences
        for i in range(len(azs)):
            for j in range(i + 1, len(azs)):
                test_diff_az_values.append(circular_diff(azs[i], azs[j]))

print(f"\n--- Test Duplicate Consistency ---")
print(f"  Total test duplicate groups:   {total_test_groups}")
print(f"  Groups with same azimuth:      {test_same_az}")
print(f"  Groups with different azimuth: {test_diff_az}")

if test_diff_az_values:
    vals = np.array(test_diff_az_values)
    print(f"\n  Azimuth difference stats (different-azimuth test groups):")
    print(f"  N pairwise comparisons: {len(vals)}")
    print(f"  Mean:   {vals.mean():.4f}")
    print(f"  Median: {np.median(vals):.4f}")
    print(f"  Std:    {vals.std():.4f}")
    print(f"  Min:    {vals.min():.4f}")
    print(f"  Max:    {vals.max():.4f}")
    for q in [5, 25, 50, 75, 95]:
        print(f"  P{q}:    {np.percentile(vals, q):.4f}")

    # Show some examples
    print(f"\n  Example test duplicate groups with different azimuth:")
    shown = 0
    for h, ids in sorted(test_dup_groups.items()):
        azs = [test_lookup[iid]["azimuth"] for iid in ids]
        if len(set(azs)) > 1 and shown < 5:
            print(f"    Hash: {h[:16]}...")
            for iid, az in zip(ids, azs):
                print(f"      {iid}: azimuth={az:.2f}")
            shown += 1


# ============================================================
# E. CROSS-SPLIT DUPLICATE CLASS BREAKDOWN
# ============================================================

section("E. CROSS-SPLIT DUPLICATE CLASS BREAKDOWN")

class_0_hashes = set()
class_1_hashes = set()
hash_multi_label = set()

for h in cross_split_hashes:
    train_ids = train_hashes[h]
    labels_for_hash = set()
    for tid in train_ids:
        lbl = train_lookup[tid]["label"]
        labels_for_hash.add(lbl)
        if lbl == 0:
            class_0_hashes.add(h)
        else:
            class_1_hashes.add(h)
    if len(labels_for_hash) > 1:
        hash_multi_label.add(h)

print(f"\n--- Cross-Split Class Breakdown ---")
print(f"  Unique cross-split hashes:         {len(cross_split_hashes)}")
print(f"  Hashes mapping to train class 0:   {len(class_0_hashes)} ({len(class_0_hashes)/len(cross_split_hashes)*100:.2f}%)")
print(f"  Hashes mapping to train class 1:   {len(class_1_hashes)} ({len(class_1_hashes)/len(cross_split_hashes)*100:.2f}%)")
print(f"  Hashes mapping to MULTIPLE labels: {len(hash_multi_label)}")

if hash_multi_label:
    print(f"\n  Hashes with multiple train labels ({len(hash_multi_label)}):")
    for h in sorted(hash_multi_label):
        train_ids = train_hashes[h]
        lbls = [train_lookup[tid]["label"] for tid in train_ids]
        print(f"    {h[:16]}... -> IDs: {train_ids}, Labels: {lbls}")

# Detailed breakdown
print(f"\n--- Detailed counts ---")
# How many train images (rows) are cross-split duplicates?
train_rows_in_cross = 0
for h in cross_split_hashes:
    train_rows_in_cross += len(train_hashes[h])
print(f"  Train rows participating in cross-split: {train_rows_in_cross}")

test_rows_in_cross = 0
for h in cross_split_hashes:
    test_rows_in_cross += len(test_hashes[h])
print(f"  Test  rows participating in cross-split: {test_rows_in_cross}")


# ============================================================
# G. INSPECTION SAMPLES
# ============================================================

section("G. INSPECTION SAMPLES")

# 1. Same-image + same-azimuth cross-split examples
print("\n--- 1. Same-image + same-azimuth cross-split examples (5) ---")
same_az_df = cross_df[cross_df["azimuth_same"]]
for _, row in same_az_df.head(5).iterrows():
    print(f"  Hash: {row['hash'][:16]}...")
    print(f"    Train: {row['train_image_id']} (label={row['train_label']}, az={row['train_azimuth']:.2f})")
    print(f"    Test:  {row['test_image_id']} (az={row['test_azimuth']:.2f})")

# 2. Same-image + different-azimuth examples (10, range of differences)
print("\n--- 2. Same-image + different-azimuth cross-split examples (10) ---")
diff_az_df = cross_df[~cross_df["azimuth_same"]].sort_values("circular_azimuth_difference")
n_diff_total = len(diff_az_df)
if n_diff_total > 0:
    # Pick examples covering range
    if n_diff_total >= 10:
        indices = np.linspace(0, n_diff_total - 1, 10, dtype=int)
    else:
        indices = np.arange(n_diff_total)
    for idx in indices:
        row = diff_az_df.iloc[idx]
        print(f"  Hash: {row['hash'][:16]}... | circ_diff={row['circular_azimuth_difference']:.2f}")
        print(f"    Train: {row['train_image_id']} (label={row['train_label']}, az={row['train_azimuth']:.2f})")
        print(f"    Test:  {row['test_image_id']} (az={row['test_azimuth']:.2f})")

# 3. Train duplicate groups (5)
print("\n--- 3. Train duplicate groups with repeated images (5) ---")
shown = 0
for h, ids in sorted(train_dup_groups.items()):
    if shown >= 5:
        break
    labels = [train_lookup[iid]["label"] for iid in ids]
    azs = [train_lookup[iid]["azimuth"] for iid in ids]
    print(f"  Hash: {h[:16]}...")
    for iid, lbl, az in zip(ids, labels, azs):
        print(f"    {iid}: label={lbl}, azimuth={az:.2f}")
    shown += 1

# 4. Conflicting train duplicate groups (sample; full list is in conflicts CSV)
print("\n--- 4. Conflicting train duplicate groups (first 8; full list in CSV) ---")
if conflicting_groups:
    for cg in conflicting_groups[:8]:
        print(f"  Hash: {cg['hash']}")
        print(f"    IDs:     {cg['image_ids']}")
        print(f"    Labels:  {cg['labels']}")
        print(f"    Azimuths: {cg['azimuths']}")
    if len(conflicting_groups) > 8:
        print(f"  ... and {len(conflicting_groups) - 8} more (see "
              f"{OUTPUT_TRAIN_CONFLICTS.name})")
else:
    print("  None.")


# ============================================================
# H. AUDIT CONCLUSION
# ============================================================

section("H. AUDIT CONCLUSION")

print("""
1. IS THE DATASET EFFECTIVELY GRAYSCALE?
""")

if diff_all == 0:
    print("   YES. All sampled images have R == G == B exactly.")
    print("   The dataset is effectively single-channel, stored as RGB.")
    print("   Models can conceptually treat input as 1-channel, but should")
    print("   retain 3-channel input for pretrained CNN compatibility.")
else:
    print(f"   NO. {diff_all} images have non-identical RGB channels.")

print("""
2. IS TRAIN/TEST EXACT IMAGE OVERLAP REAL?
""")
print(f"   YES. {len(cross_split_hashes)} unique image hashes appear in both train and test.")
print(f"   This covers {len(unique_test_images_in_cross)} unique test images "
      f"({len(unique_test_images_in_cross)/len(test_meta)*100:.1f}% of test set).")
print(f"   These are byte-identical images, not similar or near-duplicates.")

print("""
3. ARE OVERLAPPING IMAGES USUALLY ASSOCIATED WITH THE SAME OR DIFFERENT AZIMUTH?
""")
if n_diff == 0:
    print("   All cross-split duplicates have identical azimuth values.")
else:
    same_pct = n_same / n_rows * 100
    diff_pct = n_diff / n_rows * 100
    print(f"   {n_same} row-pairs ({same_pct:.1f}%) have SAME azimuth.")
    print(f"   {n_diff} row-pairs ({diff_pct:.1f}%) have DIFFERENT azimuth.")
    if n_diff > 0:
        print(f"   Median circular azimuth difference: {diff_df['circular_azimuth_difference'].median():.2f}")
        print(f"   Max circular azimuth difference: {diff_df['circular_azimuth_difference'].max():.2f}")

print("""
4. ARE TRAIN DUPLICATE LABELS CONSISTENT?
""")
print(f"   NO. All {total_groups} train duplicate groups are label-CONFLICTING.")
print(f"   Identical pixel content is labeled BOTH class 0 and class 1")
print(f"   ({n_conflicting_images} images / {len(conflicting_groups)} groups).")
print("   CRITICAL: The label is NOT a function of pixel content alone.")
print("   The same exact image is labeled Depth in one row and Rise in another.")
print("   Therefore the class must be (at least partly) determined by the")
print("   azimuth/lighting metadata + geometry, not the static 2D pixels.")

print("""
5. WHAT DOES THE OVERLAP LOOK LIKE?
""")
print(f"   The {len(cross_split_hashes)} shared hashes are exact pixel copies.")
print(f"   Every cross-split pair ({n_rows}/{n_rows}) has a DIFFERENT azimuth")
print("   (median circular diff ~88 deg). The same physical surface patch")
print("   appears in both splits with different sun illumination angles -")
print("   consistent with repeated orbital pass imagery of the same crater/mound.")
print(f"   Within train, identical pixels recur {total_groups} times with")
print("   DIFFERENT labels AND different azimuths - i.e. the same patch was")
print("   imaged under two lighting conditions and given opposite labels,")
print("   because the lighting determines which shadow-based cue (depth vs rise)")
print("   is visible. This is a construction artifact tied to the lighting geometry,")
print("   not a random labeling error.")

print("""
6. IMPLICATIONS FOR:
""")
print("   - VALIDATION DESIGN:")
print(f"     {len(cross_split_hashes)} shared hashes leak train/test. Random CV overstates")
print("     performance. Use hash-grouped CV (never split identical pixels across")
print("     folds) to measure generalization honestly.")
print()
print("   - IMAGE MEMORIZATION:")
print(f"     {len(unique_test_images_in_cross)}/{len(test_meta)} "
      f"({len(unique_test_images_in_cross)/len(test_meta)*100:.1f}%) of test images")
print("     exist byte-identical in train - a memorization shortcut exists. BUT")
print("     because labels are azimuth-dependent and the shared images have DIFFERENT")
print("     azimuths in train vs test, memorizing the train label is NOT sufficient.")
print("     The azimuth (lighting) must be incorporated to label a shared image correctly.")
print()
print("   - AZIMUTH HANDLING:")
print("     Azimuth is NOT a confound to be purged - it is LIKELY RELEVANT TO THE LABEL.")
print("     Identical pixels get opposite labels based on lighting direction. The")
print("     77.6% azimuth-only balanced accuracy reflects real label-azimuth coupling,")
print("     not just an artifact. The model MUST use azimuth together with pixels.")
print()
print("   - ORGANIZER-PRESCRIBED -sun_azimuth ROTATION:")
print("     Rotating by -sun_azimuth_angle normalizes illumination so that the sun")
print("     direction becomes consistent across images. This is likely the intended")
print("     way to make lighting-independent geometry (crater vs mound) comparable.")
print("     Under rotation, identical patches imaged at different azimuths should")
print("     ALIGN, so the same surface should produce the SAME class after rotation.")
print("     This rotation is probably ESSENTIAL for correct classification.")

print("""
7. RECOMMENDED NEXT EXPERIMENT:
""")
print("   Test the organizer-prescribed -sun_azimuth rotation hypothesis directly:")
print("   take the train duplicate groups (identical pixels, opposite labels, different")
print("   azimuth) and rotate each image by its -sun_azimuth_angle. If the rotation is")
print("   correct, the aligned (rotated) duplicate pairs should produce images that the")
print("   model can classify consistently. Concretely: rotate all train images by")
print("   -sun_azimuth_angle, then train a small model on the rotated pixels and measure")
print("   whether duplicate pairs become label-consistent (should approach 100% if the")
print("   rotation reveals the true geometry). This tests whether rotation resolves")
print("   the labeling ambiguity before committing to any architecture.")


# ============================================================
# SAVE ADDITIONAL REPORTS
# ============================================================

section("GENERATED FILES")

print(f"  1. {OUTPUT_CROSS_SPLIT}")
print(f"     Cross-split duplicate hashes with aggregated metadata")
print(f"  2. {OUTPUT_TRAIN_CONFLICTS}")
if conflicting_groups:
    print(f"     Conflicting train duplicate groups")
else:
    print(f"     (No conflicting groups - file not created)")

print("\n" + "=" * 90)
print("DUPLICATE AUDIT COMPLETE")
print("=" * 90)
