import hashlib
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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
TEST_META  = ROOT / "Test" / "test_metadata.csv"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
}

# Analyze every image for structural properties.
# Intensity statistics are sampled if there are many images.
INTENSITY_SAMPLE = 3000

np.random.seed(42)


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)


def find_images(directory):
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "mean": np.mean(values),
        "std": np.std(values),
        "min": np.min(values),
        "p01": np.percentile(values, 1),
        "p05": np.percentile(values, 5),
        "p25": np.percentile(values, 25),
        "median": np.percentile(values, 50),
        "p75": np.percentile(values, 75),
        "p95": np.percentile(values, 95),
        "p99": np.percentile(values, 99),
        "max": np.max(values),
    }


# ============================================================
# LOAD METADATA
# ============================================================

section("1. METADATA")

train = pd.read_csv(TRAIN_META)
test = pd.read_csv(TEST_META)

print("ROOT:")
print(ROOT)

print("\nTrain metadata:")
print("  Path :", TRAIN_META)
print("  Shape:", train.shape)
print("  Columns:", train.columns.tolist())

print("\nTest metadata:")
print("  Path :", TEST_META)
print("  Shape:", test.shape)
print("  Columns:", test.columns.tolist())


# ============================================================
# BASIC METADATA VALIDATION
# ============================================================

section("2. METADATA SANITY")

print("Train missing values:")
print(train.isna().sum())

print("\nTest missing values:")
print(test.isna().sum())

print("\nTrain duplicated image IDs:",
      train["image_id"].duplicated().sum())

print("Test duplicated image IDs:",
      test["image_id"].duplicated().sum())

print("\nTrain labels:")
print(train["label"].value_counts().sort_index())

print("\nTrain label proportions:")
print(train["label"].value_counts(normalize=True).sort_index())

print("\nUnique train azimuth values:",
      train["sun_azimuth_angle"].nunique())

print("Unique test azimuth values:",
      test["sun_azimuth_angle"].nunique())

print("\nTrain azimuth:")
print(train["sun_azimuth_angle"].describe())

print("\nTest azimuth:")
print(test["sun_azimuth_angle"].describe())


# ============================================================
# FIND IMAGE FILES
# ============================================================

section("3. EXACT IMAGE PATHS")

train_images = find_images(ROOT / "Train")
test_images = find_images(ROOT / "Test")

print("Train images found:", len(train_images))
print("Test images found :", len(test_images))

print("\nTRAIN SAMPLE PATHS:")
for p in train_images[:20]:
    print(" ", p.relative_to(ROOT))

print("\nTEST SAMPLE PATHS:")
for p in test_images[:20]:
    print(" ", p.relative_to(ROOT))


# ============================================================
# METADATA IDS VS IMAGE FILES
# ============================================================

section("4. METADATA ↔ IMAGE MATCHING")

train_image_names = {p.name for p in train_images}
test_image_names = {p.name for p in test_images}

train_ids = set(train["image_id"])
test_ids = set(test["image_id"])

missing_train_images = train_ids - train_image_names
missing_test_images = test_ids - test_image_names

unlisted_train_images = train_image_names - train_ids
unlisted_test_images = test_image_names - test_ids

print("Train metadata IDs:", len(train_ids))
print("Train image files :", len(train_image_names))

print("Missing train image files:",
      len(missing_train_images))

print("Extra train image files:",
      len(unlisted_train_images))

print("\nTest metadata IDs:", len(test_ids))
print("Test image files :", len(test_image_names))

print("Missing test image files:",
      len(missing_test_images))

print("Extra test image files:",
      len(unlisted_test_images))

if missing_train_images:
    print("\nExample missing train images:")
    print(list(missing_train_images)[:20])

if missing_test_images:
    print("\nExample missing test images:")
    print(list(missing_test_images)[:20])


# ============================================================
# IMAGE DIMENSIONS / MODES / FORMATS
# ============================================================

section("5. IMAGE DIMENSIONS / MODES / FORMATS")

def inspect_structure(images, name):

    dimensions = Counter()
    modes = Counter()
    formats = Counter()

    corrupt = []

    for p in images:
        try:
            with Image.open(p) as img:
                dimensions[img.size] += 1
                modes[img.mode] += 1
                formats[img.format] += 1

        except Exception as e:
            corrupt.append((str(p), repr(e)))

    print(f"\n{name}")

    print("\nDimensions:")
    for k, v in dimensions.most_common():
        print(f"  {k}: {v}")

    print("\nModes:")
    for k, v in modes.most_common():
        print(f"  {k}: {v}")

    print("\nFormats:")
    for k, v in formats.most_common():
        print(f"  {k}: {v}")

    print("\nCorrupt/unreadable:", len(corrupt))

    if corrupt:
        print("Examples:")
        for x in corrupt[:20]:
            print(" ", x)


inspect_structure(train_images, "TRAIN")
inspect_structure(test_images, "TEST")


# ============================================================
# INTENSITY / CONTRAST
# ============================================================

section("6. INTENSITY / CONTRAST")

def inspect_intensity(images, name):

    if len(images) > INTENSITY_SAMPLE:
        rng = np.random.default_rng(42)
        selected = list(
            rng.choice(images, INTENSITY_SAMPLE, replace=False)
        )
    else:
        selected = images

    means = []
    stds = []
    mins = []
    maxs = []
    medians = []
    dynamic_ranges = []
    dark_fraction = []
    bright_fraction = []

    failed = 0

    for p in selected:

        try:
            with Image.open(p) as img:
                arr = np.asarray(
                    img.convert("L"),
                    dtype=np.float32
                )

            means.append(arr.mean())
            stds.append(arr.std())
            mins.append(arr.min())
            maxs.append(arr.max())
            medians.append(np.median(arr))
            dynamic_ranges.append(arr.max() - arr.min())

            dark_fraction.append(np.mean(arr <= 25))
            bright_fraction.append(np.mean(arr >= 230))

        except Exception:
            failed += 1

    print(f"\n{name}")
    print("Sampled:", len(selected))
    print("Failed :", failed)

    stats = {
        "Mean intensity": means,
        "Std / contrast": stds,
        "Minimum": mins,
        "Median": medians,
        "Maximum": maxs,
        "Dynamic range": dynamic_ranges,
        "Fraction <= 25": dark_fraction,
        "Fraction >= 230": bright_fraction,
    }

    for label, values in stats.items():

        s = percentile_summary(values)

        print(f"\n{label}")
        print(f"  mean   = {s['mean']:.4f}")
        print(f"  std    = {s['std']:.4f}")
        print(f"  min    = {s['min']:.4f}")
        print(f"  p01    = {s['p01']:.4f}")
        print(f"  p05    = {s['p05']:.4f}")
        print(f"  p25    = {s['p25']:.4f}")
        print(f"  median = {s['median']:.4f}")
        print(f"  p75    = {s['p75']:.4f}")
        print(f"  p95    = {s['p95']:.4f}")
        print(f"  p99    = {s['p99']:.4f}")
        print(f"  max    = {s['max']:.4f}")


inspect_intensity(train_images, "TRAIN")
inspect_intensity(test_images, "TEST")


# ============================================================
# AZIMUTH DISTRIBUTION
# ============================================================

section("7. SOLAR AZIMUTH DISTRIBUTION")

def azimuth_bins(df, bin_size):

    edges = np.arange(
        0,
        360 + bin_size,
        bin_size
    )

    bins = pd.cut(
        df["sun_azimuth_angle"],
        bins=edges,
        right=False,
        include_lowest=True
    )

    counts = bins.value_counts().sort_index()

    return counts


for bin_size in [10, 30, 45, 60, 90]:

    print(f"\n--- {bin_size}° bins ---")

    counts_train = azimuth_bins(train, bin_size)
    counts_test = azimuth_bins(test, bin_size)

    result = pd.DataFrame({
        "train": counts_train,
        "test": counts_test
    })

    result["train_pct"] = (
        result["train"] / len(train) * 100
    )

    result["test_pct"] = (
        result["test"] / len(test) * 100
    )

    print(result.to_string())


# ============================================================
# CLASS VS AZIMUTH
# ============================================================

section("8. CLASS ↔ AZIMUTH RELATIONSHIP")

print("Azimuth statistics by class:")
print(
    train.groupby("label")["sun_azimuth_angle"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max"
    ])
    .to_string()
)

for bin_size in [10, 30, 45, 60, 90]:

    print(f"\n--- CLASS PROPORTION BY {bin_size}° AZIMUTH ---")

    edges = np.arange(
        0,
        360 + bin_size,
        bin_size
    )

    temp = train.copy()

    temp["az_bin"] = pd.cut(
        temp["sun_azimuth_angle"],
        bins=edges,
        right=False,
        include_lowest=True
    )

    table = pd.crosstab(
        temp["az_bin"],
        temp["label"],
        normalize="index"
    )

    print(table.round(3).to_string())


# ============================================================
# AZIMUTH-ONLY MODEL
# ============================================================

section("9. AZIMUTH-ONLY BASELINE")

theta = np.deg2rad(
    train["sun_azimuth_angle"].values
)

X = np.column_stack([
    np.sin(theta),
    np.cos(theta)
])

y = train["label"].values

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

az_model = LogisticRegression(
    max_iter=1000
)

oof = cross_val_predict(
    az_model,
    X,
    y,
    cv=cv,
    method="predict"
)

print("5-fold Balanced Accuracy:",
      balanced_accuracy_score(y, oof))

print("\nConfusion matrix:")
print(confusion_matrix(y, oof))


# ============================================================
# FILENAME STRUCTURE
# ============================================================

section("10. FILENAME / ORDER STRUCTURE")

print("First 30 train IDs:")
print(train["image_id"].head(30).tolist())

print("\nLast 30 train IDs:")
print(train["image_id"].tail(30).tolist())

print("\nFirst 30 test IDs:")
print(test["image_id"].head(30).tolist())

print("\nLast 30 test IDs:")
print(test["image_id"].tail(30).tolist())


def extract_numeric_id(series):

    values = []

    for x in series:

        m = re.search(r"(\d+)", str(x))

        if m:
            values.append(int(m.group(1)))

    return np.asarray(values)


train_nums = extract_numeric_id(train["image_id"])
test_nums = extract_numeric_id(test["image_id"])

print("\nTrain numeric ID range:",
      train_nums.min(), "→", train_nums.max())

print("Test numeric ID range:",
      test_nums.min(), "→", test_nums.max())

print("\nTrain IDs sequential:",
      np.array_equal(
          np.sort(train_nums),
          np.arange(1, len(train_nums) + 1)
      ))

print("Test IDs sequential:",
      np.array_equal(
          np.sort(test_nums),
          np.arange(1, len(test_nums) + 1)
      ))


# ============================================================
# LABEL VS IMAGE ID
# ============================================================

section("11. LABEL ↔ IMAGE ID ARTIFACT TEST")

train_id_num = train_nums

corr = np.corrcoef(
    train_id_num,
    train["label"].values
)[0, 1]

print("Correlation between numeric image ID and label:",
      corr)

# Label distribution in ID quartiles
train_tmp = train.copy()

train_tmp["numeric_id"] = train_id_num

train_tmp["id_quartile"] = pd.qcut(
    train_tmp["numeric_id"],
    4,
    duplicates="drop"
)

print("\nLabel distribution by ID quartile:")
print(
    pd.crosstab(
        train_tmp["id_quartile"],
        train_tmp["label"],
        normalize="index"
    ).round(3)
)


# ============================================================
# EXACT DUPLICATES
# ============================================================

section("12. EXACT DUPLICATE ANALYSIS")

def calculate_hashes(images):

    hashes = {}

    for p in images:

        try:
            h = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()

            hashes.setdefault(h, []).append(p)

        except Exception:
            pass

    return hashes


train_hashes = calculate_hashes(train_images)
test_hashes = calculate_hashes(test_images)

train_duplicate_groups = [
    paths for paths in train_hashes.values()
    if len(paths) > 1
]

test_duplicate_groups = [
    paths for paths in test_hashes.values()
    if len(paths) > 1
]

cross_duplicates = (
    set(train_hashes.keys())
    &
    set(test_hashes.keys())
)

print("Train unique hashes:",
      len(train_hashes))

print("Train duplicate groups:",
      len(train_duplicate_groups))

print("Test unique hashes:",
      len(test_hashes))

print("Test duplicate groups:",
      len(test_duplicate_groups))

print("Train ↔ Test exact duplicate hashes:",
      len(cross_duplicates))

if train_duplicate_groups:

    print("\nTrain duplicate examples:")

    for group in train_duplicate_groups[:10]:

        print("\nGROUP:")

        for p in group:
            print(" ", p.relative_to(ROOT))

if test_duplicate_groups:

    print("\nTest duplicate examples:")

    for group in test_duplicate_groups[:10]:

        print("\nGROUP:")

        for p in group:
            print(" ", p.relative_to(ROOT))

if cross_duplicates:

    print("\nWARNING: TRAIN/TEST EXACT DUPLICATES EXIST")

    for h in list(cross_duplicates)[:10]:

        print("\nHASH:", h)

        print("TRAIN:")
        for p in train_hashes[h]:
            print(" ", p.relative_to(ROOT))

        print("TEST:")
        for p in test_hashes[h]:
            print(" ", p.relative_to(ROOT))


# ============================================================
# IMAGE DIMENSION ARTIFACTS
# ============================================================

section("13. DIMENSION / FILESIZE ARTIFACTS")

def file_size_stats(images, name):

    sizes = np.array([
        p.stat().st_size
        for p in images
    ], dtype=np.float64)

    print(f"\n{name}")
    print("Mean bytes:", sizes.mean())
    print("Std bytes :", sizes.std())
    print("Min bytes :", sizes.min())
    print("Median    :", np.median(sizes))
    print("Max bytes :", sizes.max())

    return sizes


train_sizes = file_size_stats(
    train_images,
    "TRAIN"
)

test_sizes = file_size_stats(
    test_images,
    "TEST"
)


# ============================================================
# FILE SIZE VS LABEL
# ============================================================

section("14. FILE SIZE ↔ LABEL")

size_by_name = {
    p.name: p.stat().st_size
    for p in train_images
}

train["file_size"] = train["image_id"].map(
    size_by_name
)

print(
    train.groupby("label")["file_size"]
    .agg([
        "count",
        "mean",
        "std",
        "min",
        "median",
        "max"
    ])
    .to_string()
)

print(
    "\nFile-size correlation with label:",
    train[["file_size", "label"]]
    .corr()
    .iloc[0, 1]
)


# ============================================================
# AZIMUTH ↔ IMAGE STATISTICS
# ============================================================

section("15. AZIMUTH ↔ IMAGE STATISTICS")

# Sample images and correlate azimuth with image statistics.

rng = np.random.default_rng(42)

sample_df = train.copy()

if len(sample_df) > INTENSITY_SAMPLE:
    sample_df = sample_df.sample(
        INTENSITY_SAMPLE,
        random_state=42
    )

mean_by_id = {}
std_by_id = {}

for p in train_images:

    if p.name not in set(sample_df["image_id"]):
        continue

    try:
        with Image.open(p) as img:
            arr = np.asarray(
                img.convert("L"),
                dtype=np.float32
            )

        mean_by_id[p.name] = arr.mean()
        std_by_id[p.name] = arr.std()

    except:
        pass

sample_df["img_mean"] = sample_df["image_id"].map(mean_by_id)
sample_df["img_std"] = sample_df["image_id"].map(std_by_id)

print(
    sample_df[
        [
            "sun_azimuth_angle",
            "img_mean",
            "img_std"
        ]
    ].corr()
    .to_string()
)


# ============================================================
# TRAIN / TEST AZIMUTH SHIFT
# ============================================================

section("16. TRAIN ↔ TEST DISTRIBUTION SHIFT")

print("Train azimuth mean:",
      train["sun_azimuth_angle"].mean())

print("Test azimuth mean:",
      test["sun_azimuth_angle"].mean())

print("Train azimuth std:",
      train["sun_azimuth_angle"].std())

print("Test azimuth std:",
      test["sun_azimuth_angle"].std())


# ============================================================
# SUMMARY
# ============================================================

section("17. AUTOMATIC FLAGS")

flags = []

if len(missing_train_images) > 0:
    flags.append(
        "WARNING: train metadata references missing images"
    )

if len(missing_test_images) > 0:
    flags.append(
        "WARNING: test metadata references missing images"
    )

if len(cross_duplicates) > 0:
    flags.append(
        "WARNING: exact train/test duplicate images found"
    )

if len(train_duplicate_groups) > 0:
    flags.append(
        "NOTICE: duplicate images exist inside training data"
    )

if len(test_duplicate_groups) > 0:
    flags.append(
        "NOTICE: duplicate images exist inside test data"
    )

if train["label"].value_counts(normalize=True).min() < 0.35:
    flags.append(
        "NOTICE: meaningful class imbalance detected"
    )

az_ba = balanced_accuracy_score(y, oof)

if az_ba >= 0.60:
    flags.append(
        f"IMPORTANT: azimuth-only model has BA={az_ba:.4f}"
    )
else:
    flags.append(
        f"Azimuth-only model appears weak: BA={az_ba:.4f}"
    )

if abs(
    train["sun_azimuth_angle"].mean()
    -
    test["sun_azimuth_angle"].mean()
) > 10:
    flags.append(
        "NOTICE: substantial train/test azimuth mean shift"
    )

if not flags:
    flags.append("No automatic red flags detected.")

for flag in flags:
    print("-", flag)


print("\n")
print("=" * 90)
print("RECONNAISSANCE COMPLETE")
print("=" * 90)

