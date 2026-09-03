"""
AZIMUTH CANONICALIZATION / DUPLICATE ALIGNMENT AUDIT
=====================================================
The Pareidolia Paradox Dataset

Goal: determine whether applying the organizer-prescribed transform

    rotated = rotate(image, -sun_azimuth_angle)

makes pixel-identical images that are labeled differently become
geometrically aligned in a way that explains the label difference.

IMPORTANT: We are TESTING this hypothesis, not assuming it.
No CNN / pretrained model / leaderboard / synthetic training data.
"""

import time
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import scipy

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

OUT_DIR = Path(__file__).resolve().parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

OUT_PAIR_METRICS = Path(__file__).resolve().parent / "azimuth_rotation_pair_metrics.csv"
OUT_CROSS_SPLIT = Path(__file__).resolve().parent / "azimuth_rotation_cross_split.csv"

AZ_TOL = 1e-6
np.random.seed(42)


def section(title):
    print("\n")
    print("=" * 90)
    print(title)
    print("=" * 90)


def circular_diff(a, b):
    d = abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


# ============================================================
# ROTATION INTERFACE  (Section 1)
# ============================================================
#
# Convention documentation:
#   Library   : PIL (Pillow)
#   Function  : Image.rotate(angle, resample=Image.BILINEAR, expand=False)
#   Sign      : PIL positive angle = counter-clockwise (verified empirically;
#               rotate(+90) moves the top edge toward the left).
#   Amount    : rotated = rotate(image, -sun_azimuth_angle)
#               Because azimuth is non-negative, -azimuth is <= 0, i.e. a
#               CLOCKWISE rotation by |azimuth| degrees under PIL's
#               CCW-positive convention. This implements the organizer's
#               operational instruction "rotate by -sun_azimuth_angle".
#   expand    : False  (output same 256x256 size; content rotated about the
#               center; corners fold in / background shows at the corners)
#   Interp    : BILINEAR (deterministic)
#   Fill      : PIL default (0 = black for L mode at the exposed corners)
#   No manual cropping : since expand=False, PIL crops back to the canvas
#               automatically (as PIPElines to the same 256x256). We do not
#               do any additional manual crop.
#
# We use PIL's native rotate for ALL conventions so that any comparison is
# apples-to-apples; only the ANGLE changes between conventions.
# ============================================================

def rotate_array(arr_u8, azimuth_deg):
    """Rotate a uint8 grayscale 2-D array by -azimuth (official convention).

    arr_u8: (H, W) uint8 array. Returns (H, W) uint8 array.
    """
    img = Image.fromarray(arr_u8, mode="L")
    angle = -float(azimuth_deg)          # official: rotate by -azimuth
    out = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    return np.asarray(out, dtype=np.uint8)


def rotate_convention(arr_u8, azimuth_deg, convention):
    """Apply a given convention's angle to the raw array."""
    az = float(azimuth_deg)
    if convention == "A":   # official: -azimuth
        angle = -az
    elif convention == "B":  # +azimuth (control)
        angle = az
    elif convention == "C":  # -(azimuth-90) (control)
        angle = -(az - 90.0)
    elif convention == "D":  # +(azimuth-90) (control)
        angle = az - 90.0
    else:
        raise ValueError(f"unknown convention {convention}")
    img = Image.fromarray(arr_u8, mode="L")
    return np.asarray(img.rotate(angle, resample=Image.BILINEAR, expand=False),
                      dtype=np.uint8)


# ============================================================
# SIMILARITY METRICS
# ============================================================

def to_float(arr):
    return arr.astype(np.float32)


def mae(a, b):
    return float(np.mean(np.abs(to_float(a) - to_float(b))))


def pearson(a, b):
    fa = to_float(a).ravel()
    fb = to_float(b).ravel()
    if fa.std() == 0 or fb.std() == 0:
        return float("nan")
    return float(np.corrcoef(fa, fb)[0, 1])


def best_flip_relation(base, other):
    """Among identity/hflip/vflip/rot180 of `other`, return the Pearson
    correlation that best matches `base`, plus the winning transform."""
    H, W = other.shape
    transforms = {
        "identity": other,
        "hflip": other[:, ::-1],
        "vflip": other[::-1, :],
        "rot180": other[::-1, ::-1],
    }
    best_name = None
    best_corr = -np.inf
    for name, t in transforms.items():
        c = pearson(base, t)
        if not np.isnan(c) and c > best_corr:
            best_corr = c
            best_name = name
    return best_name, best_corr


# ============================================================
# LOAD METADATA + HASH IMAGES
# ============================================================

section("LOADING METADATA + HASHING")

train_meta = pd.read_csv(TRAIN_META)
test_meta = pd.read_csv(TEST_META)

train_lookup = {}
for _, r in train_meta.iterrows():
    train_lookup[r["image_id"]] = {
        "label": int(r["label"]),
        "azimuth": float(r["sun_azimuth_angle"]),
    }

test_lookup = {}
for _, r in test_meta.iterrows():
    test_lookup[r["image_id"]] = {"azimuth": float(r["sun_azimuth_angle"])}


def hash_images(img_dir, lookup):
    hashes = defaultdict(list)
    for p in sorted(img_dir.glob("*")):
        if p.name not in lookup:
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        hashes[h].append(p.name)
    return dict(hashes)


train_hashes = hash_images(TRAIN_IMG_DIR, train_lookup)
test_hashes = hash_images(TEST_IMG_DIR, test_lookup)
print(f"Train unique hashes: {len(train_hashes)}")
print(f"Test  unique hashes: {len(test_hashes)}")

cross_hashes = set(train_hashes.keys()) & set(test_hashes.keys())
print(f"Cross-split shared hashes: {len(cross_hashes)}")

# Load image pixels lazily: only for ids we need, cache by hash.
_pixel_cache = {}


def load_pixels_by_hash(h):
    if h not in _pixel_cache:
        # any image file in this hash group has identical bytes
        img_id = train_hashes.get(h, test_hashes.get(h, []))[0]
        img = Image.open(_path_for(img_id)).convert("L")
        _pixel_cache[h] = np.asarray(img, dtype=np.uint8)
    return _pixel_cache[h]


def _path_for(image_id):
    if image_id in train_lookup:
        return TRAIN_IMG_DIR / image_id
    if image_id in test_lookup:
        return TEST_IMG_DIR / image_id
    raise ValueError(f"unknown id {image_id}")


# ============================================================
# SECTION 2/3/4/5: TRAIN CONFLICTING PAIRS
# ============================================================

section("A. TRAIN CONFLICTING PAIR ANALYSIS (1458 groups)")

CONVENTIONS = ["A", "B", "C", "D"]

train_dup_groups = {h: ids for h, ids in train_hashes.items() if len(ids) > 1}

conflicting = []   # (hash, class0_id, az0, class1_id, az1)
for h, ids in sorted(train_dup_groups.items()):
    c0 = [i for i in ids if train_lookup[i]["label"] == 0]
    c1 = [i for i in ids if train_lookup[i]["label"] == 1]
    # briefly: all groups are 2-image groups per prior audit (num_images=2)
    if c0 and c1:
        conflicting.append((h, c0[0], c1[0]))

print(f"Conflicting train pairs: {len(conflicting)}")

pair_records = []
flip_winner_counter = defaultdict(int)

t0 = time.time()
for idx, (h, id0, id1) in enumerate(conflicting):
    az0 = train_lookup[id0]["azimuth"]
    az1 = train_lookup[id1]["azimuth"]
    circ = circular_diff(az0, az1)

    arr = load_pixels_by_hash(h)   # identical raw pixels

    # raw metrics (identical -> MAE 0, corr 1) recorded for completeness
    raw_mae = mae(arr, arr)
    raw_corr = 1.0

    rec = {
        "hash": h,
        "class0_image_id": id0,
        "class0_azimuth": az0,
        "class1_image_id": id1,
        "class1_azimuth": az1,
        "circular_azimuth_difference": circ,
        "raw_mae": raw_mae,
    }

    for conv in CONVENTIONS:
        r0 = rotate_convention(arr, az0, conv)
        r1 = rotate_convention(arr, az1, conv)
        m = mae(r0, r1)
        c = pearson(r0, r1)
        rec[f"{conv}_mae"] = m
        rec[f"{conv}_corr"] = c

    # best flip/rotation for official convention (A)
    r0A = rotate_convention(arr, az0, "A")
    r1A = rotate_convention(arr, az1, "A")
    best_name, best_sim = best_flip_relation(r0A, r1A)
    rec["best_flip_or_rotation"] = best_name
    rec["best_similarity"] = best_sim
    flip_winner_counter[best_name] += 1

    pair_records.append(rec)

    if (idx + 1) % 300 == 0:
        print(f"  [{idx+1}/{len(conflicting)}] "
              f"elapsed {time.time()-t0:.1f}s")

print(f"  Done processing {len(conflicting)} pairs in {time.time()-t0:.1f}s")

pair_df = pd.DataFrame(pair_records)
pair_df.to_csv(OUT_PAIR_METRICS, index=False)
print(f"Saved pair metrics: {OUT_PAIR_METRICS} ({len(pair_df)} rows)")

# ---- distribution comparison across conventions ----
def summarize_metric(series):
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    q = np.percentile(s, [10, 25, 50, 75, 90])
    return {
        "mean": s.mean(), "median": q[2],
        "p10": q[0], "p25": q[1], "p75": q[3], "p90": q[4],
    }

print("\n==== Convention comparison (over %d conflicting pairs) ====" % len(pair_df))
for conv in CONVENTIONS:
    ms = summarize_metric(pair_df[f"{conv}_corr"])
    print(f"\n  Convention {conv}  (corr)")
    for k, v in ms.items():
        print(f"    {k:6s} = {v:.4f}")

print("\n  Best-flip winner counts (official convention A):")
for k, v in flip_winner_counter.items():
    print(f"    {k:10s}: {v}  ({v/len(pair_df)*100:.1f}%)")

# ============================================================
# SECTION 6: VISUAL INSPECTION GRIDS
# ============================================================

section("B. VISUAL INSPECTION GRIDS")

# Select grid samples spanning azimuth-difference buckets.
grid_targets = [
    ("small", lambda c: c <= 15),
    ("45",    lambda c: 30 <= c <= 60),
    ("90",    lambda c: 70 <= c <= 110),
    ("135",   lambda c: 115 <= c <= 155),
    ("180",   lambda c: c > 155),
]
grid_selected = []
used = set()
rng = np.random.default_rng(42)
pool = [r for r in pair_records]
for name, pred in grid_targets:
    cands = [r for r in pool if pred(r["circular_azimuth_difference"])
             and r["hash"] not in used]
    if not cands:
        print(f"  No candidates for bucket {name}")
        continue
    # take up to 8 per bucket to meet 'at least 30' total
    picked = rng.choice(cands, size=min(8, len(cands)), replace=False)
    for r in picked:
        grid_selected.append(r)
        used.add(r["hash"])

# ensure >= 30 (top up with any remaining)
need = 30 - len(grid_selected)
if need > 0:
    for r in pool:
        if r["hash"] not in used:
            grid_selected.append(r)
            used.add(r["hash"])
        if len(grid_selected) >= 30:
            break

print(f"Selected {len(grid_selected)} pairs for visual grids")

PANEL = 256
PAD = 8
LABEL_H = 40


def render_pair_grid(rec):
    h = rec["hash"]
    arr = load_pixels_by_hash(h)
    az0 = rec["class0_azimuth"]
    az1 = rec["class1_azimuth"]
    id0 = rec["class0_image_id"]
    id1 = rec["class1_image_id"]

    rawL = arr
    rot0 = rotate_array(arr, az0)   # official for class0
    rot1 = rotate_array(arr, az1)   # official for class1
    diff = np.abs(to_float(rot0) - to_float(rot1))

    # columns: raw, class0-rotated, class1-rotated, difference/best-sim viz
    ncols = 4
    W = ncols * PANEL + (ncols + 1) * PAD
    H = PANEL + LABEL_H + 2 * PAD

    canvas = Image.new("L", (W, H), 255)
    draw = None  # we'll compose with numpy + PIL text is complex; instead overlay title with PIL text.

    panels = []

    def pil_from(arr_u8):
        return Image.fromarray(arr_u8, mode="L")

    # normalize diff to 0-255 for viewing
    dmin, dmax = diff.min(), diff.max()
    dvis = np.zeros_like(diff, dtype=np.uint8)
    if dmax > dmin:
        dvis = (255 * (diff - dmin) / (dmax - dmin)).astype(np.uint8)

    show_panels = [
        ("RAW", rawL),
        (f"class0 rot -az\n{id0} az={az0:.1f} lab=0", rot0),
        (f"class1 rot -az\n{id1} az={az1:.1f} lab=1", rot1),
        (f"|diff| MAE={rec['official_rotated_mae'] if 'official_rotated_mae' in rec else rec['A_mae']:.1f}", dvis),
    ]

    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for col, (title, panel_arr) in enumerate(show_panels):
        x = PAD + col * (PANEL + PAD)
        y = PAD + LABEL_H
        canvas.paste(pil_from(panel_arr), (x, y))
        if font is not None:
            d.text((x + 4, y - LABEL_H + 4), title, fill=0, font=font)

    return canvas


pair_out = OUT_DIR / "pair_grids"
pair_out.mkdir(exist_ok=True)
for idx, rec in enumerate(grid_selected):
    canvas = render_pair_grid(rec)
    title = (f"az_diff={rec['circular_azimuth_difference']:.1f} "
             f"best_sim={rec['best_similarity']:.3f} "
             f"({rec['best_flip_or_rotation']})")
    canvas = canvas.resize((canvas.width, canvas.height + 30))
    from PIL import ImageDraw, ImageFont
    dd = ImageDraw.Draw(canvas)
    try:
        fnt = ImageFont.load_default()
    except Exception:
        fnt = None
    dd.text((10, canvas.height - 24), title, fill=128, font=fnt)
    out_path = pair_out / f"pair_grid_{idx:03d}_hdiff{rec['circular_azimuth_difference']:.0f}.png"
    canvas.save(out_path)
    print(f"  saved {out_path.name}")

print(f"Saved {len(grid_selected)} pair grids to {pair_out}")

# ============================================================
# SECTION 7: CROSS-SPLIT TEST DUPLICATE ANALYSIS
# ============================================================

section("C. CROSS-SPLIT DUPLICATE ANALYSIS (829)")

cross_records = []
t0 = time.time()
for idx, h in enumerate(sorted(cross_hashes)):
    tr_ids = train_hashes[h]
    te_ids = test_hashes[h]
    # per prior audit these are 1:1, but handle generally
    for tri in tr_ids:
        for tei in te_ids:
            az_tr = train_lookup[tri]["azimuth"]
            az_te = test_lookup[tei]["azimuth"]
            circ = circular_diff(az_tr, az_te)
            arr = load_pixels_by_hash(h)

            # official rotations
            r_tr = rotate_array(arr, az_tr)
            r_te = rotate_array(arr, az_te)
            m = mae(r_tr, r_te)
            c = pearson(r_tr, r_te)
            best_name, best_sim = best_flip_relation(r_tr, r_te)

            cross_records.append({
                "hash": h,
                "train_image_id": tri,
                "train_azimuth": az_tr,
                "train_label": train_lookup[tri]["label"],
                "test_image_id": tei,
                "test_azimuth": az_te,
                "circular_azimuth_difference": circ,
                "raw_identical": True,
                "official_rotated_mae": m,
                "official_rotated_corr": c,
                "best_flip_or_rotation": best_name,
                "best_similarity": best_sim,
            })
    if (idx + 1) % 200 == 0:
        print(f"  [{idx+1}/829] {time.time()-t0:.1f}s")

cross_df = pd.DataFrame(cross_records)
cross_df.to_csv(OUT_CROSS_SPLIT, index=False)
print(f"Saved cross-split: {OUT_CROSS_SPLIT} ({len(cross_df)} rows)")

print("\n  Cross-split canonicalized similarity (official A):")
cc = cross_df["official_rotated_corr"].to_numpy(dtype=float)
cc_fin = cc[np.isfinite(cc)]
print(f"    mean corr = {np.nanmean(cc):.4f}, median = {np.nanmedian(cc):.4f} "
      f"(valid rows: {len(cc_fin)}/{len(cc)})")
print(f"    mean MAE  = {cross_df['official_rotated_mae'].mean():.4f}")
print("\n  Best-flip winner (official A):")
cw = defaultdict(int)
for v in cross_df["best_flip_or_rotation"]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = "NONE"
    cw[v] += 1
for k, v in sorted(cw.items(), key=lambda kv: -kv[1]):
    print(f"    {str(k):10s}: {v} ({v/len(cross_df)*100:.1f}%)")

# also estimate: does canonicalized test resemble canonicalized train more
# than expected? We compare against the global distribution of corr between
# two random rotations of 256x256 — approximate control: compare computed corr
# to the distribution of corr for all train conflicting pairs (independent
# scenes). We simply report the cross-split corr distribution relative to the
# train-pair distribution. Statistical comparison is done in the summary.
print("\n  Training conflicting-pair official corr distribution (reference):")
tc = pair_df["A_corr"].to_numpy(dtype=float)
for name, arr_ in [("cross_split", cc), ("train_conflict", tc)]:
    arr_ = np.asarray(arr_, dtype=float)
    arr_ = arr_[np.isfinite(arr_)]
    q = np.percentile(arr_, [10, 25, 50, 75, 90])
    print(f"    {name:14s}: mean={arr_.mean():.4f} median={np.median(arr_):.4f} "
          f"P10..P90={q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f},{q[4]:.3f}")

# ============================================================
# SECTION 9: SYNTHETIC GEOMETRY SANITY CHECK
# ============================================================

section("D. SYNTHETIC GEOMETRY SANITY CHECK")

# Purpose: verify that "different illumination directions + prescribed
# rotation (~ -azimuth)" approximately aligns illumination orientation.
# We render a shaded disk (with a directional light source) at two azimuths
# and show that after -azimuth rotation the light comes from the same screen
# direction. This validates the rotation sign/convention physically.

def make_shaded_disk(light_dir_deg):
    """Render a 256x256 shaded disk with a directional light from light_dir_deg
    (0 = from the right, 90 = from the top, in screen coords)."""
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    cx = cy = 128.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = 128.0 * np.ones_like(xx)          # mid-grey background
    disk = r <= 60.0
    # base disk brightness (convex mound)
    base = 200.0
    img[disk] = base
    # simple shading: brightness scaled by alignment with light dir.
    ang = np.deg2rad(light_dir_deg)
    lx, ly = np.cos(ang), np.sin(ang)       # light direction (unit)
    # direction from centre to each pixel
    dx = (xx - cx) / 60.0
    dy = (yy - cy) / 60.0
    bump_img = img.copy()
    inside = disk
    align = dx * lx + dy * ly
    bump_img[inside] += 60.0 * align[inside]   # lit side brighter
    bump_img = np.clip(bump_img, 0, 255).astype(np.uint8)
    return bump_img


def render_synthetic_grid():
    panels = []
    for light in [0, 90, 180, 270]:
        disk = make_shaded_disk(light)
        # official rotation by -azimuth; the "azimuth" here is the light dir.
        # after rotate(-light), the light source should move to a canonical
        # direction (the same for all).
        rotated = Image.fromarray(disk, "L").rotate(
            -light, resample=Image.BILINEAR, expand=False)
        rotated = np.asarray(rotated, dtype=np.uint8)
        panels.append(("light=%d raw" % light, disk))
        panels.append(("light=%d rot(-az)" % light, rotated))

    # also check whether rotated versions align: corr between rotated at 0 and
    # rotated at 90 etc.
    rotmaps = {}
    for light in [0, 90, 180, 270]:
        disk = make_shaded_disk(light)
        rotmaps[light] = Image.fromarray(disk, "L").rotate(
            -light, resample=Image.BILINEAR, expand=False)
    print("  Synthetic: corr between -az rotations of lights")
    lights = [0, 90, 180, 270]
    for a in lights:
        for b in lights:
            if a < b:
                ca = np.asarray(rotmaps[a], dtype=np.uint8)
                cb = np.asarray(rotmaps[b], dtype=np.uint8)
                pc = pearson(ca, cb)
                print(f"    corr(rot(-{a}),rot(-{b})) = {pc:.4f}")

    ncols = 4
    nrows = 2
    P = 200
    canvas = np.full((nrows * P, ncols * P), 128, dtype=np.uint8)
    for k, (title, panel) in enumerate(panels):
        r = k // ncols
        c = k % ncols
        p = Image.fromarray(panel, "L").resize((P, P))
        canvas[r * P:(r + 1) * P, c * P:(c + 1) * P] = np.asarray(p)
    out = OUT_DIR / "synthetic_sanity.png"
    Image.fromarray(canvas, "L").save(out)
    print(f"  Saved {out}")
    return out


synthetic_out = render_synthetic_grid()

# ============================================================
# SUMMARY MARKDOWN
# ============================================================

section("E. WRITING SUMMARY MARKDOWN")

# finalized statistical comparison
tc = pair_df["A_corr"].to_numpy(dtype=float)
cc = cross_df["official_rotated_corr"].to_numpy(dtype=float)

def fmt(series):
    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    q = np.percentile(s, [10, 25, 50, 75, 90])
    return (s.mean(), q[2], q[0], q[1], q[3], q[4])

lines = []
A = lines.append
A("# Azimuth Rotation / Duplicate Alignment Audit")
A("")
A(f"Run date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
A("")
A("## Rotation convention (documented)")
A("")
A("- Library: PIL/Pillow `Image.rotate`")
A("- Amount: `rotated = rotate(image, -sun_azimuth_angle)` (official, convention A)")
A("- Sign: PIL positive angle = counter-clockwise (verified empirically)")
A("- Interpolation: BILINEAR")
A("- expand: False (output stays 256x256; content rotated about center; corner fill = 0/black per PIL default for L mode)")
A("- Manual crop: none (expand=False is PIL's own descale to canvas); no additional manual crop")
A("- Deterministic: yes")
A("")
A("Conventions tested (controls only): A=-az, B=+az, C=-(az-90), D=+(az-90).")
A("Convention A is the organizer-prescribed one and remains authoritative.")
A("")
A("## 1. Train conflicting duplicate pairs (1458)")
A("")
A("Each pair: identical raw pixels, one labeled Depth (0), one Rise (1), different azimuths.")
A("")
A("### Correlation distributions across rotation conventions")
A("")
A("| Convention | mean corr | median corr | P10 | P25 | P75 | P90 |")
A("|---|---|---|---|---|---|---|")
for conv in CONVENTIONS:
    m = summarize_metric(pair_df[f"{conv}_corr"])
    A(f"| {conv} | {m['mean']:.4f} | {m['median']:.4f} | "
      f"{m['p10']:.4f} | {m['p25']:.4f} | {m['p75']:.4f} | {m['p90']:.4f} |")
A("")
A("### Best flip/rotation relationship (official A)")
A("")
A("For each pair we take the canonicalized class-0 image vs canonicalized class-1 image and find which of {identity, horizontal flip, vertical flip, 180° rotation} maximizes Pearson correlation.")
A("")
A("| Winner | count | % |")
A("|---|---|---|")
for k, v in sorted(flip_winner_counter.items()):
    A(f"| {k} | {v} | {v/len(pair_df)*100:.1f}% |")
A("")
A("### MAE distributions (official A)")
mA = summarize_metric(pair_df["A_mae"])
A(f"Official A MAE: mean={mA['mean']:.2f}, median={mA['median']:.2f}")
A("")
A("## 2. Cross-split exact duplicates (829)")
A("")
A("| Quantity | value |")
A("|---|---|")
A(f"| shared hashes | {len(cross_df)} |")
A(f"| official canonicalized corr mean | {np.nanmean(cc):.4f} |")
A(f"| official canonicalized corr median | {np.nanmedian(cc):.4f} |")
A(f"| official canonicalized MAE mean | {cross_df['official_rotated_mae'].mean():.4f} |")
A("")
A("### Cross-split best flip/rotation (official A)")
A("")
A("| Winner | count | % |")
A("|---|---|---|")
for k, v in sorted(cw.items(), key=lambda kv: -kv[1]):
    A(f"| {k} | {v} | {v/len(cross_df)*100:.1f}% |")
A("")
A("### Reference: correlation between canonicalized class-0 vs class-1 images of INDEPENDENT conflicting pairs")
mc = fmt(cc)
mt = fmt(tc)
A(f"- cross-split canonicalized corr: mean={mc[0]:.4f}, median={mc[1]:.4f}")
A(f"- train conflicting-pair canonicalized corr: mean={mt[0]:.4f}, median={mt[1]:.4f}")
A("")
A("## 3. Visual grids")
A("")
A(f"- Pair grids: `azimuth_rotation_audit/pair_grids/pair_grid_*.png` ({len(grid_selected)} pairs)")
A("- Synthetic sanity check: `azimuth_rotation_audit/synthetic_sanity.png`")
A("")
A("## 4. Outputs")
A("")
A(f"- `{OUT_PAIR_METRICS.name}` ({len(pair_df)} rows)")
A(f"- `{OUT_CROSS_SPLIT.name}` ({len(cross_df)} rows)")
A("")
A("## 5. Raw audit (Section 8) caveat")
A("")
A("Raw identical pixels rotated by different azimuths necessarily produce different arrays.")
A("Simple rotated-image similarity is NOT proof of canonicalization by itself. We therefore")
A("report (a) the flip/rotation structure and (b) how canonicalized similarity for real pairs")
A("compares to the reference distribution, and flag that the synthetic check validates the")
A("rotation sign physically before drawing any conclusion.")

md = "\n".join(lines) + "\n"
md_path = Path(__file__).resolve().parent / "azimuth_rotation_summary.md"
md_path.write_text(md)
print(f"Saved summary: {md_path}")

section("COMPLETE")
print("Generated files:")
for f in [OUT_PAIR_METRICS, OUT_CROSS_SPLIT, md_path,
          OUT_DIR / "synthetic_sanity.png"]:
    print("  ", f)
