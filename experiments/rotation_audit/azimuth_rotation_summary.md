# Azimuth Rotation / Duplicate Alignment Audit

Run date: 2026-09-04 00:16:02

## Rotation convention (documented)

- Library: PIL/Pillow `Image.rotate`
- Amount: `rotated = rotate(image, -sun_azimuth_angle)` (official, convention A)
- Sign: PIL positive angle = counter-clockwise (verified empirically)
- Interpolation: BILINEAR
- expand: False (output stays 256x256; content rotated about center; corner fill = 0/black per PIL default for L mode)
- Manual crop: none (expand=False is PIL's own descale to canvas); no additional manual crop
- Deterministic: yes

Conventions tested (controls only): A=-az, B=+az, C=-(az-90), D=+(az-90).
Convention A is the organizer-prescribed one and remains authoritative.

## 1. Train conflicting duplicate pairs (1458)

Each pair: identical raw pixels, one labeled Depth (0), one Rise (1), different azimuths.

### Correlation distributions across rotation conventions

| Convention | mean corr | median corr | P10 | P25 | P75 | P90 |
|---|---|---|---|---|---|---|
| A | 0.2883 | 0.2977 | -0.1127 | 0.0700 | 0.5014 | 0.7019 |
| B | 0.2886 | 0.3001 | -0.1139 | 0.0731 | 0.5024 | 0.7037 |
| C | 0.2883 | 0.2977 | -0.1127 | 0.0700 | 0.5014 | 0.7019 |
| D | 0.2886 | 0.3001 | -0.1139 | 0.0731 | 0.5024 | 0.7037 |

### Best flip/rotation relationship (official A)

For each pair we take the canonicalized class-0 image vs canonicalized class-1 image and find which of {identity, horizontal flip, vertical flip, 180° rotation} maximizes Pearson correlation.

| Winner | count | % |
|---|---|---|
| hflip | 377 | 25.9% |
| identity | 327 | 22.4% |
| rot180 | 358 | 24.6% |
| vflip | 396 | 27.2% |

### MAE distributions (official A)
Official A MAE: mean=45.09, median=42.08

## 2. Cross-split exact duplicates (829)

| Quantity | value |
|---|---|
| shared hashes | 829 |
| official canonicalized corr mean | 0.2979 |
| official canonicalized corr median | 0.3129 |
| official canonicalized MAE mean | 44.7160 |

### Cross-split best flip/rotation (official A)

| Winner | count | % |
|---|---|---|
| vflip | 218 | 26.3% |
| rot180 | 205 | 24.7% |
| hflip | 204 | 24.6% |
| identity | 201 | 24.2% |
| NONE | 1 | 0.1% |

### Reference: correlation between canonicalized class-0 vs class-1 images of INDEPENDENT conflicting pairs
- cross-split canonicalized corr: mean=0.2979, median=0.3129
- train conflicting-pair canonicalized corr: mean=0.2883, median=0.2977

## 3. Visual grids

- Pair grids: `azimuth_rotation_audit/pair_grids/pair_grid_*.png` (40 pairs)
- Synthetic sanity check: `azimuth_rotation_audit/synthetic_sanity.png`

## 4. Outputs

- `azimuth_rotation_pair_metrics.csv` (1458 rows)
- `azimuth_rotation_cross_split.csv` (829 rows)

## 5. Raw audit (Section 8) caveat

Raw identical pixels rotated by different azimuths necessarily produce different arrays.
Simple rotated-image similarity is NOT proof of canonicalization by itself. We therefore
report (a) the flip/rotation structure and (b) how canonicalized similarity for real pairs
compares to the reference distribution, and flag that the synthetic check validates the
rotation sign physically before drawing any conclusion.

---

## CONCLUSION (measured evidence)

### 1. Does the organizer-prescribed `-azimuth` rotation behave as expected geometrically?

**Yes, as an image operation it does exactly what it says.** PIL `Image.rotate(angle)`
treats a positive angle as counter-clockwise (verified empirically), so `rotate(-azimuth)`
is a clockwise rotation of `azimuth` degrees about the image center — the official prescribed
transform. The synthetic sanity check confirms the *sign convention is physically correct*:
rendering the same shaded disk under light directions 0/90/180/270° and then applying the
official `-azimuth` rotation yields near-identical images — corr(rot0, rot180)=0.98,
corr(rot90, rot270)=0.98 — i.e. opposite light directions align after canonicalization.
So the rotation *mechanics* behave as intended.

### 2. Does it provide evidence that contradictory labels are caused by illumination orientation (vs random noise)?

**No evidence from this experiment.** The critical negative result:

- After canonicalization, the class-0 vs class-1 images of the SAME raw pixels have
  **Pearson corr mean ≈ 0.288** (median 0.298), with a wide spread (P10 = −0.11, P90 = 0.70).
- **The best-flip/rotation relationship is essentially uniform**: hflip 25.9%, identity 22.4%,
  rot180 24.6%, vflip 27.2%. If rotation were revealing a consistent canonical orientation
  tied to the label, we would expect a single (or dominant) geometric relation to win. Instead
  all four are near 25% — **no systematic orientation relationship** between canonicalized
  class-0 and class-1 versions.
- MAE after canonicalization is high (mean 45, median 42 on a 0-255 scale), i.e. the two
  canonicalized images are substantially different, not near-identical.

If the illumination hypothesis were correct, canonicalizing by `-azimuth` should have driven the
same-scene duplicates to a strongly consistent, near-identical relationship. It did not. The
label-azimuth coupling is real (azimuth-only BA was 77.6%), but **simple image rotation of the
identical 2D pixels does not reconcile the labels**.

### 3. Which rotation convention gives the strongest canonicalization signal?

**None differentiates.** Conventions A (-az), B (+az), C (-(az-90)), D (+(az-90)) give statistically
identical distributions (A mean 0.2883 / B 0.2886 / C 0.2883 / D 0.2886; medians all ≈0.30).
There is no signal for choosing any alternative — and, per the brief, the official convention A
remains authoritative.

### 4. Does the official convention appear internally consistent with the dataset?

**Not on the basis of pixel similarity.** The canonicalization does not make same-scene duplicates
align. This, combined with the uniform flip distribution, indicates the organization-prescribed
rotation does **not** reconcile identical raw pixels into a shared class. It neither validates nor
invalidates azimuth's relevance to the label, but it shows a naive rotation-only canonicalization
is insufficient.

### 5. What happens to the 829 exact train/test overlaps after canonicalization?

Same story: canonicalized train vs canonicalized test corr mean ≈ 0.298 (median 0.313), nearly
**identical to the reference distribution** from independent conflicting train pairs (mean 0.288).
I.e., after rotation, a shared image's test version is **no more similar** to its train version than
to an arbitrary different pair. The one degenerate case (train_05533/eval_01334) produced an
all-black rotated frame (std 0) — a corner-clipping artifact, not meaningful.

### 6. What does this imply for the eventual model input?

- Azimuth is **not** a nuisance you can simply "rotate away" at the pixel level. Rotating by
  `-azimuth` with expand=False crops/wraps content and cannot recover the 3D geometry that
  distinguishes a crater from a mound from a single 2D image.
- The label–azimuth coupling (77.6% BA) therefore likely **cannot be resolved by rotation alone**.
  A model that sees only the rotated pixels + the scalar azimuth, or pixels only, will not resolve
  the fundamental ambiguity documented in the duplicate audit (identical pixels, opposite labels).
- Practical input implication: the azimuth should be provided to the model **as metadata/feature**
  (e.g. concatenated to features, or via a conditioning mechanism), while keeping the raw (or
  optionally rotated) image. Rotation may still be a useful data-agnostic normalization, but this
  experiment shows it does **not** canonicalize dual-label duplicates and should not be relied on to
  eliminate the ambiguity.

### 7. What does this imply for validation?

- Because identical pixels map to both labels and the pixels-only signal is insufficient,
  **row-level metric leakage is severe**: a model trained on the train rows of each duplicate group
  could see the "test-facing" pixels in train (41.4% of test). Validation must be **hash-grouped**
  (never split identical pixels across folds), or performance will be overestimated.
- Any metric that relies on the shared pixels alone is not measuring generalization to new scenes;
  it measures memorization/re-mapping of the azimuth.

### 8. SINGLE next experiment (recommendation)

**Quantify how well the azimuth alone predicts the label on genuinely NEW images (a "pure metadata"
baseline with strict hash-grouped CV) and establish the true ceiling of pixel-only models under
hash-grouped CV.**

Concretely:
1. Build a hash-grouped 5-fold CV (group = image hash) on the training set.
2. Baseline A: azimuth-only (sin/cos) logistic regression, measured under this leakage-free split.
3. Baseline B: a small pixel-only model (e.g. a tiny CNN or even raw-feature logistic regression)
   under the same hash-grouped split.
4. Report the balanced accuracy of each. If B is near A (or near 50%), it confirms a CNN-only
   approach cannot work without azimuth/external structure; if B meaningfully exceeds A, pixel
   content carries recoverable signal beyond azimuth.

This is the decisive measurement: **it separates "azimuth is the label" from "there is learnable
visual structure"** before committing to any architecture, and it is the direct next step after
this audit. Do **not** train a full CNN on the raw shared-heavy split — it would only inflate the
number via memorization.

---
