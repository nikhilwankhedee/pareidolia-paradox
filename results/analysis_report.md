# Results Report — The Pareidolia Paradox Dataset

**Date:** 2026-09-06
**Scope:** Reconnaissance, duplicate audit, rotation audit, and full Experiment 3 (Models A–E)
**Status:** Complete. CNN models B–E results retrieved from the sibling `results/` directory
(`/home/nikhil/projects/ieee-paradox-comp/results/`); see §6.

---

## 1. Executive summary

The dataset expresses a **label–azimuth coupling** that is real and measurable, and it **dominates
the learnable signal**. Azimuth-only logistic regression reaches OOF balanced accuracy ≈ **0.777**,
which **beats every tested CNN** (C/E ≈ 0.74, D ≈ 0.71), while a pixels-only CNN collapses to
**chance (≈ 0.500)** under leakage-free, hash-grouped 5-fold CV. The organizer-prescribed
`−sun_azimuth` rotation behaves correctly as an image operation (it lifts the pixels-only CNN to
0.71) but does **not** reconcile the fundamental ambiguity: **every one of the 1,458 train duplicate
groups contains pixel-identical images carrying *opposite* labels** (Depth=0 and Rise=1) under
different azimuths. Rotation alone cannot resolve this, and the azimuth must be used as
**metadata/feature** rather than expected to be "rotated away."

---

## 2. Reconnaissance (data integrity)

- **7,854 train + 2,000 test** grayscale 256×256 PNG images; all images are RGB-mode but have
  **R == G == B exactly** (single-channel content, 3-channel files).
- No missing metadata, no corrupt/unreadable files, all IDs sequential and matched 1:1 to files.
- Class imbalance: **0 → 2,854 (36.3%), 1 → 5,000 (63.7%)**.
- Label is **independent of image-ID ordering and file size** (correlations ≈ 0.01), so no obvious
  positional/byte-size labeling artifact.

### Train/test distribution shift (important)
Train azimuth (mean 218.6°, std 107.5°) vs test azimuth (mean 159.0°, std 80.5°). Test azimuths
cluster heavily around 90–180° (61.5% of test in [90°,180°) vs 18.2% of train). A large fraction of
train sits in [270°,360°). Train and test azimuth distributions **differ substantially** — this shift
must be accounted for when evaluating (and may challenge generalization).

### Class ↔ azimuth relationship
Class label is strongly tied to azimuth:
- For azimuth < ~270°, class 1 (Rise) dominates (~85–90%).
- For azimuth ≥ ~270° (**270–360°**), class 0 (Depth) dominates (~63–67%).

This single clean bifurcation is why an azimuth-only model does well: it predicts ⬇ illumination
angle mostly → Rise, ⬆/western angle mostly → Depth.

---

## 3. Duplicate audit

- **Cross-split exact duplicates:** **829** unique hashes appear byte-identical in both train and
  test (**41.4%** of the test set). Every one of the 829 has a **different** azimuth in test (0% share
  azimuth; median circular difference ≈ 88.2°).
- **Train duplicates:** **1,458** multi-image groups; **every one is label-conflicting**
  (identical pixels → both class 0 and class 1). Zero groups are internally label-consistent.
- **Test duplicates:** 98 groups, all with differing azimuths (no labels available).

**Key inference:** The label is **not a function of the pixels alone**. The same exact 2D image is
labeled Depth under one azimuth and Rise under another, implying the class depends on **azimuth /
lighting geometry** (perceptual shadow cue flips direction), consistent with the pareidolia framing
(crater vs. mound read from shadow direction), not random label noise.

---

## 4. Rotation audit

Test directly whether official `rotate(image, −sun_azimuth)` (PIL BILINEAR, expand=False)
canonicalizes the contradictory duplicates.

- **Mechanics correct:** PIL rotation sign verified empirically; synthetic shaded-disk check confirms
  `−az` brings opposite light directions into alignment (corr 0.98). The transform does what it claims.
- **Null result on canonicalization:** after `−az` rotation, class-0 vs class-1 versions of the *same*
  raw pixels have **Pearson corr mean ≈ 0.288** (median ≈ 0.30, spread P10=−0.11 → P90=0.70), not
  near-1. Best flip/rotation relationship is essentially uniform across identity/hflip/vflip/rot180
  (~22–27% each) — **no systematic canonical orientation** is revealed.
- **Cross-split duplicates behave the same:** canonicalized train vs canonicalized test corr
  mean ≈ 0.298, statistically identical to the reference from independent train pairs (0.288). After
  rotation, a test image is **no more similar** to its own train twin than to an arbitrary pair.
- **All four rotation conventions are statistically identical** (A −az, B +az, C/D ±(az−90)).

**Conclusion:** rotation is retained (organizer-prescribed preprocessing) but does **not** reconcile
dual-label duplicates, and 2D rotation of a single 2D image **cannot recover the 3D geometry** that
separates crater from mound.

---

## 5. Experiment 3 — Model A: azimuth-only baseline

Leakage-free, deterministic **hash-grouped 5-fold CV** (seed 42); no exact pixel duplicate crosses
folds. Mixed (0+1) groups distributed round-robin to preserve balance. Fold class balance is nearly
identical across folds (~36.3% / ~63.7%).

| Metric | Value |
|---|---|
| Mean fold BA | **0.7765** |
| Std fold BA | 0.00924 |
| **OOF BA** (t=0.5) | **0.7765** |
| OOF recall class 0 | 0.804 |
| OOF recall class 1 | 0.749 |
| OOF ROC-AUC | 0.785 |
| OOF-optimal threshold / BA | 0.51 / 0.781 |
| Training time | ~1.2 s |

**Reading:** a model that sees *only* `sin/cos(azimuth)` hits ~0.78 balanced accuracy. This is the
**real signal floor the pixels must beat** when combined with azimuth — and the pixel-only ceiling to
test under the same hash-grouped split.

---

## 6. Models B–E (CNN) — full results

The notebook `notebooks/experiment_3_signal_decomposition.ipynb` defines four small CNNs
(`SmallCNN`, 12 epochs, AdamW 1e-3, plain BCE, batch 32, AMP, seed 42) run on **Kaggle GPU**, sharing
the exact hash-grouped fold assignment used for Model A. Results (saved by the run under
`/home/nikhil/projects/ieee-paradox-comp/results/`) are reproduced below.

| Model | Image | Azimuth | mean_BA | std_BA | OOF_BA | OOF_R0 | OOF_R1 | OOF_AUC | opt_BA |
|---|---|---|---|---|---|---|---|---|---|
| **A** `azimuth` | — | ✓ (logreg) | **0.777** | 0.009 | **0.777** | 0.805 | 0.749 | 0.785 | 0.781 |
| E `rotated_azimuth` | −az | ✓ | 0.741 | 0.016 | 0.741 | 0.744 | 0.738 | 0.775 | 0.751 |
| C `raw_image_azimuth` | raw | ✓ | 0.739 | 0.020 | 0.739 | 0.690 | 0.789 | 0.784 | 0.772 |
| D `rotated_image` | −az | — | 0.712 | 0.011 | 0.712 | 0.674 | 0.750 | 0.758 | 0.717 |
| B `raw_image` | raw | — | **0.500** | 0.000 | **0.500** | 0.000 | 1.000 | 0.498 | 0.502 |

Single-OOF-optimized threshold reported (no per-fold tuning). Training times: A ~0.04s; B/C ~29 min
each; D/E ~38 min each.

### The decisive reading

1. **Model B (pixels only, no azimuth) ≈ 0.50 = chance.** The pixel-only CNN learned essentially
   nothing — it collapses to the majority class (Rise), OOF AUC ≈ 0.50. This **confirms the pixel
   ceiling is ~chance** under leakage-free hash-grouped CV: identical pixels map to both labels, so
   there is no pixels-alone signal to extract.

2. **Azimuth alone (A, 0.777) beats every CNN.** The best CNN with azimuth (C/E ~0.74, D ~0.71) all
   fall **short of pure azimuth (0.777)**. Adding the image to the azimuth does **not** improve on
   azimuth alone; it slightly *hurts*.

3. **Rotation helps the pixels-only CNN** (B 0.500 → D 0.712), but not to the level of azimuth and
   not beyond it. Adding azimuth on top of rotated pixels (D 0.712 → E 0.741) recovers only a small
   gain and still trails A.

4. **Net:** the learnable signal is dominated by the **azimuth metadata**; pixel content contributes
   little additional predictive value beyond azimuth on this split. There is no evidence the CNN
   extracts visual "depth vs. rise" geometry beyond what the sun angle alone already encodes —
   consistent with the label being tied to illumination angle rather than recoverable 3D structure
   from a single 2D image.

### Singleton-only check (memorization guard)
Evaluating only on singleton hashes (excluding all 1,458 duplicate groups) gives essentially the same
OOF BA for every model (A 0.775, B 0.500, C 0.741, D 0.713, E 0.741). So the results are **not** an
artifact of the duplicate groups / memorization — the same signal (and non-signal) holds on genuinely
unique images.

---

## 7. Key takeaways

1. **Azimuth ≈ the label.** Azimuth-only logistic regression hits OOF BA ≈ **0.777**, the best of all
   five models. It is a *determinant*, not a confound to purge.
2. **Pixels alone cannot decide — empirically confirmed.** The pixel-only CNN (Model B) achieves
   **BA ≈ 0.500 (chance)** under hash-grouped CV. With identical pixels labeled both 0 and 1, there is
   no recoverable pixel-only signal.
3. **The image adds little over azimuth.** Every CNN-with-azimuth (C/E ≈ 0.74, D ≈ 0.71) *undershoots*
   azimuth-only (0.777). Pixel content does not meaningfully beat the sun angle alone. This argues
   **against** a pure-pixels strategy and raises the question of whether the 2D image carries any
   additional signal beyond azimuth at all.
4. **Rotation doesn't fix it.** The official `−az` rotation behaves correctly as an image operation and
   empirically helps the pixels-only CNN (B 0.500 → D 0.712), but still trails azimuth and does not
   reconcile dual-label duplicates.
5. **Results are not memorization artifacts.** Singleton-only OOF BA ≈ full OOF BA for every model.
6. **Validation must be hash-grouped.** 41.4% of test images already exist byte-identical in train.
7. **Watch for azimuth distribution shift** (train mean 219° vs test 159°) when extrapolating to the
   held-out test set.

---

## 8. Open / next steps

- The measured floor/ceiling pair is now established under one leakage-free split: **azimuth-only ≈
  0.777** vs **pixels-only ≈ 0.500**. The leading question for a submission now is **why the image does
  not beat azimuth** — possible next moves:
  - A larger / pretrained image backbone (the `SmallCNN` is small and 12 epochs); verify whether a
    stronger vision model recovers pixel signal beyond azimuth (may still be fundamentally bounded).
  - Treat this as **cheating-aware**: never rely on raw-pixel memorization; the azimuth is the
    dominant, robust signal.
  - **Consolidate results into the repo**: this analysis lives in the sibling
    `/home/nikhil/projects/ieee-paradox-comp/results/` directory (full `experiment_results.csv`,
    `signal_decomposition_oof.csv`, `confusion_matrices.csv`, `azimuth_bin_diagnostics.csv`, model
    checkpoints). Copy/commit them into `experiments/experiment_3/outputs/` so the report and repo
    stay in sync.
