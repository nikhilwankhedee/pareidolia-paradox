# Pareidolia Paradox — Duplicate + Azimuth Audit

**Experiment:** Duplicate + Azimuth Audit
**Script:** `duplicate_audit.py`
**Run date:** 2026-09-03
**Scope:** All 7,854 train + 2,000 test images (9,854 total), exact SHA-256 pixel hashing, full metadata cross-checks.
**Result:** Audit complete, exit 0, no hashing/RGB errors.

---

## Hashing Summary

| Quantity | Value |
|---|---|
| Train images hashed | 7,854 |
| Train unique hashes | 6,396 |
| Train errors | 0 |
| Test images hashed | 2,000 |
| Test unique hashes | 1,902 |
| Test errors | 0 |
| **Cross-split shared hashes (train ∩ test)** | **829** |

---

## A. RGB Channel Audit

### A.1 Train (all 7,854 image files)

| Metric | Value |
|---|---|
| Sampled | 7,854 |
| Failed | 0 |
| R==G==B exactly | 7,854 (100.00%) |
| Any channel difference | 0 (0.00%) |
| Max channel difference | 0 |

### A.2 Test (all 2,000 image files)

| Metric | Value |
|---|---|
| Sampled | 2,000 |
| Failed | 0 |
| R==G==B exactly | 2,000 (100.00%) |
| Any channel difference | 0 (0.00%) |
| Max channel difference | 0 |

### A.3 Conclusion

**The dataset is effectively grayscale, replicated identically across the R, G, and B channels** (max channel difference = 0 on every sampled image). Future models can safely treat the input as single-channel *conceptually*, but the implementation should **retain 3 input channels** for pretrained-CNN compatibility (e.g. ConvNeXt/EfficientNet expects RGB).

---

## B. Train ↔ Test Exact Duplicate Audit

All 829 shared hashes are byte-identical images appearing in both train and test. These are exact copies, not near-duplicates.

### B.1 Cross-split summary

| Quantity | Count | % |
|---|---|---|
| Cross-split image × image row-pairs | 829 | 100% |
| Unique cross-split hashes | 829 | — |
| Unique test images involved | 829 | — |
| **SAME azimuth** | **0** | **0.00%** |
| **DIFFERENT azimuth** | **829** | **100.00%** |

Every shared image in train has a **different** sun azimuth than its test counterpart. No shared hash has the same azimuth across splits.

### B.2 Circular azimuth difference statistics (different-azimuth group, n=829)

| Statistic | Circular diff (°) | Raw diff (°) |
|---|---|---|
| Mean | 88.9686 | 112.1600 |
| Std | 53.2737 | 80.7898 |
| Min | 0.2500 | 0.2500 |
| P5 | 9.0720 | — |
| P25 | 39.6000 | — |
| **Median (P50)** | **88.1800** | 97.5800 |
| P75 | 134.4600 | — |
| P95 | 171.5260 | — |
| Max | 179.8100 | 353.8200 |

### B.3 Per-hash azimuth consistency

| Hash category | Count |
|---|---|
| Hashes where ALL pairs have same azimuth | 0 |
| Hashes where ALL pairs have different azimuth | 829 |
| Hashes with mixed (some same, some different) | 0 |

---

## C. Train Duplicate Label Consistency

There are **1,458 train duplicate groups** (hashes with >1 train image).

### C.1 Summary

| Category | Count |
|---|---|
| Total train duplicate groups | 1,458 |
| Groups with ONLY class 0 | 0 |
| Groups with ONLY class 1 | 0 |
| **Groups with BOTH class 0 AND class 1** | **1,458** |
| Images participating in conflicting groups | 2,916 |

### C.2 ⚠️ Key finding

**ALL 1,458 train duplicate groups are label-CONFLICTING.** Identical pixel content is labeled as **both** class 0 (Depth) and class 1 (Rise) within the training set.

> The label is **NOT** a function of the image pixels alone. The same exact image is labeled Depth under one row and Rise under another. The class must depend at least partly on the **azimuth/lighting metadata + geometry**, not the static 2D pixels.

### C.3 First 5 conflicting groups (all 1,458 saved to `duplicate_audit_train_conflicts.csv`)

| Hash | Image IDs | Labels | Azimuths |
|---|---|---|---|
| `0014f691...` | train_03333.png; train_05451.png | 0; 1 | 242.72; 127.14 |
| `003311ca...` | train_00589.png; train_03920.png | 0; 1 | 306.41; 200.86 |
| `0081849a...` | train_03083.png; train_04838.png | 0; 1 | 316.40; 51.93 |
| `00b0911a...` | train_03593.png; train_07629.png | 0; 1 | 312.15; 35.90 |
| `00b68fe2...` | train_03672.png; train_04404.png | 1; 0 | 98.40; 307.19 |

---

## D. Test Duplicate Consistency

There are **98 test duplicate groups** (hashes with >1 test image). No labels exist for test, so none are inferred.

### D.1 Summary

| Category | Count |
|---|---|
| Total test duplicate groups | 98 |
| Groups with SAME azimuth | 0 |
| Groups with DIFFERENT azimuth | 98 |

### D.2 Circular azimuth difference statistics (n=98 pairwise)

| Statistic | Value (°) |
|---|---|
| Mean | 103.3227 |
| Std | 48.8930 |
| Min | 1.7100 |
| P5 | 23.3475 |
| P25 | 65.5250 |
| **Median (P50)** | **105.8450** |
| P75 | 146.3625 |
| P95 | 173.9975 |
| Max | 179.4300 |

### D.3 Example test duplicate groups with different azimuth

| Hash | IDs | Azimuths |
|---|---|---|
| `00d0d27a...` | eval_00340.png; eval_01945.png | 220.15; 124.41 |
| `00f1fffe...` | eval_00154.png; eval_01159.png | 231.38; 101.03 |
| `048dba07...` | eval_00308.png; eval_00559.png | 26.39; 135.45 |
| `04dde58b...` | eval_00163.png; eval_01082.png | 145.00; 305.74 |
| `0a1312e1...` | eval_00128.png; eval_00677.png | 117.90; 285.74 |

---

## E. Cross-Split Duplicate Class Breakdown

For the 829 shared hashes, mapping each hash to its train label(s):

| Quantity | Value |
|---|---|
| Unique cross-split hashes | 829 |
| Hashes → train class 0 | 311 (37.52%) |
| Hashes → train class 1 | 518 (62.48%) |
| Hashes mapping to MULTIPLE train labels | **0** |

### E.1 Row accounting (counting hygiene)

The 829 shared hashes map to exactly **829 train rows** and **829 test rows** (1:1, no duplicate-hash double counting):

| Quantity | Value |
|---|---|
| Train rows participating in cross-split | 829 |
| Test rows participating in cross-split | 829 |
| Hash → multiple train labels | 0 |

No shared hash maps to more than one train image or more than one test image.

---

## F. Output Files

| File | Purpose | Rows |
|---|---|---|
| `duplicate_audit_cross_split.csv` | One row per shared hash: hash, train_image_ids, train_labels, train_azimuths, test_image_ids, test_azimuths, azimuth_same, circular_azimuth_difference | 829 |
| `duplicate_audit_train_conflicts.csv` | Every conflicting train duplicate group: hash, image_ids, labels, azimuths, num_images | 1,458 |

---

## G. Inspection Samples

### G.1 Same-image + same-azimuth cross-split examples (5)

**None exist** — 0 of 829 cross-split pairs have identical azimuth.

### G.2 Same-image + different-azimuth cross-split examples (10, spanning the range)

| Hash | Train | Test | Circ diff |
|---|---|---|---|
| `4473a998...` | train_03445.png (label=1, az=158.15) | eval_01387.png (az=157.90) | 0.25 |
| `4d26d758...` | train_01110.png (label=1, az=123.88) | eval_01464.png (az=141.75) | 17.87 |
| `3cc28d36...` | train_03624.png (label=0, az=300.90) | eval_00224.png (az=264.95) | 35.95 |
| `bdd8c7f0...` | train_07224.png (label=1, az=201.87) | eval_01920.png (az=145.27) | 56.60 |
| `8274df5d...` | train_05584.png (label=1, az=225.54) | eval_01704.png (az=148.65) | 76.89 |
| `f884079a...` | train_05714.png (label=1, az=24.70) | eval_01846.png (az=123.58) | 98.88 |
| `33bb8a36...` | train_02180.png (label=0, az=327.04) | eval_00311.png (az=207.26) | 119.78 |
| `28c4bd81...` | train_05326.png (label=0, az=298.05) | eval_00573.png (az=78.57) | 140.52 |
| `57255bca...` | train_00751.png (label=1, az=274.20) | eval_00352.png (az=111.52) | 162.68 |
| `a6c59d3d...` | train_00378.png (label=1, az=322.03) | eval_01262.png (az=142.22) | 179.81 |

### G.3 Train duplicate groups with repeated images (5)

| Hash | IDs | Labels | Azimuths |
|---|---|---|---|
| `0014f691...` | train_03333.png; train_05451.png | 0; 1 | 242.72; 127.14 |
| `003311ca...` | train_00589.png; train_03920.png | 0; 1 | 306.41; 200.86 |
| `0081849a...` | train_03083.png; train_04838.png | 0; 1 | 316.40; 51.93 |
| `00b0911a...` | train_03593.png; train_07629.png | 0; 1 | 312.15; 35.90 |
| `00b68fe2...` | train_03672.png; train_04404.png | 1; 0 | 98.40; 307.19 |

### G.4 Conflicting train duplicate groups

The first 8 are shown below; **all 1,458** are saved in full to `duplicate_audit_train_conflicts.csv`.

| Hash | IDs | Labels | Azimuths |
|---|---|---|---|
| `0014f691...` | train_03333.png; train_05451.png | 0; 1 | 242.72; 127.14 |
| `003311ca...` | train_00589.png; train_03920.png | 0; 1 | 306.41; 200.86 |
| `0081849a...` | train_03083.png; train_04838.png | 0; 1 | 316.40; 51.93 |
| `00b0911a...` | train_03593.png; train_07629.png | 0; 1 | 312.15; 35.90 |
| `00b68fe2...` | train_03672.png; train_04404.png | 1; 0 | 98.40; 307.19 |
| `00f1e1b6...` | train_02459.png; train_06163.png | 0; 1 | 320.64; 88.28 |
| `00fc4b66...` | train_01055.png; train_04665.png | 0; 1 | 296.55; 134.07 |
| `0111d161...` | train_06019.png; train_06035.png | 1; 0 | 178.64; 285.89 |

…and 1,450 more.

---

## H. Audit Conclusion

### 1. Is the dataset effectively grayscale?

**Yes.** All 9,854 sampled images have R == G == B exactly (max channel diff = 0). It is single-channel conceptually; keep 3 channels for pretrained-CNN compatibility.

### 2. Is train/test exact image overlap real?

**Yes.** 829 unique image hashes appear in both train and test, covering **829 unique test images (41.4% of the test set)**. These are byte-identical, not similar/near-duplicates.

### 3. Are overlapping images usually associated with the same or different azimuth?

**Always different.** 0/829 pairs share azimuth (0.0% same, 100.0% different). Median circular difference ≈ 88.2°, max ≈ 179.8°.

### 4. Are train duplicate labels consistent?

**No — completely inconsistent.** All 1,458 train duplicate groups are label-conflicting: identical pixels are labeled both Depth (0) and Rise (1). **The label is not a function of pixel content alone** — it depends on azimuth/lighting.

### 5. What does the overlap look like (from actual evidence)?

The 829 shared hashes are exact pixel copies, each appearing under **different azimuths** in train vs test (median Δ ≈ 88°). Inside train, the same pixels recur 1,458 times with **different labels AND different azimuths**. This is consistent with the same physical crater/mound patch being imaged at different sun angles, where the lighting determines which shadow-based cue (depth vs. rise) is perceived. This is a **construction artifact tied to lighting geometry**, not random labeling noise.

### 6. Implications

- **Validation design:** 829 shared hashes leak train/test; random CV overstates generalization. Use **hash-grouped CV** (never split identical pixels across folds).
- **Image memorization:** 829/2000 (41.4%) of test images exist byte-identical in train — a memorization shortcut exists. **But** since labels are azimuth-dependent and shared images have different azimuths/labels across splits, memorizing the train label is insufficient; azimuth must be incorporated.
- **Azimuth handling:** Azimuth is **NOT a confounding artifact to purge** — it is **likely relevant to the label**. Identical pixels get opposite labels based on lighting direction; the 77.6% azimuth-only BA reflects real label–azimuth coupling. The model must use azimuth together with pixels.
- **Organizer-prescribed −sun_azimuth rotation:** Rotating by −sun_azimuth_angle normalizes illumination direction, making lighting-independent geometry (crater vs. mound) comparable. Under correct rotation, identical patches imaged at different azimuths should **align** and yield the *same* class. This rotation is **probably essential**.

### 7. Recommended next experiment

**Test the −sun_azimuth rotation hypothesis directly.** Take the 1,458 currently-conflicting train duplicate pairs (identical pixels, opposite labels, different azimuths), rotate each by its `-sun_azimuth_angle`, and check whether the aligned pairs become **label-consistent**. If yes, rotation resolves the labeling ambiguity and is the key preprocessing step before any model work. This should be done before committing to an architecture (e.g., skip ConvNeXt/EfficientNet for now).

---

*Generated by `duplicate_audit.py`. Companion CSVs: `duplicate_audit_cross_split.csv`, `duplicate_audit_train_conflicts.csv`.*
