# The Pareidolia Paradox — Solution & Methodology

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official competition repository for **"The Pareidolia Paradox"** machine learning challenge.

---

## Executive Summary

The Pareidolia Paradox challenges participants to classify 256×256 grayscale lunar surface images into **Depth (class 0)** or **Rise (class 1)** under varying solar illumination angles (`sun_azimuth_angle`).

Through systematic data forensics, mathematical modeling, and leakage-free cross-validation, we identified the true data-generating process of the competition dataset:
1. **The Pareidolia Optical Mechanism**: Top-lit vs bottom-lit lunar terrain inverts human and computer vision depth perception (craters appear as mounds and vice versa).
2. **Duplicate Forensics**: Exactly 1,458 duplicate pairs exist in the training data. Every single pair (100.0%) consists of pixel-identical images with conflicting labels under different solar azimuths.
3. **Hidden Test Structural Decomposition**: The 2,000 hidden test images decompose into three distinct structural regimes:
   - **Regime 1: Overlaps (829 images, 41.45%)**: Exact byte-level pixel duplicates of training images.
   - **Regime 2: Intra-test Duplicates (196 images / 98 pairs, 9.80%)**: Exact byte-level pixel duplicates paired within the test set.
   - **Regime 3: Novel Images (975 images, 48.75%)**: Novel lunar terrain instances with no pixel twin in train or test.
4. **Regime-Specific Architecture**:
   - **Overlap Regime**: Solved via the validated **Learned Flip Rule (World C)**: $y_{\text{test}} = 1 - y_{\text{train}}$ ($\text{BA} = 1.0000$ on train duplicates).
   - **Intra-test Regime**: Solved via the **$lo\_\Delta$ Transition Model** conditioned on $(az_{\text{lower\_band}} \times \Delta az_{\text{band}})$, achieving **0.8035** grouped CV Balanced Accuracy.
   - **Novel Regime**: Solved via a **Harmonic Order-1 LightGBM Classifier** and the Bayes-optimal step decision boundary at **$270.0^\circ$** ($\text{BA} \approx 0.7356$ on distribution-matched proxy, $\text{BA} \approx 0.7825$ on train OOF).
5. **Expected Performance**:
   - Regime-weighted validation estimate: **~0.8512** (distribution-matched proxy) / **~0.8747** (full-train OOF convention).
   - *Note: These are validation estimates; actual hidden-test Balanced Accuracy is evaluated by the competition organizers.*

---

## Competition Dataset & Metric

- **Task**: Binary classification of lunar terrain features.
  - Class 0: Depth (crater / depression)
  - Class 1: Rise (mound / hill / elevated terrain)
- **Train Set**: 7,854 images (256×256 grayscale, R==G==B).
- **Test Set**: 2,000 hidden images.
- **Evaluation Metric**: Balanced Accuracy:
  $$\text{Balanced Accuracy} = \frac{1}{2} \left( \text{Recall}_0 + \text{Recall}_1 \right) = \frac{1}{2} \left( \frac{\text{TP}}{P} + \frac{\text{TN}}{N} \right)$$
- **Crucial Metadata**: `sun_azimuth_angle` (degrees in $[0, 360)$).

---

## Methodology & Treatment of `sun_azimuth_angle`

### 1. Physical & Empirical Analysis of Azimuth
Extensive empirical audits demonstrated that the label is not determined by raw pixel textures in isolation, but by the illumination geometry encoded by `sun_azimuth_angle`:
- **The $270^\circ$ Boundary**:
  - For $\text{azimuth} \in [0^\circ, 270^\circ)$: $\sim 91.6\%$ of images are labeled Class 1 (Rise).
  - For $\text{azimuth} \in [270^\circ, 360^\circ)$: $\sim 55.7\% - 65.0\%$ of images are labeled Class 0 (Depth).
  - High-resolution (1-degree) empirical binning proves the transition boundary is located precisely at **$270.0^\circ$**.
- **Harmonic Feature Representation**:
  - Azimuth is represented circularly via 1st-order harmonic sine/cosine basis functions:
    $$X_{\text{harm}} = \left[ \sin\left(\frac{2\pi \cdot \theta}{360}\right), \cos\left(\frac{2\pi \cdot \theta}{360}\right) \right]$$

### 2. Regime 1: Train $\leftrightarrow$ Test Overlaps (829 Images)
- 829 test images share identical SHA-256 byte hashes with images in the training set.
- In the training set, duplicate pairs have conflicting labels 100.00% of the time across all $\Delta\theta$ buckets ($[0, 45^\circ), [45^\circ, 90^\circ), [90^\circ, 135^\circ), [135^\circ, 180^\circ]$).
- **Rule**: For each test image $t$ matching training image $s$:
  $$y_t = 1 - y_s$$
- Validated performance: **1.0000** Balanced Accuracy on train duplicate pairs.

### 3. Regime 2: Intra-Test Duplicates (196 Images / 98 Pairs)
- 98 pairs of test images share identical byte hashes with each other, but have no twin in the training set.
- Because duplicate twins always carry opposite labels, predicting one member determines the other: $y_{\text{hi}} = 1 - y_{\text{lo}}$.
- We evaluate the empirical distribution of $P(y_{\text{lo}} = 1)$ conditioned on the lower azimuth band and angular difference $\Delta\theta = |\theta_{\text{hi}} - \theta_{\text{lo}}|$:
  - When the pair crosses the $270^\circ$ boundary ($\theta_{\text{lo}} < 270^\circ$ and $\theta_{\text{hi}} \ge 270^\circ$): **93.32%** of pairs have $y_{\text{lo}} = 1$ and $y_{\text{hi}} = 0$! (37 out of 98 test pairs cross this boundary).
  - When both images have $\theta < 270^\circ$: predictions are assigned based on the empirical joint distribution table.
- Validated performance: **0.8035** grouped CV Balanced Accuracy across 15 independent resamples (outperforming the baseline E_joint rule of 0.7942).

### 4. Regime 3: Novel Images (975 Images)
- 975 test images have no pixel match in train or test.
- Audits revealed that fusing raw image retrieval cosine similarity damaged performance because similarity is strictly positive ($\sim 0.7 - 1.0$), artificially biasing novel predictions towards Class 1 and reducing proxy BA to 0.5600.
- Eliminating the noisy similarity bias and deploying the pure validated azimuth model:
  $$y_{\text{novel}} = \mathbb{I}\left(P_{\text{LGBM}}(y=1 \mid \theta) \ge 0.5\right) \equiv \mathbb{I}\left(\theta < 270.0^\circ\right)$$
- Corrects **114 novel predictions** (from 1 to 0 where $\theta \ge 270^\circ$) and restores novel proxy performance to **0.7356** BA.

---

## Validation & Performance Summary

All validation is strictly grouped by duplicate hash to prevent data leakage.

| Regime | Test Count | Mechanism / Model | Validated BA | Regime Contribution |
|---|---|---|---|---|
| **Overlap** | 829 (41.45%) | Learned Flip (World C) | **1.0000** | 829.0 |
| **Intra-test** | 196 (9.80%) | $lo\_\Delta$ Transition Model | **0.8035** | 157.5 |
| **Novel** | 975 (48.75%) | Harmonic LightGBM / Step-270 | **0.7356** | 717.2 |
| **OVERALL** | **2,000 (100%)** | **Regime-Weighted Pipeline** | **~0.8512** | **1,703.7 / 2,000** |

*Under the campaign's optimistic full-train OOF convention (novel = 0.7825): estimated overall BA is **~0.8747**.*

---

## Downloadable Model Weights

The complete trained model weights and deterministic inference lookup structures are fully packaged and downloadable:

- **Repository Artifacts**: Located directly in [`artifacts/`](artifacts/)
- **Direct Download Archive**: [`model_weights.zip`](https://github.com/nikhilwankhedee/pareidolia-paradox/raw/main/model_weights.zip)
- **Artifact Checksums (SHA-256)**:
  - `azimuth_model.joblib`: `ce2c3913e8747bea14573d839c347e8c31899e100fb2866fcdb106df9cc672b6`
  - `azimuth_model.txt`: `3737f32351323b31f59f3bd96538468bd1a16c867db95de35ef8937629eccb6d`
  - `intra_transition_model.json`: `bb4b65d9edc7c3724c7937eaa350fee0e3766692936f24908b3a4f0bcb083224`
  - `train_hash_lookup.json`: `972c9a23c23544e7b6793ce96bfdad02845a75e60d45f699acf50d2231c86ce7`
  - `pipeline_config.json`: `7adca47d83dc7fe173076aa60ad7c8c94a08492c46d507729a82c7b458bb048c`
  - `model_weights.zip`: `644b36d0e6959f1384be35780e5e965bd4a110627338e8575fc13af951e2aeaf`

---

## Environment Setup

### Requirements
- Python 3.10+
- Linux / macOS / Windows

Install dependencies via:
```bash
pip install -r requirements.txt
```

---

## Step-by-Step Execution & Reproducibility

The entire pipeline is 100% reproducible from scratch.

### 1. Training from Scratch
Run `train.py` to process the training dataset, build the hash lookup tables, fit the transition model, and train the azimuth classifier:
```bash
python train.py --train_dir ./Train --artifacts_dir ./artifacts --seed 42
```
*Expected runtime: ~15–20 seconds.*

### 2. Running Inference
Run `inference.py` to load the saved artifacts, partition the test images, and generate the final `submission.csv`:
```bash
python inference.py --test_dir ./Test --artifacts_dir ./artifacts --output ./submission.csv
```
*Expected runtime: ~1–2 seconds.*

---

## Final Submission Verification (QA)

The output `submission.csv` has been thoroughly verified against all competition requirements:

- [x] **File exists**: `submission.csv`
- [x] **Row count**: Exactly 2,000 prediction rows (+ 1 header row)
- [x] **Columns**: Exactly `image_id,label`
- [x] **Unique IDs**: Exactly 2,000 unique `image_id` strings matching `test_metadata.csv`
- [x] **Labels**: Binary integers strictly in `{0, 1}`
- [x] **Missing Values**: 0 null / NaN values
- [x] **Class Counts**: Class 0: **768** (38.4%), Class 1: **1,232** (61.6%)
- [x] **Submission SHA-256 Checksum**:
  ```
  e19b624b979286a47da42a270e3fcfc6e5bb697849fd2092793ecaac50fc3b41
  ```
- [x] **Reproducibility**: `train.py` and `inference.py` regenerate the exact byte-identical submission file from scratch.

---

## Repository Structure

```
pareidolia-paradox/
├── README.md                      # Comprehensive competition methodology & documentation
├── requirements.txt               # Complete runtime dependencies
├── train.py                       # Real training entrypoint (saves models & lookup structures)
├── inference.py                   # Real inference entrypoint (generates submission.csv)
├── submission.csv                 # Official final 2,000-row competition submission
├── model_weights.zip              # Downloadable packaged model weights archive
├── LINKEDIN_POST.md               # LinkedIn participation announcement template
├── artifacts/                     # Serialized model weights and deterministic artifacts
│   ├── azimuth_model.joblib
│   ├── azimuth_model.txt
│   ├── intra_transition_model.json
│   ├── train_hash_lookup.json
│   └── pipeline_config.json
├── experiments/                   # Detailed experiment records and audits
│   ├── final_day/                 # Final day benchmark suite, frozen champions, and QA
│   │   ├── champions/             # Protected champion checkpoints (baseline & candidates)
│   │   ├── phase2_novel_attack.py # Novel regime benchmark script
│   │   └── outputs/               # Benchmark tables and model correlations
│   ├── experiment_3/              # Signal decomposition experiments
│   ├── experiment_4/              # CNN, FiLM, and residual feature sweeps
│   └── rotation_audit/            # Azimuth rotation convention audit
├── src/                           # Reusable core modules
│   ├── azimuth.py                 # Circular arithmetic & harmonic transformations
│   ├── data.py                    # Dataset discovery & image utilities
│   ├── duplicates.py              # Duplicate mechanics & direction tables
│   ├── hashes.py                  # Exact SHA-256 hashing & regime partition logic
│   ├── models.py                  # LightGBM & classifier wrappers
│   └── validation.py              # Grouped-stratified cross-validation
└── Test/                          # (Local data) test metadata and images
└── Train/                         # (Local data) train metadata and images
```

---

## Authors & Acknowledgments

- **Nikhil Wankhede** — Lead ML Competition Engineer
- Competition: *The Pareidolia Paradox* (IEEE / Competition Organizers)
- Submission Date: September 2026
