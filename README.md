# The Pareidolia Paradox — Repository

Scientific signal-localization study of the Pareidolia Paradox ML competition dataset.

The competition task: given a 256×256 grayscale lunar image and its sun azimuth angle,
classify each surface feature as **Depth (class 0)** or **Rise (class 1)**.

This repository is the *research record*. It contains code, experiment definitions,
reports, lightweight metadata, and reproducibility information. Raw competition
images are deliberately **not** committed (see `data/README.md`).

## Key scientific facts established (see `experiments/`)
- All 9,854 images are grayscale (R==G==B exactly).
- 829 test images (41.4%) have an exact duplicate in train.
- All 829 cross-split duplicate pairs have different azimuths.
- **All 1,458 train duplicate groups are label-conflicting**: pixel-identical images
  carry both label 0 and label 1, under different azimuths. The label is therefore
  not a function of pixels alone; azimuth is a likely determinant (not a pure confound).
- Official rotation (`rotate(image, -sun_azimuth_angle)`, PIL BILINEAR, expand=False)
  does **not** reconcile the contradiction (all four orientation conventions are
  statistically identical). Rotation is retained because it is the organizer-prescribed
  preprocessing for later model training.
- Azimuth-only sin/cos logistic regression reaches **OOF balanced accuracy ≈ 0.776**
  on grouped-stratified, leakage-free folds.

## Repository layout
```
src/            reusable modules (hashing, grouped folds, azimuth features, CNN)
experiments/    each investigation under its own subdirectory + report
notebooks/      Kaggle GPU notebook (Experiment 3)
configs/        training / experiment configs
results/        aggregated result tables and summaries
research/       hypotheses, experiment notes, literature
data/           (gitignored) placement for raw competition data
```

## Reproducing
Detailed setup and execution: `research/results/KAGGLE_SETUP_GUIDE.md`.

- Experiment 3 CNN training (Models B–E) runs on **Kaggle GPU** via
  `notebooks/experiment_3_signal_decomposition.ipynb`.
- Model A (azimuth-only) and fold construction run locally:
  `experiments/experiment_3/experiment_3_folds.py`.

## Compliance notes
- Grouped 5-fold CV by exact image hash; no exact pixel duplicate crosses folds.
- The test set is never used for model/threshold/feature selection.
