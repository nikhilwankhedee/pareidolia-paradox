# Experiment log / notes

Chronological research notes. Each experiment lives in `experiments/<stage>/`
with its script, outputs, and report. This file is a short index.

## 2026-09-01 — Reconnaissance (`experiments/reconnaissance`)
Dataset structure, class counts (0=Depth 2854, 1=Rise 5000), image size, and
grayscale verification (R==G==B for all images).

## 2026-09-02 — Duplicate audit (`experiments/duplicate_audit`)
Exact-hash duplicate analysis:
- 829 test images (41.4%) share an exact train hash; all cross-split pairs
  differ in azimuth.
- 1,458 train duplicate groups, ALL label-conflicting ({0,1} identical pixels).
This established that the label is not a function of pixels alone.

## 2026-09-03 — Official rotation audit (`experiments/rotation_audit`)
Tested whether `rotate(image, -azimuth)` reconciles contradictory duplicates.
Result: negative. Canonicalization does not resolve the label contradiction
(mean corr 0.288; hflip 25.9 / identity 22.4 / rot180 24.6 / vflip 27.2 BA).
Retained as the official preprocessing for model training regardless.

## 2026-09-04 — Experiment 3 (`experiments/experiment_3`)
Leakage-free grouped 5-fold CV; Model A measured locally (OOF BA ≈ 0.7765);
Models B–E (CNNs) run on Kaggle GPU via `notebooks/...ipynb`.
