# Experiment 3 — Kaggle Setup Guide

Scientific signal-decomposition experiment (grouped 5-fold CV by exact image hash).
Run this on Kaggle GPU. CNN models (B–E) train there; you report results back.

## What was already done locally (verified, trustworthy)
- **Fold assignment** `experiment_3_outputs/fold_assignments.csv` — deterministic, seed=42,
  grouped by exact sha256 image hash, leakage-free (0 hashes cross folds), class-balanced.
  All 5 models reuse these exact folds.
- **Model A** (azimuth-only sin/cos logistic regression) already measured on those folds:
  `OOF BA = 0.7765`, mean fold BA = 0.7765 ± 0.0092, OOF AUC = 0.7853.
  This confirms the ~0.776 azimuth signal **survives leakage-free grouped validation**.
- **Dataset package** (images + metadata + hashes + folds) zipped for upload.

## Handoff you perform on Kaggle (web interface)

1. **Upload the dataset package.**
   - Kaggle → Datasets → New Dataset.
   - Upload `pareidolia_experiment3_kaggle.zip`.
   - It auto-extracts to `/kaggle/input/<dataset-name>/` containing `images/`,
     `train_metadata.csv`, `image_hashes.csv`, `fold_assignments.csv`, `README.txt`.
2. **Create / open a Kaggle notebook** with **GPU accelerator** enabled (Settings → Accelerator → GPU).
3. **Attach the dataset** you just uploaded (right panel → Data → Add → your dataset).
4. **Import the notebook**, e.g. File → Upload Notebook → choose `experiment_3_signal_decomposition.ipynb`.
   (Or copy the cells into an empty notebook.)
5. **Network off** (optional; we don't download anything; set Network → Off for determinism).
6. **Run All**.
   - Auto-detects the dataset path; validates hashes/folds (aborts if any leak);
   - trains A (seconds) then B,C,D,E (the main GPU cost, ~1–3 h total);
   - computes per-fold + aggregate metrics, OOF-optimized threshold, confusion matrices,
     azimuth-bin diagnostics, duplicate-group diagnostics;
   - writes everything to `/kaggle/working/experiment_3_outputs/` and zips it.
7. **Download** `/kaggle/working/experiment_3_results.zip`.
8. **Send the results back** to the lead researcher.

## Expected outputs (in experiment_3_results.zip)
```
signal_decomposition_oof.csv   # image_id,hash,fold,label,azimuth + prob_A..E (7854 rows)
experiment_results.csv         # per model x fold + training times
aggregate_results.csv          # mean/std BA, OOF BA, recalls, AUC, optimal threshold
fold_assignments.csv           # deterministic fold IDs
fold_statistics.csv            # per-fold class balance
confusion_matrices.csv         # per model @ t=0.5 and OOF-optimal
azimuth_bin_diagnostics.csv    # performance by 8 azimuth bins
training_config.json
experiment_report.md           # full writeup + the six scientific questions
model_checkpoints/*.pt         # saved OOF arrays per model
logs/
```

## If you hit an error
Report the full error text and the cell that produced it; we'll fix it together.
Common ones: dataset name/path mismatch (the notebook auto-detects, but if it
can't find `train_metadata.csv` it raises a clear message — then attach the right
dataset), or out-of-memory (reduce `CFG['batch_size']` from 32 to 16).

## Notes
- Threshold policy: results reported at t=0.5 first; then a single global
  OOF-optimized threshold (no per-fold tuning reported as untouched).
- No test images used. No ensemble, no augmentation, no ConvNeXt/EfficientNet/SfS.
