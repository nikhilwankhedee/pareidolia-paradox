# Kaggle run order

## One-GPU path

Run `00_full_sweep.ipynb`. It runs morphology and SfS first, then DINOv2, CLIP, torchvision, hybrid, and TTA phases sequentially. It reuses files in `/kaggle/working/clean_breakthrough/cache`, calls garbage collection, clears CUDA cache between phases, and writes `results/leaderboard.csv`. Optional encoders are skipped with an explicit error if unavailable.

## Independent path

1. `01_dino_b14.ipynb` — DINOv2 ViT-B/14
2. `02_dino_b14_azimuth.ipynb` — DINOv2 plus circular azimuth
3. `03_dino_multiview.ipynb` — DINOv2 multiview
4. `04_clip.ipynb` — CLIP
5. `05_pretrained_vision.ipynb` — torchvision encoder
6. `06_morphology.ipynb` — CPU morphology
7. `07_sfs.ipynb` — CPU shape-from-shading
8. `08_dino_morphology.ipynb` — DINOv2 plus morphology
9. `09_dino_sfs.ipynb` — DINOv2 plus SfS
10. `10_tta.ipynb` — test-time augmentation audit
11. `11_ensemble.ipynb` — OOF ensemble and final result

Each notebook independently discovers the dataset, verifies/recomputes hashes,
assigns grouped folds, extracts or loads a cache, writes OOF predictions and
metrics, and never accesses test labels. Optional foundation-model phases skip
with an actionable dependency/model message.
