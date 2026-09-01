# Raw competition data

The raw competition images and archives are **not** committed to git
(they are large and delivered by the competition platform).

Place them in this directory before running local scripts:

```
data/raw/
    Train/train_metadata.csv
    Train/images/train_images/*.png
    Test/test_metadata.csv
    Test/images/eval_images/*.png
```

Local scripts and fold-building expect the following relative layout
(as used by `reconnaissance.py`, `duplicate_audit.py`,
`azimuth_rotation_audit.py`, and `experiment_3_folds.py`):

```
.                            <- repository root
├── Train/
│   ├── train_metadata.csv
│   └── images/train_images/*.png
└── Test/
    ├── test_metadata.csv
    └── images/eval_images/*.png
```

## Lightweight, committed metadata
Precomputed exact-image hashes and the deterministic fold assignment are
**committed** under `experiments/experiment_3/outputs/` because they are small,
deterministic, and reproducibility-critical:
- `image_hashes.csv`-equivalent data in each audit's outputs
- `fold_assignments.csv`, `fold_statistics.csv`
- `signal_decomposition_oof.csv` (Model A partial)

These let you reproduce folding and the azimuth-only baseline without the images.
