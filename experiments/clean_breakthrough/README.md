# Clean Breakthrough experiment suite

A Kaggle-ready, leakage-controlled suite for the Pareidolia Paradox Dataset. The exact independent phases are `01_dino_b14.ipynb` through `11_ensemble.ipynb`. For a single-GPU run, use `00_full_sweep.ipynb`: it executes the high-value phases sequentially, reuses caches, and frees GPU memory between encoders. Every experiment uses exact-image hashes only for grouped OOF folds, never reads test labels, and writes caches/artifacts under `/kaggle/working/clean_breakthrough` (or `--output`). Attach the competition dataset and run either the single-GPU orchestrator or individual notebooks; each phase invokes the shared extraction/OOF pipeline.

The orchestrator bootstraps this suite from the locked public commit
`94e064d80d1d6d69d1efbba3f2534089721cb8bd` of
`https://github.com/nikhilwankhedee/pareidolia-paradox.git`, so the notebook does
not depend on source files being manually uploaded alongside it.

Optional encoders (DINOv2, CLIP, torchvision) are loaded only when requested; notebooks report a clear skip if a model or dependency is unavailable. CPU-friendly morphology and shape-from-shading descriptors provide dependable runnable baselines. No expensive training is run by repository smoke tests.

**Rules:** recompute/verify hashes; fail if a hash crosses folds; fit preprocessing inside each fold; select one OOF threshold; preserve `image_id` order; save predictions and metrics with schema.
