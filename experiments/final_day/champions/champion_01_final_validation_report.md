# Champion 01 Final Validation Report

## 1. Champion identity

- Champion: `candidate_v1_azfix`
- Locked commit: `94e064d80d1d6d69d1efbba3f2534089721cb8bd`
- Locked submission: `submission.csv`
- Submission SHA-256: `e19b624b979286a47da42a270e3fcfc6e5bb697849fd2092793ecaac50fc3b41`
- Locked weights: `model_weights.zip`
- Weights SHA-256: `644b36d0e6959f1384be35780e5e965bd4a110627338e8575fc13af951e2aeaf`

The locked submission and inference logic were not modified during this audit. No competition submission was made.

## 2. Integrity and reproducibility

The locked submission has exactly 2,000 rows, columns `image_id,label`, 2,000 unique IDs, no nulls, labels only in `{0, 1}`, and class counts `{0: 768, 1: 1232}`.

`model_weights.zip` passed archive-integrity testing. Its five required artifacts loaded successfully:

- `artifacts/train_hash_lookup.json`
- `artifacts/intra_transition_model.json`
- `artifacts/pipeline_config.json`
- `artifacts/azimuth_model.joblib`
- `artifacts/azimuth_model.txt`

A clean scratch run using:

```bash
python3 train.py --train_dir ./Train --artifacts_dir /tmp/champion01_repro_artifacts --seed 42
python3 inference.py --test_dir ./Test --artifacts_dir /tmp/champion01_repro_artifacts --output /tmp/champion01_repro_submission.csv
```

regenerated the locked submission byte-for-byte. Generated SHA-256:
`e19b624b979286a47da42a270e3fcfc6e5bb697849fd2092793ecaac50fc3b41`.

The pipeline uses metadata labels only for training and train-hash lookup construction. No test labels are present or consumed. No absolute machine-local path is required by `train.py` or `inference.py`.

## 3. Dataset and regime reconstruction

Exact byte-level SHA-256 image hashing reconstructed the following mutually exclusive test regimes:

| Regime | Count |
|---|---:|
| Train-to-test exact overlaps | 829 |
| Test-internal duplicate regime | 196 |
| Novel test images | 975 |
| **Total** | **2,000** |

The exact IDs are in `champion_01_regime_partition.csv` and the corresponding JSON artifact. The partition satisfies `829 + 196 + 975 = 2000`.

## 4. Validation methodology and results

### Overlap regime

The locked mechanism is the learned flip rule: predict `1 - train_label` for a test image with an exact train-image counterpart. A leave-one-member-as-query pseudo-hidden evaluation over all 1,458 train duplicate pairs produced:

- Members evaluated: 2,916
- Balanced Accuracy: **1.0000**

This is measured pseudo-hidden validation on the train duplicate structure, not a leaderboard score.

### Intra-test regime

The locked mechanism orders each duplicate pair by azimuth, bins the lower azimuth and absolute azimuth difference, predicts the lower member from the train-only `lo_delta` transition table, and infers the other member as its complement.

Five-fold grouped validation kept each duplicate pair together and fit transition frequencies only on the other pairs:

- Members evaluated: 2,916
- Balanced Accuracy: **0.8025**
- Fold BA standard deviation: **0.0212**

### Novel regime

The locked novel predictor is order-1 harmonic azimuth encoding (`sin(theta)`, `cos(theta)`) followed by the exact LightGBM configuration in `train.py`, with decision threshold `0.5`.

Validation used only train singletons (images without a train duplicate), with five stratified OOF folds. No validation labels were used to fit each fold:

- Members evaluated: 4,938 singleton images
- Ordinary grouped/OOF Balanced Accuracy: **0.7479**
- Class-0 recall: **0.7256**
- Class-1 recall: **0.7702**
- Fold BA standard deviation: **0.0156**
- Confusion matrix and per-fold metrics: `champion_01_novel_fold_metrics.csv`
- Threshold: **0.5**

The ordinary OOF result is the direct validation estimate for the locked novel model under this singleton protocol. It is not a hidden-test result.

## 5. Test-azimuth-distribution weighting

The novel test azimuth distribution was compared with the singleton validation pool in the eight 45-degree bins used by the repository. OOF observations were reweighted by:

`test-novel-bin proportion / singleton-validation-bin proportion`.

This is a distribution-weighted diagnostic, not a new fitted model and not a hidden-label estimate. It produced:

- Test-azimuth-weighted novel Balanced Accuracy: **0.7075**
- Test-azimuth-weighted class-0 recall: **0.4876**
- Test-azimuth-weighted class-1 recall: **0.9274**
- Weighted fold BA standard deviation: **0.0233**

Both ordinary and weighted values are retained; neither was selected opportunistically.

## 6. Regime-weighted estimates

Using the reconstructed regime counts and the measured regime validation values:

```text
ordinary =
  (829/2000)*1.0000
  + (196/2000)*0.8025
  + (975/2000)*0.7479
  = 0.8578

test-azimuth-weighted =
  (829/2000)*1.0000
  + (196/2000)*0.8025
  + (975/2000)*0.7075
  = 0.8381
```

**REGIME-WEIGHTED ESTIMATE — NOT LEADERBOARD PERFORMANCE**

- Ordinary validation regime-weighted estimate: **~0.8578**
- Test-azimuth-weighted novel regime estimate: **~0.8381**

These estimates combine train-derived pseudo-hidden/OOF evidence with the known structural regime counts. They do not use hidden test labels.

## 7. Uncertainty and limitations

Fold standard deviations are reported descriptively. No confidence interval is asserted: the hidden labels are unavailable, the real overlap and intra labels are unobserved, and the test-regime construction is not a random sample with a fully specified sampling model. The weighted novel result is sensitive to binning and covariate-shift assumptions.

The actual hidden-test Balanced Accuracy is unknown until the organizers score the submission.

## 8. Artifact inventory

- `champion_01_final_validation_report.md`
- `champion_01_hidden_ba_estimate.png`
- `champion_01_validation_metrics.csv`
- `champion_01_validation_metrics.json`
- `champion_01_novel_fold_metrics.csv`
- `champion_01_regime_partition.csv`
- `champion_01_regime_partition.json`
- `submission.csv`
- `model_weights.zip`
- `checksums.txt`

The requested publication-quality figure is:

`experiments/final_day/champions/champion_01_hidden_ba_estimate.png`

It explicitly labels both totals as estimates and not hidden scores.

## 9. Final statement

Our current regime-weighted estimate is **~0.8578** under ordinary validation and **~0.8381** when the novel validation is weighted to the observed novel-test azimuth distribution; the actual hidden-test BA remains unknown until the organizer scores the submission.
