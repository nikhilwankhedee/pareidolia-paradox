# PAREIDOLIA PARADOX — CHAMPION REGISTRY (PHASE 0 AUDIT)

This document and directory preserve the baseline and current best candidate predictions, ensuring protected checkpoints that are NEVER overwritten.

---

## 1. Champion 00: Baseline Deployed Submission (`final_campaign`)

- **File**: `experiments/final_day/champions/champion_00_baseline/final_submission.csv`
- **SHA-256 Checksum**: `f3ecf99d72df95c4ee38314c4ff6a43f6ee82d59a923c9297c5d62a7b761e5aa`
- **Total Predictions**: 2,000 rows
- **Columns**: `image_id,label`
- **Class Counts**: Class 0: 654 (32.7%), Class 1: 1,346 (67.3%)
- **Source Pipeline**: `final_campaign/final_campaign.ipynb` (executed 2026-09-18 12:35 UTC)
- **Regime Architecture**:
  - Overlap (829 images): World C learned flip rule ($y_{test} = 1 - y_{train}$) → 1.0000 on train duplicates.
  - Intra-test (196 images / 98 pairs): Direction table rule E_joint → CV BA 0.7942.
  - Novel (975 images): A2SEL retrieval fusion: $p = 0.65 \cdot p_{az} + 0.35 \cdot \text{sim}_{\text{raw48}}$.
- **Audited Defect**: The raw48 top-1 cosine similarity is always positive (~0.7–1.0), shifting novel predictions heavily toward class 1. Honest distribution-matched proxy evaluation shows novel BA of only 0.5600 (full train OOF 0.5404), far below the standalone azimuth component (0.7343 proxy / 0.7825 full).
- **Validation Metric**: Reported optimistic regime-weighted estimate was 0.8738 (crediting novel with 0.7825), but true honest proxy estimate was **0.7653**.

---

## 2. Champion 01: Current Front-Runner (`candidate_v1_azfix`)

- **File**: `experiments/final_day/champions/champion_01_candidate_v1_azfix/candidate_v1_azfix.csv`
- **SHA-256 Checksum**: `e19b624b979286a47da42a270e3fcfc6e5bb697849fd2092793ecaac50fc3b41`
- **Total Predictions**: 2,000 rows
- **Columns**: `image_id,label`
- **Class Counts**: Class 0: 768 (38.4%), Class 1: 1,232 (61.6%)
- **Source Pipeline**: `experiments/improvement/exp9_assemble.py` and `exp10_candidates.py`
- **Regime Architecture**:
  - Overlap (829 images): Learned flip (World C) — identical to Champion 00 (0 differences).
  - Intra-test (196 images / 98 pairs): `lo_delta` (az_lo-band $\times$ delta-band) transition model → Grouped CV BA **0.8035** (vs E_joint 0.7805/0.7942), 26 pair-swaps.
  - Novel (975 images): Validated pure azimuth component ($p_{az} \ge 0.5$). Fixes the +0.35 similarity bias, correcting **114 novel predictions** (all where $\text{azimuth} \ge 270^\circ$, changing 1 → 0).
- **Validation Metrics**:
  - Overlap: 1.0000 (829 images)
  - Intra: 0.8035 (196 images)
  - Novel: 0.7343 (975 images, azimuth-matched proxy)
  - **Regime-weighted Honest Estimate**: **0.8512** (+0.0859 over Champion 00). Under the campaign's optimistic convention (novel = 0.7825 train-OOF): **0.8747**.
- **Prediction Diff vs Champion 00**: 140 / 2000 differences (7.0%). Exactly 0 overlap changes, 114 novel changes, 26 intra pair swaps.

---

## 3. Champion 02: Conservative Candidate (`candidate_v2_novelfix_only`)

- **File**: `experiments/final_day/champions/champion_02_candidate_v2_novelfix/candidate_v2_novelfix_only.csv`
- **SHA-256 Checksum**: `d0c51f5f09382e03aad53caef88e6a1db47dbc1c9c44027aa6f988607998136d`
- **Total Predictions**: 2,000 rows
- **Class Counts**: Class 0: 768 (38.4%), Class 1: 1,232 (61.6%)
- **Diff vs Champion 00**: Exactly 114 novel changes; intra-test and overlap are identical to Champion 00.

---

## 4. Frozen File Index

| Identifier | Checksum (SHA-256) | File Path |
|---|---|---|
| Champion 00 Baseline | `f3ecf99d72df95c4ee38314c4ff6a43f6ee82d59a923c9297c5d62a7b761e5aa` | [final_submission.csv](file:///home/nikhil/projects/ieee-paradox-comp/The%20Pareidolia%20Paradox%20Dataset/experiments/final_day/champions/champion_00_baseline/final_submission.csv) |
| Champion 01 (v1 azfix) | `e19b624b979286a47da42a270e3fcfc6e5bb697849fd2092793ecaac50fc3b41` | [candidate_v1_azfix.csv](file:///home/nikhil/projects/ieee-paradox-comp/The%20Pareidolia Paradox Dataset/experiments/final_day/champions/champion_01_candidate_v1_azfix/candidate_v1_azfix.csv) |
| Champion 02 (v2 novelfix) | `d0c51f5f09382e03aad53caef88e6a1db47dbc1c9c44027aa6f988607998136d` | [candidate_v2_novelfix_only.csv](file:///home/nikhil/projects/ieee-paradox-comp/The%20Pareidolia%20Paradox%20Dataset/experiments/final_day/champions/champion_02_candidate_v2_novelfix/candidate_v2_novelfix_only.csv) |
