# Pareidolia Paradox — Improvement Campaign Report (gen-2)

All validation is grouped by duplicate hash. No test labels or leaderboard were used.
The current `final_campaign` submission was never modified; candidates are separate.

## Headline

**The single biggest improvement is an audited fix to the NOVEL regime, not a new model.**
The campaign fused the novel azimuth probability with the raw48 top-1 *cosine similarity*
(`p = 0.65·p_az + 0.35·sim`). Similarity is always positive/high, so it biases borderline
predictions toward class 1. On honest grouped OOF and on an azimuth-matched novel proxy the
deployed fusion scores **0.54 / 0.56**, far below the azimuth component it was supposed to
dominate (**0.7825 / 0.7343**). The decision table credited novel with the azimuth component's
0.7825, so the deployed submission was materially worse than the recorded 0.8738 estimate.

Replacing the novel fusion with the validated azimuth component changes **114 novel predictions**
(all `az≥270°`, current 1 → candidate 0) and restores the intended performance.

## Experiment verdicts

| # | Experiment | Best honest result | Baseline | Verdict |
|---|---|---|---|---|
| 1 | Lunar (az + solar geometry) | 0.7816 opt / 0.7637 singleton | az 0.7836 / 0.7823 | **REJECT** (AUC 0.761 vs 0.787; stack 0.778) |
| 2 | Illumination-invariant retrieval + transition | stack 0.7817 full; highpass neighbour label 0.361→flip 0.639 | az 0.7825 | **REJECT** for novel |
| 3 | Joint structural + azimuth (148 feats, LGBM/LR) | 0.7391 full / 0.7074 proxy | az 0.7556 / 0.7103 (same fitter) | **REJECT** (structure adds nothing beyond az) |
| 4 | Intra pair transition model (`lo_delta`) | 0.8035 (15 grouped resamples) | E_joint 0.7805 / 0.7942 | **PROMOTE** (small) |
| 5 | Duplicate forensics (1458 pairs) | `lo_delta` grid; deterministic window fails (0.51) | — | informs #4 |
| 6 | Novel-regime simulation | az 0.7343; every structural/lunar/fusion ≤ az | — | no structural gain |
| 7 | Azimuth model/threshold bakeoff | step270 0.7356 proxy (vs az LGBM 0.7343) | — | marginal (< +0.008 gate) |
| 8 | **Novel fusion audit/fix** | az 0.7343 proxy (vs deployed 0.5600) | +0.174 novel | **PROMOTE (best)** |

Key mechanistic conclusions:
- `P(label=1|azimuth)` is a clean step at **270°** (≈0.88 below, ≈0.35 above); this is the dominant signal.
- Raw/illumination-invariant retrieval is anti-correlated with the label (0.35–0.36) because the
  nearest image is usually the same terrain under a different azimuth. Flipping recovers ~0.64,
  but only for duplicate-associated images — it collapses to chance (0.52) on true novel images.
- Image content adds no usable signal beyond azimuth (raw CNN ≈ chance, FiLM ≈ az, structural ≈ az).

## 1. Best new experiment
**EXP8 — Novel fusion audit and fix.** The deployed `0.65·p_az + 0.35·top1_similarity` fusion was
replaced by the validated azimuth component (`p_az ≥ 0.5`), selected on the azimuth-matched proxy.
`step270` over az leads to **identical** test-novel labels (0 differing ids).

## 2. Honest OOF improvement
- Novel regime: matched-proxy BA **0.5600 → 0.7343** (+0.174); full-train OOF of the deployed
  fusion 0.5404 vs azimuth 0.7825.
- Intra regime: grouped pair BA **E_joint 0.7805 → lo_delta 0.8035** (+0.023, 15 resamples;
  5-fold variant 0.7942→0.8032).
- Overlap regime: unchanged (1.0000).

## 3. Estimated overall BA
Honest, regime-weighted using the azimuth-matched novel proxy:

| regime | n | BA | contribution |
|---|---|---|---|
| overlap | 829 | 1.0000 | 829.0 |
| intra | 196 | 0.8035 | 157.5 |
| novel | 975 | 0.7343 | 715.9 |
| **overall** | **2000** | **0.8512** | 1702.4 |

Current deployed submission, same basis (novel fusion = 0.5600): **0.7653** → **+0.0859**.
Under the campaign's own (optimistic) convention (novel = 0.7825 train-OOF): candidate **0.8747**.
None of these is a leaderboard claim.

## 3b. EXP41 — intra second-signal audit (final, honest)
Same grouped-OOF protocol as the EXP4 gate (GroupKFold over dup hash; member-level BA;
15 grouped resamples). No intra rule could honestly beat the v1 `lo_delta` intra reference:

| rule | grouped-OOF BA_member |
|---|---|
| global majority | 0.7963 |
| `lo_delta` (v1 intra) | **0.8032 — gate** |
| `lo_delta_banded` | 0.7778 |
| `small_delta_banded` | 0.7840 |
| `mid` | 0.7956 |

**Verdict: NO intra promote.** `lo_delta` stays the only intra mechanism; novel+overlap
unchanged. `outputs/exp41_intra_ranked.csv` written.

## 4. Which regime improved
- **Novel (975): large, validated** — fusion fix, +114 corrections.
- **Intra (196): small, validated** — `lo_delta` transition, 26 pair-swaps.
- **Overlap (829): untouched** — no superior mechanism found.

## 5. Exact prediction changes vs current submission
- Total **140 / 2000** (7.0%). Overlap **0**.
- Novel **114**: every change is `azimuth ≥ 270°`, current `1` → candidate `0` (no `az<270` change).
- Intra **26**: labels swapped pairwise (98 class-1 kept per side; class counts unchanged).
- Class counts: current `{0:654, 1:1346}` → candidate `{0:768, 1:1232}`.

## 6. Should the new candidate replace the current submission?
**Yes — recommend `candidate_v1_azfix.csv`** (novel fix + intra `lo_delta`).
- It fixes a demonstrable defect in the deployed novel predictor (validated, no leakage).
- Alternatively `candidate_v2_novelfix_only.csv` makes only the novel change (novel-only diff of 114).
- Do **not** replace before a final end-to-end rerun if the packet must stay self-contained;
  the existing `final_campaign` package is untouched and remains the safe rollback.

## 7. Files produced
- `experiments/improvement/research.py` — harness (baseline az OOF 0.7825 reproduced exactly).
- `exp1_lunar.py`, `exp2_retrieval.py`, `exp3_joint.py`, `exp5_forensics.py`, `exp5b_compare.py`,
  `exp6_sim.py`, `exp7_azbakeoff.py`, `exp8_fusion_fix.py`, `exp9_assemble.py`, `exp10_candidates.py`
- `structfeat.py` — illumination-invariant structural features (148) + caches.
- `outputs/exp*.csv`, `outputs/exp*.npz`, `outputs/*_run.log`
- `outputs/candidates/candidate_v1_azfix.csv` — **recommended candidate**
- `outputs/candidates/candidate_v2_novelfix_only.csv` — conservative variant
- `outputs/candidate_changes_vs_current.csv` — full 140-row diff
- `outputs/candidate_v1_estimate.csv` — regime estimate table

## Integrity
- 2000 predictions; ids exactly aligned to `test_metadata`; labels ⊆ {0,1}; no nulls.
- No test labels touched; test azimuth/image_id metadata only.
- Overlap labels byte-identical to current submission.
- Retrieval OOF masks self **and** same-hash/same-fold neighbours (no self-retrieval, no twin leakage).
- Candidates written under `outputs/candidates/`; `final_campaign/` unchanged.
