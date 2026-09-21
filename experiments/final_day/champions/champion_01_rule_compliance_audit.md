# Champion 01 Competition-Rule Compliance Audit

**Audit date:** 2026-09-21  
**Champion:** `candidate_v1_azfix`  
**Locked commit:** `94e064d80d1d6d69d1efbba3f2534089721cb8bd`  
**Scope:** Rule/compliance review only. `submission.csv`, `model_weights.zip`, `train.py`, and `inference.py` were not modified. No submission was made.

## Executive conclusion

No official competition rulebook, problem-statement PDF, organizer announcement, or host-specific integrity/submission policy was located in the repository or through the repository's available public links. The repository contains participant-authored documentation and experiment notes, not authoritative organizer rules.

The method demonstrably hashes the supplied train and evaluation images, detects exact train-to-evaluation overlaps, uses the supplied `sun_azimuth_angle`, and derives predictions from train labels. The implementation does not access evaluation labels, external label sources, or a programmatic submission endpoint.

However, exact train-to-evaluation overlap exploitation is a material rule-interpretation issue. It may be permitted if the evaluation images and metadata are supplied for local inference and no rule bans duplicate detection; it may be prohibited by an unlocated anti-leakage or competition-integrity rule. **No explicit prohibition was located in the reviewed materials; organizer interpretation remains unresolved.**

**Final submission recommendation: AMBIGUOUS — ask organizer.**

This is not a claim that the method is compliant or non-compliant. The organizer should explicitly confirm whether using exact duplicates between the supplied training and evaluation image files, including train-label-derived inversion rules, is allowed.

## Materials reviewed

### Repository and locked release

1. Public repository: <https://github.com/nikhilwankhedee/pareidolia-paradox>
2. Locked commit tree: <https://github.com/nikhilwankhedee/pareidolia-paradox/tree/94e064d80d1d6d69d1efbba3f2534089721cb8bd>
3. Participant README: `README.md`, locked commit, sections “Methodology & Treatment of `sun_azimuth_angle`”, “Step-by-Step Execution & Reproducibility”, and “Final Submission Verification”.
4. Participant data notes: `data/README.md`
5. Participant Kaggle setup guide: `research/results/KAGGLE_SETUP_GUIDE.md`
6. Participant package notes: `pareidolia_experiment3_kaggle/README.txt`
7. Participant duplicate audit: `experiments/duplicate_audit/duplicate_audit_report.txt`
8. Locked inference implementation: `inference.py`
9. Locked training implementation: `train.py`

The GitHub repository metadata and tree were also checked through the public GitHub API:

- <https://api.github.com/repos/nikhilwankhedee/pareidolia-paradox>
- <https://api.github.com/repos/nikhilwankhedee/pareidolia-paradox/git/trees/94e064d80d1d6d69d1efbba3f2534089721cb8bd?recursive=1>

### Search result for official rules

The repository contains no PDF files and no file named or clearly identified as official rules, problem statement, organizer announcement, or host submission policy. The only competition-facing documents found are participant-authored README/setup/research files listed above.

The public web search for the exact competition title and rules did not identify an authoritative organizer rules page. The project repository itself is not an organizer rule source.

Accordingly, there is no authoritative rule text to quote for a prohibition or permission on duplicate detection, transductive inference, metadata use, or train/evaluation overlap exploitation.

## Exact relevant text located

The following quotations are included for evidence, but they are **participant-authored descriptions, not official rules**.

### Problem inputs and supplied evaluation data

Source: `README.md`, locked commit, “Competition Dataset & Metric”:

> “**Train Set**: 7,854 images (256×256 grayscale, R==G==B).”

> “**Test Set**: 2,000 hidden images.”

> “**Crucial Metadata**: `sun_azimuth_angle` (degrees in $[0^\circ, 360)$).”

Source: `pareidolia_experiment3_kaggle/README.txt`:

> “`test_metadata.csv` — (image_id, sun_azimuth_angle) — 2000 rows, **NO labels**”

These establish what the participant package says was supplied; they do not establish an organizer permission to inspect relationships among supplied evaluation images.

### Duplicate and overlap findings

Source: `README.md`, “Methodology & Treatment of `sun_azimuth_angle`”:

> “829 test images share identical SHA-256 byte hashes with images in the training set.”

> “**Rule**: For each test image $t$ matching training image $s$: $$y_t = 1 - y_s$$”

Source: `experiments/duplicate_audit/duplicate_audit_report.txt`, section “B. TRAIN <-> TEST EXACT DUPLICATE AUDIT”:

> “Cross-split shared hashes: 829”

Source: `experiments/duplicate_audit/duplicate_audit_report.txt`, section “C. TRAIN DUPLICATE LABEL CONSISTENCY”:

> “ALL 1458 train duplicate groups are label-CONFLICTING.”

These are participant findings and the exact method under review. They are not organizer language authorizing or forbidding the practice.

### Leakage language in the repository

Source: `README.md`, “Validation & Performance Summary”:

> “All validation is strictly grouped by duplicate hash to prevent data leakage.”

Source: `research/results/KAGGLE_SETUP_GUIDE.md`:

> “grouped by exact sha256 image hash, leakage-free (0 hashes cross folds), class-balanced.”

This use of “leakage-free” refers to validation-fold construction. It does not resolve whether discovering a train/evaluation duplicate is permitted under competition rules.

### Submission instructions located

Source: `README.md`, “Step-by-Step Execution & Reproducibility”:

> “Run `inference.py` to load the saved artifacts, partition the test images, and generate the final `submission.csv`”

Source: `README.md`, “Final Submission Verification (QA)”:

> “Exactly 2,000 prediction rows (+ 1 header row)”

> “Exact columns: `image_id,label`”

These are repository packaging instructions, not official host submission rules.

## Method-by-method compliance analysis

| Method or behavior | Evidence in locked implementation | Explicitly permitted | Explicitly prohibited | Status |
|---|---|---|---|---|
| Hash supplied train images | `train.py` computes SHA-256 over files under the supplied train directory | No official rule located | No official rule located | Not addressed / ambiguous |
| Hash supplied evaluation images | `inference.py` computes SHA-256 over files under the supplied test directory | No official rule located | No official rule located | Not addressed / ambiguous |
| Detect exact train-to-evaluation image overlap | `inference.py` intersects evaluation hashes with `train_hash_lookup.json` | No official rule located | No official rule located | **Material ambiguity** |
| Use supplied `sun_azimuth_angle` | `train.py` and `inference.py` read `sun_azimuth_angle` / `azimuth`; README identifies it as crucial metadata | It is supplied as a competition input; no contrary rule located | No official rule located | Not addressed / likely ordinary input use, but not formally confirmed |
| Derive overlap predictions from train labels | `inference.py` applies `1 - train_label` for overlap hashes | No official rule located | No official rule located | **Material ambiguity** |
| Use test-internal duplicate relationships | `inference.py` groups evaluation hashes and uses a train-derived transition table | No official rule located | No official rule located | Not addressed / ambiguous |
| Access evaluation labels | `test_metadata.csv` has no labels according to participant package; no evaluation-label file or code path was found | No evidence of access | No evidence of access | No violation found in reviewed materials |
| Query external labels or reverse-image services | No network/API/reverse-search code appears in locked train/inference pipeline | No official rule located | No official rule located | No violation found in reviewed materials |
| Use external data | Locked pipeline consumes local supplied train/evaluation files and serialized artifacts; no external data source is queried | No official rule located | No official rule located | No violation found in reviewed materials |
| Programmatically submit | `inference.py` writes a local CSV and contains no submission API/client | No official rule located | No official rule located | No violation found in reviewed materials |

## Requested search terms and findings

The repository and available public project materials were searched for:

- `data leakage`, `leakage`
- `duplicate images`, `duplicate detection`
- `train/test overlap`
- `test-set inspection`
- `external data`, `external labels`
- `exploiting metadata`
- `reverse image search`
- `hidden labels`
- `prohibited techniques`
- `transductive learning`
- `competition integrity`
- `disqualification`

Findings:

1. “Leakage” appears in participant validation documentation, primarily to describe grouped cross-validation.
2. Duplicate detection and train/evaluation overlap are documented as participant discoveries.
3. No authoritative text says that duplicate detection is allowed.
4. No authoritative text says that duplicate detection, overlap exploitation, metadata use, transductive inference, or test-set inspection is prohibited.
5. No hidden labels, external labels, reverse-image queries, or programmatic submission code was found in the locked inference/training path.

## Risk assessment

### Lower-risk aspects

- Reading the supplied `sun_azimuth_angle` field as a model feature.
- Training from supplied training images and labels.
- Hashing local files for reproducibility and deduplication.
- Producing a local CSV in the required shape.
- Not querying external label sources.
- Not accessing evaluation labels.
- Not submitting programmatically.

### High-risk unresolved aspect

The overlap branch is intentionally transductive with respect to the supplied evaluation images: it identifies evaluation images that are byte-identical to training images and uses the corresponding training label to predict the evaluation label, with an inversion rule learned from train duplicates. The test-internal branch likewise uses relationships among evaluation images.

Even if the files were intentionally supplied and the labels are not accessed, an organizer may regard this as exploiting a dataset construction artifact or as test-set inspection. The reviewed materials do not answer that question.

## Required organizer clarification

Ask the organizer, in writing, a specific question such as:

> “Are participants permitted to compute exact hashes of the supplied training and evaluation image files, identify byte-identical train-to-evaluation or evaluation-internal duplicates, and use labels/statistics learned from the training duplicates to make predictions for those evaluation images, provided that no evaluation labels or external label sources are accessed?”

Retain the organizer’s response with the submission records. If the organizer disallows this behavior, Champion 01 should not be submitted in its current form.

## Final recommendation

**AMBIGUOUS — ask organizer.**

No explicit prohibition was located in the reviewed materials; organizer interpretation remains unresolved.

