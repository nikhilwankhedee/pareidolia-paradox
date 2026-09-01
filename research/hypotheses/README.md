# Hypotheses (living document)

Research question: what drives the "Depth vs Rise" label in the Pareidolia
Paradox dataset, and where does the predictive signal live?

## H1 (initial, discarded by measurement)
The label is a deterministic function of the raw pixels alone, and any apparent
azimuth signal is an artifact (pure confound).

**Status:** Rejected. Exact-duplicate analysis (`experiments/duplicate_audit`)
showed pixel-identical images labeled both 0 and 1. The label is therefore
NOT a function of pixels alone.

## H2
The label is determined by the sun azimuth angle (illumination geometry), not
solely by surface shape. Azimuth encodes the lighting direction that makes
mounds look like craters / craters look like mounds under a particular convention.

**Status:** Supported as a strong signal. Azimuth-only sin/cos logistic
regression reaches ~0.776 grouped-stratified OOF balanced accuracy.

## H3
The organizer's official rotation `rotate(image, -azimuth)` canonicalizes the
contradictory duplicates (makes identical pixels align so the label difference
becomes geometrically explainable).

**Status:** Rejected by `experiments/rotation_audit`. Canonicalization does NOT
resolve the label contradiction (mean corr ~0.29, all four conventions
statistically identical).

## H4 (Experiment 3, in progress)
The predictive signal decomposes as:
    AZIMUTH  +  PIXELS  +  AZIMUTH x PIXELS  +  ROTATION
and we must measure each term on leakage-free grouped folds (Models A-E).
