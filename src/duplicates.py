"""Duplicate-regime mechanics trained strictly on train pairs.

Includes:
- pseudo-hidden overlap validation (train dup pairs -> hidden analogue)
- flip table by |delta azimuth|
- intra-pair rules A-E validated with hash-grouped folds
"""
import itertools
import numpy as np
import pandas as pd

from .azimuth import bin_index, circdiff


def flip_prob_table(pair_df, a_bins=(0, 45, 90, 135, 180, 360)):
    """P(test label = 1 - train label) by |delta azimuth| bin. Estimated on TRAIN
    duplicate pairs only (all conflicting by construction)."""
    d = np.clip(np.digitize(pair_df["abs_delta"].values, a_bins[1:-1], right=True),
                0, len(a_bins) - 2)
    out = pd.DataFrame({"bin": range(len(a_bins) - 1),
                        "range": [f"{a_bins[i]}-{a_bins[i+1]}" for i in range(len(a_bins) - 1)]})

    def _row(b):
        m = d == b
        if m.sum() == 0:
            return np.nan, 0
        return float((pair_df["y_B"].values[m] != pair_df["y_A"].values[m]).mean()), int(m.sum())

    out["flip_rate"], out["n"] = zip(*[_row(b) for b in range(len(a_bins) - 1)])
    return out


def dir_band_table(pair_df, az_bins=(0, 45, 90, 135, 180, 225, 270, 315, 360)):
    """P(lower-az member is class 1) keyed by (az_lo_band, az_hi_band)."""
    za = pair_df["az_A"].values
    zb = pair_df["az_B"].values
    yA = pair_df["y_A"].values
    yB = pair_df["y_B"].values
    a = np.argmin(np.vstack([za, zb]), axis=0)
    lo_band = np.where(a == 0, bin_index(za), bin_index(zb))
    hi_band = np.where(a == 0, bin_index(zb), bin_index(za))
    lo_is_1 = np.where(a == 0, yA, yB).astype(int)
    tab = {}
    for lo, hi in zip(lo_band, hi_band):
        key = (int(lo), int(hi))
        e = tab.setdefault(key, [0, 0])
        e[0] += 1
    for lo, hi, v in zip(lo_band, hi_band, lo_is_1):
        tab[(int(lo), int(hi))][1] += v
    return {k: (v[1] / max(v[0], 1)) for k, v in tab.items()}, tab


def direction_table_acc(tab_acc, n_seen):
    return {k: tab_acc[k] for k in n_seen}


# ---------------------------------------------------------------- pseudo-hidden overlap
def make_overlap_pairs(pair_ids_by_hash, members, rng):
    """Random directed overlap simulation on train duplicate pairs."""
    ph = list(pair_ids_by_hash)
    rng.shuffle(ph)
    rows = []
    for h in ph:
        a, b = members[h]
        if rng.random() < 0.5:
            rows.append((h, a, b))
        else:
            rows.append((h, b, a))
    return rows


def pseudo_hidden_overlap(df, pair_hashes, members, n_reps=20, seed=7, side=256):
    """For n_reps seeds, split train dup pairs into 'train analogue' + 'hidden query'
    and evaluate the flip mechanism. Returns per-seed per-bin BA rows."""
    from .metrics import balanced_accuracy_score  # noqa
    from sklearn.metrics import balanced_accuracy_score as bas
    lab = dict(zip(df["image_id"], df["label"]))
    azs = dict(zip(df["image_id"], df["azimuth"].astype(float)))
    rows = []
    for rep in range(n_reps):
        rng = np.random.default_rng(seed * 1000 + rep)
        pairs = make_overlap_pairs(pair_hashes, members, rng)
        y_true, p_pred, dbin = [], [], []
        for h, a, b in pairs:
            y_true.append(int(lab[b]))
            p_pred.append(1 - int(lab[a]))          # flip mechanism
            d = abs(circdiff([azs[a]], [azs[b]])[0])
            dbin.append(int(np.searchsorted([45, 90, 135], d, side="right")))
        y_true = np.array(y_true)
        p_pred = np.array(p_pred)
        dbin = np.array(dbin)
        rows.append({
            "rep": rep,
            "overall_ba": bas(y_true, p_pred),
            "ba_lt45": bas(y_true[dbin == 0], p_pred[dbin == 0]) if (dbin == 0).sum() else np.nan,
            "ba_45_90": bas(y_true[dbin == 1], p_pred[dbin == 1]) if (dbin == 1).sum() else np.nan,
            "ba_90_135": bas(y_true[dbin == 2], p_pred[dbin == 2]) if (dbin == 2).sum() else np.nan,
            "ba_gt135": bas(y_true[dbin == 3], p_pred[dbin == 3]) if (dbin == 3).sum() else np.nan,
            "n": len(y_true),
        })
    return pd.DataFrame(rows)


def overlap_predictions(part, df, az_by_id=None, az_test_by_id=None):
    """Build overlap_predictions.csv rows for the REAL 829 overlaps (no test labels used).
    Mechanism: learned flip (World C) with 1.0 confidence in train."""
    lab = df.set_index("image_id")["label"].to_dict()
    if az_by_id is None:
        az_by_id = df.set_index("image_id")["azimuth"].astype(float).to_dict()
    az = dict(az_by_id)
    if az_test_by_id:
        az.update(az_test_by_id)
    rows = []
    for h, ti, te in part["overlap_rows"]:
        d = abs(circdiff([az[ti]], [az[te]])[0])
        rows.append({
            "image_id": te, "source_train_id": ti, "source_label": int(lab[ti]),
            "source_azimuth": float(az[ti]), "test_azimuth": float(az[te]),
            "delta_azimuth": float(d), "flip_probability": 1.0,
            "predicted_label": int(1 - lab[ti]), "method": "learned_flip_world_c",
        })
    return pd.DataFrame(rows), rows


# ---------------------------------------------------------------- intra rules
def fit_direction_table_fold_aware(pair_df):
    """Direction table fitted on a subset of pairs (caller passes the fold-train pairs)."""
    tab, ns = dir_band_table(pair_df)
    return tab


def rule_predictions(pair_meta, tab_acc, az_pred_map, default=0.5):
    """Apply direction-table + az-model "predict one, infer the other" logic.
    pair_meta: list of (pair_id, id_lo, id_hi, az_lo, az_hi).
    Returns list of dicts {image_id, predicted_label, confidence, method}."""
    rows = []
    for pid, ilo, ihi, azlo, azhi in pair_meta:
        key = (int(bin_index([azlo])[0]), int(bin_index([azhi])[0]))
        acc = tab_acc.get(key, default)
        # joint assignment under 'opposite' assumption
        if acc >= 0.5:
            y_lo, y_hi = 1, 0          # lower-az member is class 1
        else:
            y_lo, y_hi = 0, 1
        rows.append({"pair_id": pid, "image_id": ilo, "predicted_label": int(y_lo),
                     "confidence": float(max(acc, 1 - acc)), "method": "joint_direction_table"})
        rows.append({"pair_id": pid, "image_id": ihi, "predicted_label": int(y_hi),
                     "confidence": float(max(acc, 1 - acc)), "method": "joint_direction_table"})
    return rows


def build_pair_table(part, df):
    """Construct unordered duplicate-pair rows from the partition (train only)."""
    lab = dict(zip(df["image_id"], df["label"]))
    azs = dict(zip(df["image_id"], df["azimuth"].astype(float)))
    rows = []
    for h, mem in part["train_dup_groups"].items():
        if len(mem) != 2:
            continue
        a, b = sorted(mem)
        rows.append({"hash": h, "id_A": a, "y_A": int(lab[a]), "az_A": float(azs[a]),
                     "id_B": b, "y_B": int(lab[b]), "az_B": float(azs[b]),
                     "abs_delta": float(abs(circdiff([azs[a]], [azs[b]])[0]))})
    return pd.DataFrame(rows)


def _fold_for(fold_by_id, r):
    return int(fold_by_id.get(r["id_A"], 0))


def intra_rules_cv(pair_df, p_az_map, fold_by_id, singleton_ids, az_by_id, lab_all,
                   n_folds=5, seed=42):
    """Cross-validated BA for intra-pair rules A-E on train dup pairs (all fit on
    fold-external pairs only). Returns (summary_rows, per_pair_predictions)."""
    from .metrics import balanced_accuracy_score as bas
    pair_df = pair_df.reset_index(drop=True)
    fold_vals = [int(fold_by_id.get(r["id_A"], 0)) for r in pair_df.to_dict("records")]
    pair_df["fold"] = fold_vals
    lab = dict(zip(pair_df["id_A"], pair_df["y_A"]))
    lab.update(dict(zip(pair_df["id_B"], pair_df["y_B"])))

    rules = {k: {} for k in ["A_flip_absdelta", "B_cond_bands", "C_nearest_az",
                             "D_predict_one", "E_joint"]}

    all_true, all_member_ids = [], []
    for k in range(n_folds):
        tr_pairs = pair_df[pair_df["fold"] != k]
        va_pairs = pair_df[pair_df["fold"] == k]
        tab, _ = dir_band_table(tr_pairs)
        pflip = tr_pairs.assign(
            flip=(tr_pairs["y_A"] != tr_pairs["y_B"]).astype(int),
            dbin=np.clip(np.digitize(tr_pairs["abs_delta"].values, [45, 90, 135]), 0, 3)
        )

        for _, r in va_pairs.iterrows():
            a, b = r["id_A"], r["id_B"]
            za, zb, ra, rb = r["az_A"], r["az_B"], r["y_A"], r["y_B"]
            lo = a if za <= zb else b
            hi = b if lo == a else a
            key = (int(bin_index([min(za, zb)])[0]), int(bin_index([max(za, zb)])[0]))
            acc = tab.get(key, 0.5)
            # A: constant flip (mechanism reference)
            rules["A_flip_absdelta"][a] = int(1 - ra)
            rules["A_flip_absdelta"][b] = int(1 - rb)
            # B: conditional direction table (lower-az->1 or not)
            y_lo_b = 1 if acc >= 0.5 else 0
            rules["B_cond_bands"][lo] = y_lo_b
            rules["B_cond_bands"][hi] = 1 - y_lo_b
            # E: joint assignment (same as B but with joint-likelihood confidence)
            rules["E_joint"][lo] = y_lo_b
            rules["E_joint"][hi] = 1 - y_lo_b
            # D: predict-one via az model -> infer other as opposite
            pa, pb = p_az_map.get(a, 0.5), p_az_map.get(b, 0.5)
            hi_d = a if pa >= pb else b
            lo_d = b if hi_d == a else a
            rules["D_predict_one"][hi_d] = 1
            rules["D_predict_one"][lo_d] = 0
            # C: nearest train analogue by azimuth (exclude pair's own members), then flip
            na = min((i for i in singleton_ids if i not in (a, b)),
                     key=lambda i: abs(az_by_id[i] - za))
            nb = min((i for i in singleton_ids if i not in (a, b)),
                     key=lambda i: abs(az_by_id[i] - zb))
            rules["C_nearest_az"][a] = int(lab_all[na])
            rules["C_nearest_az"][b] = int(lab_all[nb])
            all_true.append(ra); all_true.append(rb)
            all_member_ids.append(a); all_member_ids.append(b)

    # deterministic global rule for predictions on real test-intra (fitted on ALL pairs)
    tab_all, _ = dir_band_table(pair_df)

    summary = []
    for rname, pvec in rules.items():
        yt = np.array(all_true)
        yp = np.array([pvec[i] for i in all_member_ids])
        summary.append({"rule": rname, "BA": float(bas(yt, yp)), "n": len(yt)})
    return pd.DataFrame(summary), tab_all


def predict_test_intra(intra_groups, az_by_id, tab, default=0.5):
    """Apply the joint direction-table rule to the real 98 test-internal pairs."""
    rows = []
    for pid, (h, mem) in enumerate(sorted(intra_groups.items())):
        a, b = sorted(mem)
        za, zb = az_by_id[a], az_by_id[b]
        lo = a if za <= zb else b
        hi = b if lo == a else a
        key = (int(bin_index([min(za, zb)])[0]), int(bin_index([max(za, zb)])[0]))
        acc = tab.get(key, default)
        y_lo = 1 if acc >= 0.5 else 0
        rows.append({"pair_id": h, "image_id": lo, "predicted_label": int(y_lo),
                     "confidence": float(max(acc, 1 - acc)), "method": "E_joint_direction"})
        rows.append({"pair_id": h, "image_id": hi, "predicted_label": int(1 - y_lo),
                     "confidence": float(max(acc, 1 - acc)), "method": "E_joint_direction"})
    return pd.DataFrame(rows)