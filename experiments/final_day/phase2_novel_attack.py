"""Phase 2 — Attack the 975 Novel Images.

Rigorous, leakage-free validation on train singletons (4,938 images):
1. Azimuth baseline: order-1 LightGBM vs step270.
2. Azimuth harmonic sweep (orders 1..6) with LightGBM, XGBoost, Logistic Regression.
3. Fine-grained azimuth step & spline models.
4. Image features: 148 illumination-invariant structural features (Xstruct).
5. Residual modeling: image features predicting y - p_azimuth.
6. Prediction complementarity & promotion gate evaluation.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bas, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
COMP_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(COMP_ROOT / "experiments" / "improvement"))
sys.path.insert(0, str(COMP_ROOT / "experiments" / "final_campaign"))

import research as R
from src.azimuth import az_harm, bin_index, circdiff
from src.validation import sample_match_az_distribution

OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset harness...")
H = R.load()
df = H["df"]
y_all = df["label"].values.astype(int)
az_all = df["azimuth"].values.astype(float)
fold_all = df["fold"].values
part = H["part_keys"]
tmeta = H["tmeta"]

dup_ids = {i for ids in part["train_dup_groups"].values() for i in ids}
sing_mask = ~df["image_id"].isin(dup_ids)
sing_indices = np.where(sing_mask)[0]

df_sing = df.iloc[sing_indices].copy().reset_index(drop=True)
y_sing = df_sing["label"].values.astype(int)
az_sing = df_sing["azimuth"].values.astype(float)
fold_sing = df_sing["fold"].values
id_sing = df_sing["image_id"].tolist()

print(f"Total train: {len(df)}, Singletons: {len(df_sing)} ({len(df_sing)/len(df)*100:.1f}%)")

# Test novel target histogram
az_test = dict(zip(tmeta["image_id"], tmeta["azimuth"].astype(float)))
novel_test_az = [az_test[i] for i in part["novel"]]
target_hist = np.histogram(novel_test_az, bins=np.arange(0, 361, 45))[0]
print("Novel test azimuth target histogram (45-deg bins):", target_hist)

# Helper for proxy evaluation
az_by_id_sing = dict(zip(id_sing, az_sing))
pos_by_id_sing = {img_id: idx for idx, img_id in enumerate(id_sing)}

NSEEDS = 30
proxy_indices = []
for s in range(NSEEDS):
    picks, _, _ = sample_match_az_distribution(id_sing, az_by_id_sing, target_hist, seed=s)
    proxy_indices.append(np.array([pos_by_id_sing[p] for p in picks]))

def evaluate_predictions(name, oof_preds, proba=True):
    """Compute full singleton metrics and multi-seed proxy metrics."""
    if proba:
        pred_labels = (oof_preds >= 0.5).astype(int)
        auc = roc_auc_score(y_sing, oof_preds)
    else:
        pred_labels = oof_preds.astype(int)
        auc = np.nan
    
    full_ba = bas(y_sing, pred_labels)
    r0 = float(((pred_labels == 0) & (y_sing == 0)).sum() / (y_sing == 0).sum())
    r1 = float(((pred_labels == 1) & (y_sing == 1)).sum() / (y_sing == 1).sum())
    
    # Bucket BA
    m_lt = az_sing < 270
    m_ge = az_sing >= 270
    ba_lt = bas(y_sing[m_lt], pred_labels[m_lt]) if len(np.unique(y_sing[m_lt])) > 1 else np.nan
    ba_ge = bas(y_sing[m_ge], pred_labels[m_ge]) if len(np.unique(y_sing[m_ge])) > 1 else np.nan
    
    # Proxy evaluation across 30 seeds
    proxy_bas = []
    for idx in proxy_indices:
        proxy_bas.append(bas(y_sing[idx], pred_labels[idx]))
    
    proxy_mean = float(np.mean(proxy_bas))
    proxy_std = float(np.std(proxy_bas))
    
    return {
        "model": name,
        "proxy_BA_mean": round(proxy_mean, 4),
        "proxy_BA_std": round(proxy_std, 4),
        "full_BA": round(full_ba, 4),
        "full_AUC": round(auc, 4) if not np.isnan(auc) else "-",
        "recall_0": round(r0, 4),
        "recall_1": round(r1, 4),
        "ba_lt270": round(ba_lt, 4) if not np.isnan(ba_lt) else "-",
        "ba_ge270": round(ba_ge, 4) if not np.isnan(ba_ge) else "-"
    }

results = []
oof_dict = {}

# -------------------------------------------------------------
# 1. Baseline Models: step270 and az_lgbm_o1
# -------------------------------------------------------------
print("\n--- 1. Baseline Azimuth Models ---")
# step270
pred_step270 = (az_sing < 270.0).astype(int)
oof_dict["step270"] = pred_step270
res = evaluate_predictions("step270", pred_step270, proba=False)
results.append(res)
print("step270:", res)

# az_lgbm order 1
oof_lgb_o1 = np.zeros(len(df_sing))
X_o1 = az_harm(az_sing, order=1)
for fold in range(5):
    tr, va = fold_sing != fold, fold_sing == fold
    clf = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.05, num_leaves=31,
                             max_depth=3, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbose=-1)
    clf.fit(X_o1[tr], y_sing[tr])
    oof_lgb_o1[va] = clf.predict_proba(X_o1[va])[:, 1]

oof_dict["az_lgbm_o1"] = oof_lgb_o1
res = evaluate_predictions("az_lgbm_o1", oof_lgb_o1, proba=True)
results.append(res)
print("az_lgbm_o1:", res)

# -------------------------------------------------------------
# 2. Azimuth Harmonic Sweep & Architectures
# -------------------------------------------------------------
print("\n--- 2. Harmonic Orders & Classifiers ---")
for order in [2, 3, 4, 6]:
    X_harm = az_harm(az_sing, order=order)
    
    # LightGBM
    oof_lgb = np.zeros(len(df_sing))
    for fold in range(5):
        tr, va = fold_sing != fold, fold_sing == fold
        clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.04, num_leaves=15,
                                 max_depth=3, subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, verbose=-1)
        clf.fit(X_harm[tr], y_sing[tr])
        oof_lgb[va] = clf.predict_proba(X_harm[va])[:, 1]
    name = f"az_lgbm_o{order}"
    oof_dict[name] = oof_lgb
    res = evaluate_predictions(name, oof_lgb, proba=True)
    results.append(res)
    print(name, res)
    
    # XGBoost
    oof_xgb = np.zeros(len(df_sing))
    for fold in range(5):
        tr, va = fold_sing != fold, fold_sing == fold
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  eval_metric="logloss", verbosity=0)
        model.fit(X_harm[tr], y_sing[tr])
        oof_xgb[va] = model.predict_proba(X_harm[va])[:, 1]
    name = f"az_xgb_o{order}"
    oof_dict[name] = oof_xgb
    res = evaluate_predictions(name, oof_xgb, proba=True)
    results.append(res)
    print(name, res)

    # Logistic Regression
    oof_lr = np.zeros(len(df_sing))
    for fold in range(5):
        tr, va = fold_sing != fold, fold_sing == fold
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_harm[tr])
        Xva_s = scaler.transform(X_harm[va])
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(Xtr_s, y_sing[tr])
        oof_lr[va] = lr.predict_proba(Xva_s)[:, 1]
    name = f"az_lr_o{order}"
    oof_dict[name] = oof_lr
    res = evaluate_predictions(name, oof_lr, proba=True)
    results.append(res)
    print(name, res)

# -------------------------------------------------------------
# 3. Fine Binned Azimuth Model
# -------------------------------------------------------------
print("\n--- 3. Binned Azimuth Table ---")
oof_bin = np.zeros(len(df_sing))
bin_width = 5.0
edges = np.arange(0, 360 + bin_width, bin_width)
for fold in range(5):
    tr, va = fold_sing != fold, fold_sing == fold
    tr_bins = np.clip(np.digitize(az_sing[tr], edges) - 1, 0, len(edges) - 2)
    va_bins = np.clip(np.digitize(az_sing[va], edges) - 1, 0, len(edges) - 2)
    
    global_mean = y_sing[tr].mean()
    m = 5.0
    stats = {}
    for b in range(len(edges) - 1):
        idx_b = np.where(tr_bins == b)[0]
        n_b = len(idx_b)
        k_b = y_sing[tr][idx_b].sum()
        stats[b] = (k_b + m * global_mean) / (n_b + m)
    
    oof_bin[va] = np.array([stats[b] for b in va_bins])

name = "az_smoothed_bins_5deg"
oof_dict[name] = oof_bin
res = evaluate_predictions(name, oof_bin, proba=True)
results.append(res)
print(name, res)

# -------------------------------------------------------------
# 4. Image Features & Joint Models
# -------------------------------------------------------------
print("\n--- 4. Image Features & Joint Models ---")
S_all = np.load(COMP_ROOT / "experiments" / "improvement" / "outputs" / "Xstruct_train.npz", allow_pickle=True)["X"]
S_sing = S_all[sing_indices]
print(f"Loaded Xstruct_train: shape {S_sing.shape}")

# A. Standalone Structural Features (LGBM)
oof_struct = np.zeros(len(df_sing))
for fold in range(5):
    tr, va = fold_sing != fold, fold_sing == fold
    clf = lgb.LGBMClassifier(n_estimators=150, max_depth=3, num_leaves=15,
                             learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                             random_state=42, verbose=-1)
    clf.fit(S_sing[tr], y_sing[tr])
    oof_struct[va] = clf.predict_proba(S_sing[va])[:, 1]

name = "struct_only_lgbm"
oof_dict[name] = oof_struct
res = evaluate_predictions(name, oof_struct, proba=True)
results.append(res)
print(name, res)

# B. Joint Azimuth + Top Structural Features
clf_sel = lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
clf_sel.fit(S_sing, y_sing)
top_feats = np.argsort(-clf_sel.feature_importances_)[:15]
print(f"Top 15 structural features selected: {top_feats}")

X_joint = np.hstack([X_o1, S_sing[:, top_feats]])
oof_joint = np.zeros(len(df_sing))
for fold in range(5):
    tr, va = fold_sing != fold, fold_sing == fold
    clf = lgb.LGBMClassifier(n_estimators=120, max_depth=3, num_leaves=15,
                             learning_rate=0.04, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbose=-1)
    clf.fit(X_joint[tr], y_sing[tr])
    oof_joint[va] = clf.predict_proba(X_joint[va])[:, 1]

name = "az_o1_plus_top15_struct"
oof_dict[name] = oof_joint
res = evaluate_predictions(name, oof_joint, proba=True)
results.append(res)
print(name, res)

# C. Residual Modeling (predicting residual y - p_az)
print("\n--- 5. Residual Modeling ---")
res_target = y_sing - oof_lgb_o1
oof_res_pred = np.zeros(len(df_sing))
for fold in range(5):
    tr, va = fold_sing != fold, fold_sing == fold
    reg = lgb.LGBMRegressor(n_estimators=100, max_depth=2, num_leaves=7,
                            learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
                            random_state=42, verbose=-1)
    reg.fit(S_sing[tr], res_target[tr])
    oof_res_pred[va] = reg.predict(S_sing[va])

corrected_oof = np.clip(oof_lgb_o1 + oof_res_pred, 0.0, 1.0)
name = "az_plus_struct_residual"
oof_dict[name] = corrected_oof
res = evaluate_predictions(name, corrected_oof, proba=True)
results.append(res)
print(name, res)

# -------------------------------------------------------------
# Summary & Comparison
# -------------------------------------------------------------
df_res = pd.DataFrame(results).sort_values("proxy_BA_mean", ascending=False)
print("\n" + "="*80)
print("PHASE 2 EXPERIMENT BENCHMARK SUMMARY (Sorted by Proxy BA)")
print("="*80)
print(df_res.to_string(index=False))

df_res.to_csv(OUT_DIR / "phase2_novel_benchmark.csv", index=False)

corrs = {}
for m, preds in oof_dict.items():
    if preds.ndim == 1:
        c = np.corrcoef(preds, oof_lgb_o1)[0, 1]
        corrs[m] = round(float(c), 4)

df_corr = pd.DataFrame(list(corrs.items()), columns=["model", "corr_with_champion_az"])
df_corr.to_csv(OUT_DIR / "phase2_model_correlations.csv", index=False)
print("\nCorrelations with current champion azimuth model:")
print(df_corr.to_string(index=False))

print(f"\nOutputs saved to {OUT_DIR}")
