"""Metrics: balanced accuracy, confusion, recalls, ROC-AUC, threshold tuning."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


def report_metrics(clf_name, y_true, p1, thresh=0.5):
    yt = np.asarray(y_true, dtype=int)
    pp = np.asarray(p1, dtype=float)
    yp = (pp >= thresh).astype(int)
    ba = balanced_accuracy_score(yt, yp)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    r0 = cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1)
    r1 = cm[1, 1] / max(cm[1, 0] + cm[1, 1], 1)
    try:
        auc = roc_auc_score(yt, pp)
    except ValueError:
        auc = float("nan")
    return {
        "model": clf_name,
        "threshold": float(thresh),
        "BA": float(ba),
        "accuracy": float(np.mean(yp == yt)),
        "recall_0": float(r0),
        "recall_1": float(r1),
        "roc_auc": float(auc),
        "n": int(len(yt)),
    }


def scan_thresholds(y_true, p1, grid=(0.40, 0.45, 0.50, 0.55, 0.60, 0.65)):
    rows = []
    for t in grid:
        rows.append(report_metrics("scan", y_true, p1, thresh=t))
    df = pd.DataFrame(rows)
    return df, df.sort_values("BA", ascending=False).iloc[0]


def best_threshold(y_true, p1, grid=(0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)):
    best = None
    bt = 0.5
    for t in grid:
        r = report_metrics("scan", y_true, p1, thresh=t)
        if best is None or r["BA"] > best["BA"]:
            best = r["BA"]
            bt = t
    return float(bt), float(best)


def ba_from_probs(y_true, p1):
    yt = np.asarray(y_true, dtype=int)
    pp = np.asarray(p1, dtype=float)
    return balanced_accuracy_score(yt, (pp >= 0.5).astype(int))


def add_metrics_row(rows, name, y_true, p1, thresh=None, note=""):
    r = report_metrics(name, y_true, p1, thresh=0.5 if thresh is None else thresh)
    r["note"] = note
    rows.append(r)
    return r