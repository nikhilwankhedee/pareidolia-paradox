"""Novel-image retrieval: representations, OOF nearest-neighbour, protocols."""
import numpy as np
import pandas as pd

from .azimuth import bin_index, circdiff
from .metrics import balanced_accuracy_score as _bas

try:
    import cv2  # noqa
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False


# ------------------------------------------------------------- representations
def resize(g, m):
    if _HAVE_CV2:
        return cv2.resize(g, (m, m), interpolation=cv2.INTER_AREA)
    g = np.asarray(g, dtype=np.float32)
    H = W = g.shape[0]
    seg = np.linspace(0, H, m + 1).astype(int)
    out = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            out[i, j] = g[seg[i]:seg[i + 1], seg[j]:seg[j + 1]].mean()
    return out


def _grad(g):
    dy, dx = np.gradient(g)
    return np.sqrt(dx * dx + dy * dy)


def _lap(g):
    gy, gx = np.gradient(g)
    gyy, gxy = np.gradient(gy)
    gyx, gxx = np.gradient(gx)
    return np.abs(gyy + gxx)


def _hog(g, blocks=(4, 4), obins=8):
    dy, dx = np.gradient(g)
    mag = np.sqrt(dx * dx + dy * dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    H, W = g.shape
    bh, bw = H // blocks[0], W // blocks[1]
    bins = np.zeros((blocks[0], blocks[1], obins))
    o = np.clip((ang / 180.0 * obins).astype(int), 0, obins - 1)
    for bi in range(blocks[0]):
        for bj in range(blocks[1]):
            ys, xs = slice(bi * bh, (bi + 1) * bh), slice(bj * bw, (bj + 1) * bw)
            m = mag[ys, xs]
            for ob in range(obins):
                bins[bi, bj, ob] = m[o[ys, xs] == ob].sum()
    return bins.reshape(-1)


def _z(g):
    return (g - g.mean()) / (g.std() + 1e-6)


def rep_raw(g, m=48):
    return _z(resize(g, m)).reshape(-1)


def rep_down(g, m=24):
    return _z(resize(g, m)).reshape(-1)


def rep_grad(g, m=48):
    return resize(_grad(g), m).reshape(-1)


def rep_lap(g, m=48):
    return resize(_lap(g), m).reshape(-1)


def rep_multiscale(g):
    vecs = [resize(g, 16), resize(_z(resize(g, 32)), 16),
            resize(_grad(resize(g, 32)), 16), resize(_lap(resize(g, 32)), 16)]
    return np.concatenate([v.reshape(-1) for v in vecs])


def rep_hog(g):
    return _hog(_z(resize(g, 64))).reshape(-1)


REPS = {
    "raw48": rep_raw, "down24": rep_down, "grad48": rep_grad, "lap48": rep_lap,
    "multiscale": rep_multiscale, "hog": rep_hog,
}


def build_feature_matrix(img_dir, ids, rep_name, cache_dir=None):
    """Compute feature matrix for a list of ids under a representation (cached)."""
    from .utils import cache_load, cache_save
    from .data import open_image
    key = {"mod": "features", "rep": rep_name, "n": len(ids),
           "first": id(ids[0]) if ids else None}
    if cache_dir is not None:
        cached = cache_load(cache_dir, {**key, "ids": ids[:5]})
        if cached is not None and len(cached) == len(ids):
            return cached
    fn = REPS[rep_name]
    feats = []
    for i in ids:
        img = open_image((img_dir / i) if not isinstance(img_dir, str) else img_dir + "/" + i)
        feats.append(fn(img))
    mat = np.vstack(feats).astype(np.float32)
    if cache_dir is not None:
        cache_save(cache_dir, {**key, "ids": ids[:5]}, mat)
    return mat


def l2norm(mat):
    mat = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(n, 1e-6)


# ------------------------------------------------------------------ NN search
def nn_search(ftr, fqr, k=10, batch=256):
    """Cosine nearest-neighbours: index Ft (n x d), query Fq (m x d), both
    L2-normalised. Exclusion of self/twin handled by caller (query not in index).
    Returns (idx, sims): (m x k) int, (m x k) float."""
    ftr = l2norm(ftr)
    fqr = l2norm(fqr)
    n = ftr.shape[0]
    kk = min(k, n)
    idx = np.zeros((fqr.shape[0], kk), dtype=np.int64)
    sims = np.zeros((fqr.shape[0], kk), dtype=np.float32)
    for s in range(0, fqr.shape[0], batch):
        e = min(s + batch, fqr.shape[0])
        sim = fqr[s:e] @ ftr.T
        if kk < n:
            part = np.argpartition(sim, -kk, axis=1)[:, -kk:]
            for r in range(part.shape[0]):
                p = part[r]
                order = np.argsort(sim[r, p])[::-1]
                idx[s + r, :] = p[order]
                sims[s + r, :] = sim[r, p[order]]
        else:
            idx[s:e] = np.arange(n)[None, :]
            sims[s:e, :] = sim
    return idx, sims


# ------------------------------------------------------------ retrieval records
def oof_retrieval(feats, df, fold_by_id, reps, k=10):
    """For every train image, retrieve top-k from the other folds (twin excluded).
    Returns long-form records DataFrame (one row per query image per rep)."""
    order = list(df["image_id"])
    pos = {i: t for t, i in enumerate(order)}
    rows = []
    for rep in reps:
        F = l2norm(feats[rep])
        for kf in sorted(set(fold_by_id.values())):
            q_idx = [t for t, i in enumerate(order) if int(fold_by_id[i]) == kf]
            tr_idx = [t for t, i in enumerate(order) if int(fold_by_id[i]) != kf]
            if not q_idx or not tr_idx:
                continue
            qm = np.arange(len(order))[np.array(q_idx)]
            idx, sims = nn_search(F[tr_idx], F[q_idx], k=k, batch=512)
            for r, qi in enumerate(qm):
                qid = order[qi]
                nb_ids = [order[tr_idx[j]] for j in idx[r]]
                nb_sims = [float(s) for s in sims[r]]
                rows.append({"rep": rep, "image_id": qid,
                             "nb_id_1": nb_ids[0], "sim_1": nb_sims[0],
                             "nb_id_3": nb_ids[:3], "sim_3": nb_sims[:3],
                             "nb_id_5": nb_ids[:5], "sim_5": nb_sims[:5],
                             "nb_id_10": nb_ids, "sim_10": nb_sims})
    return pd.DataFrame(rows)


def enrich_records(records, df, az_by_id, fold_by_id):
    """Attach query labels/azimuth + neighbour labels/azimuth to retrieval records."""
    lab = dict(zip(df["image_id"], df["label"]))
    rec = records.copy()
    rec["y"] = rec["image_id"].map(lab).astype(int)
    rec["az"] = rec["image_id"].map(az_by_id).astype(float)
    for n in (1, 3, 5, 10):
        col = rec[f"nb_id_{n}"].map(
            lambda v: [lab.get(x, -1) for x in v] if isinstance(v, list)
            else lab.get(v, -1))
        rec[f"nb_y_{n}"] = col
    return rec


def tp1_binary(vals):
    """Map top-1 predictions (scalar or 1-element list) to binary 0/1."""
    out = []
    for v in vals:
        v = v[0] if isinstance(v, (list, tuple)) else v
        out.append(1 if v == 1 else 0)
    return out


def knn_summary(records, k=1, thresh=None):
    """BA of the top-k majority/weighted prediction over a record subset."""
    rows = []
    for typ in ("top1", "top3", "top5", "top10"):
        if typ == "top1":
            yp = tp1_binary(records["nb_y_1"])
        else:
            kk = int(typ[3:])
            yp = []
            for v in records[f"nb_y_{kk}"]:
                vv = [x for x in v if x >= 0]
                yp.append(1 if (sum(vv) / len(vv)) >= 0.5 else 0)
        rows.append({"variant": typ,
                     "BA": float(_bas(records["y"].values, np.array(yp))),
                     "n": len(records)})
    return pd.DataFrame(rows)


def protocol_breakdown(records, az_by_id, tr_hash):
    """Protocol B/D/E rows: retention by top1-similarity buckets and by
    |delta az| of the top-1 neighbour."""
    from .utils import save_csv  # noqa
    lab = records["y"]
    p1 = tp1_binary(records["nb_y_1"])
    sims = records["sim_1"].values
    out = []
    for lo, hi in ((0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.0, 0.85)):
        m = (sims >= lo) & (sims < hi)
        if m.sum() == 0:
            continue
        out.append({"protocol": f"D_sim_{lo:.2f}_{hi:.2f}", "BA": float(_bas(lab[m], np.array(p1)[m])),
                    "n": int(m.sum()), "mean_sim": float(sims[m].mean())})
    top1_id = records[f"nb_id_1"]
    daz = []
    for i, nid in zip(records["image_id"], top1_id):
        daz.append(abs(circdiff([az_by_id[i]], [az_by_id[nid]])[0]))
    daz = np.array(daz)
    p1 = np.array(p1)
    for lo, hi in ((0, 45), (45, 90), (90, 135), (135, 181)):
        m = (daz >= lo) & (daz < hi)
        if m.sum() == 0:
            continue
        out.append({"protocol": f"E_daz_{lo}_{hi}", "BA": float(_bas(lab[m], p1[m])),
                    "n": int(m.sum()), "mean_sim": float(sims[m].mean())})
    return pd.DataFrame(out)


def distribution_matched_ba(records, az_by_id, tcounts):
    """Per-bin BA reweighted to the TEST azimuth distribution (Protocol C)."""
    az = records["image_id"].map(az_by_id).astype(float).values
    bins = bin_index(az)
    w = np.asarray(tcounts, dtype=float)
    w = w / w.sum()
    p1 = tp1_binary(records["nb_y_1"])
    lab = records["y"].values
    ba_k = np.zeros(len(w))
    ok = np.zeros(len(w), dtype=bool)
    for k in range(len(w)):
        m = bins == k
        if m.sum() == 0:
            continue
        ba_k[k] = _bas(lab[m], np.array(p1)[m])
        ok[k] = True
    if ok.sum() < 2:
        return float("nan"), float("nan")
    return float((ba_k[ok] * w[ok]).sum()), float(_bas(lab, np.array(p1)))