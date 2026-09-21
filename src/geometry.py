"""Lightweight solar-aligned geometry features (shape-from-shade proxies).

Features per image:
  - gradient magnitude energy (robust percentiles)
  - energy parallel / perpendicular to the solar-illumination direction
  - signed directional asymmetry (gradient projection sign imbalance)
  - per-scale directional energy profile (gradient angle rotated by solar az)
  - shadow-side asymmetry proxy (lower half vs upper half of illuminated gradient)
All vectorised; expected OOF signal: does geometry add anything beyond solar azimuth?
"""
import numpy as np

from .azimuth import az_harm, bin_index


def _grads(g):
    dy, dx = np.gradient(g.astype(np.float32))
    return dx, dy, np.sqrt(dx * dx + dy * dy)


def compute_geometry_features(g, solar_az):
    """solar_az: solar azimuth in degrees for the image."""
    theta = np.deg2rad(solar_az % 360.0)
    ux, uy = np.cos(theta), np.sin(theta)
    gx, gy, mag = _grads(g)
    proj = gx * ux + gy * uy            # aligned with illumination
    perp = -gx * uy + gy * ux           # perpendicular
    feats = []
    feats.append(float(np.mean(mag)))
    feats.append(float(np.median(mag)))
    feats.append(float(np.percentile(mag, 90)))
    feats.append(float(np.mean(proj * proj)))            # energy parallel
    feats.append(float(np.mean(perp * perp)))            # energy perpendicular
    feats.append(float(np.percentile(mag, 90) - np.percentile(mag, 10)))
    feats.append(float(np.mean(np.sign(proj))))          # signed asymmetry
    feats.append(float(np.mean((proj > 0).astype(float)) - 0.5))
    feats.append(float(np.mean(np.abs(proj)) / max(np.mean(mag), 1e-6)))
    # directional energy profile rotated into solar frame
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    ob = 8
    o = np.clip((ang / 180.0 * ob).astype(int), 0, ob - 1)
    e = np.zeros(ob)
    for k in range(ob):
        e[k] = mag[o == k].sum()
    e = e / max(e.sum(), 1e-6)
    feats.extend(e.tolist())
    # multi-scale (downsampled) contrast proxies
    for m in (32, 16):
        g_s = _down(g, m)
        mag_s = _grads(g_s)[2]
        feats.append(float(mag_s.mean()))
        feats.append(float(np.percentile(mag_s, 90) - np.percentile(mag_s, 10)))
    return np.array(feats, dtype=np.float32)


def _down(g, m):
    H = g.shape[0]
    seg = np.linspace(0, H, m + 1).astype(int)
    out = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            out[i, j] = g[seg[i]:seg[i + 1], seg[j]:seg[j + 1]].mean()
    return out


def build_geometry_matrix(img_dir, ids, az_by_id, cache_dir=None):
    from .utils import cache_load, cache_save
    from .data import open_image
    key = {"mod": "geometry", "n": len(ids)}
    if cache_dir is not None:
        cached = cache_load(cache_dir, key)
        if cached is not None and len(cached) == len(ids):
            return cached
    rows = []
    for i in ids:
        img = open_image((img_dir / i) if not isinstance(img_dir, str) else img_dir + "/" + i)
        rows.append(compute_geometry_features(img, float(az_by_id[i])))
    M = np.vstack(rows).astype(np.float32)
    if cache_dir is not None:
        cache_save(cache_dir, key, M)
    return M