"""Solar-azimuth helpers: circular arithmetic, binning, harmonic encoding."""
import numpy as np


def circdiff(a, b):
    """Circular difference (b - a) wrapped into (-180, 180]. Degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (b - a) % 360.0
    d = np.where(d > 180.0, d - 360.0, d)
    return d


def az_harm(az, order=1):
    az = np.asarray(az, dtype=float)
    rad = np.deg2rad(az)
    cols = []
    for k in range(1, order + 1):
        cols.append(np.sin(k * rad))
        cols.append(np.cos(k * rad))
    return np.column_stack(cols) if cols else np.zeros((len(az), 0))


AZ_BINS = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360.0])


def bin_index(az, bins=AZ_BINS):
    az = np.asarray(az, dtype=float)
    return np.clip(np.searchsorted(bins, az, side="right") - 1, 0, len(bins) - 2)


def bin_counts(az, bins=AZ_BINS):
    return np.bincount(bin_index(az, bins), minlength=len(bins) - 1).astype(float)


AZ_BIN_LABELS = ["0-45", "45-90", "90-135", "135-180", "180-225", "225-270", "270-315", "315-360"]