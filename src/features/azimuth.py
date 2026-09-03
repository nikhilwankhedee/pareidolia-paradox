"""Azimuth feature helpers.

The sun azimuth angle is a circular quantity. For linear models we use the
bounded, phase-preserving features sin(azimuth) and cos(azimuth).
"""
from __future__ import annotations

import numpy as np


def sin_cos_azimuth(azimuth_deg):
    """Convert an azimuth angle (degrees) to (sin, cos) features.

    Parameters
    ----------
    azimuth_deg : array-like
        Sun azimuth angle in degrees.

    Returns
    -------
    np.ndarray of shape (n, 2), columns [sin, cos].
    """
    a = np.asarray(azimuth_deg, dtype=np.float64)
    theta = np.deg2rad(a)
    return np.stack([np.sin(theta), np.cos(theta)], axis=1)
