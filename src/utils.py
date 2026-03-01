"""
utils.py — Shared helpers for feature extraction.
"""

import numpy as np
from scipy.fftpack import dct


def dct2(array: np.ndarray) -> np.ndarray:
    """Apply 2D orthonormal DCT."""
    return dct(dct(array.T, norm='ortho').T, norm='ortho')


def chi_squared(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compute chi-squared divergence of hist1 from hist2.
    Bins where hist2 == 0 are skipped (undefined).
    """
    mask = hist2 > 0
    diff = hist1[mask].astype(float) - hist2[mask].astype(float)
    return float(np.sum(diff ** 2 / hist2[mask]))
