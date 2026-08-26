from __future__ import annotations

import numpy as np


def transforms(value: np.ndarray, name: str) -> np.ndarray:
    """Return transforms in Python-native ``(n, 4, 4)`` layout."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape == (4, 4):
        array = array[None, ...]
    elif array.ndim == 3 and array.shape[1:] == (4, 4):
        pass
    elif array.ndim == 3 and array.shape[:2] == (4, 4):
        array = np.moveaxis(array, 2, 0)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (n, 4, 4) or (4, 4, n)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def covariances(value: np.ndarray, n: int, name: str) -> np.ndarray:
    """Return 3-by-3 covariance/precision matrices in ``(n, 3, 3)`` layout."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape == (3, 3):
        array = np.repeat(array[None, ...], n, axis=0)
    elif array.ndim == 3 and array.shape[1:] == (3, 3):
        pass
    elif array.ndim == 3 and array.shape[:2] == (3, 3):
        array = np.moveaxis(array, 2, 0)
    if array.shape != (n, 3, 3):
        raise ValueError(f"{name} must have shape (n, 3, 3), (3, 3, n), or (3, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()


def transform(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array.copy()
