from __future__ import annotations

import numpy as np

from ._validation import covariances, transform, transforms
from .lie import skew


def _weighted_mapping(q: np.ndarray, covariance: np.ndarray, precision: np.ndarray):
    information = q.T @ precision @ q
    mapping = -np.linalg.pinv(information) @ q.T @ precision
    estimate_covariance = mapping @ covariance @ mapping.T
    return estimate_covariance, mapping


def compute_uncertainty(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    inv_sigma_wn: np.ndarray,
    inv_sigma_pn: np.ndarray,
    inv_sigma_wm: np.ndarray,
    inv_sigma_pm: np.ndarray,
    noise_configuration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration covariance for noise configuration 1 or 2 (paper Section VII)."""
    if noise_configuration not in (1, 2):
        raise ValueError("noise_configuration must be 1 or 2")
    x_array, y_array = transform(x, "x"), transform(y, "y")
    c_array = transforms(c, "c")
    count = len(c_array)
    precisions = tuple(
        covariances(value, count, name)
        for value, name in (
            (inv_sigma_wn, "inv_sigma_wn"),
            (inv_sigma_pn, "inv_sigma_pn"),
            (inv_sigma_wm, "inv_sigma_wm"),
            (inv_sigma_pm, "inv_sigma_pm"),
        )
    )
    qnx, qny = np.zeros((6 * count, 6)), np.zeros((6 * count, 6))
    qnc = np.zeros((6 * count, 6 * count))
    qmx, qmy = np.zeros_like(qnx), np.zeros_like(qny)
    qmc = np.zeros_like(qnc)
    rx, ry = x_array[:3, :3], y_array[:3, :3]
    px, py = x_array[:3, 3], y_array[:3, 3]
    for i, ci in enumerate(c_array):
        rc, pc = ci[:3, :3], ci[:3, 3]
        rotation = slice(6 * i, 6 * i + 3)
        position = slice(6 * i + 3, 6 * i + 6)
        if noise_configuration == 1:
            qnx[rotation, :3] = -rc
            qnx[position, :3], qnx[position, 3:] = -skew(pc) @ rc, -rc
            qnc[rotation, rotation] = rc
            qnc[position, rotation], qnc[position, position] = skew(pc) @ rc, rc
        else:
            qnx[rotation, :3] = rx
            qnx[position, :3], qnx[position, 3:] = skew(px) @ rx, rx
            qnc[rotation, rotation] = -rx
            qnc[position, rotation], qnc[position, position] = -skew(px) @ rx, -rx
        rotated_y = rc.T @ ry
        qmy[rotation, :3] = rotated_y
        qmy[position, :3] = skew(rc.T @ (py - pc)) @ rotated_y
        qmy[position, 3:] = rotated_y
        qmc[rotation, rotation] = -np.eye(3)
        qmc[position, position] = -np.eye(3)
    q = np.block([[qnx, qny, qnc], [qmx, qmy, qmc]])
    covariance = np.zeros((12 * count, 12 * count))
    precision = np.zeros_like(covariance)
    for i in range(count):
        blocks = (6 * i, 6 * i + 3, 6 * count + 6 * i, 6 * count + 6 * i + 3)
        for start, inverse in zip(blocks, precisions):
            precision[start : start + 3, start : start + 3] = inverse[i]
            covariance[start : start + 3, start : start + 3] = np.linalg.inv(inverse[i])
    estimated, mapping = _weighted_mapping(q, covariance, precision)
    return estimated[:6, :6], estimated[6:12, 6:12], mapping


def compute_uncertainty_noiseless_a(
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    inv_sigma_wm: np.ndarray,
    inv_sigma_pm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibration covariance for noise configuration 3."""
    transform(x, "x")
    transform(y, "y")
    b_array = transforms(b, "b")
    count = len(b_array)
    wm_precision = covariances(inv_sigma_wm, count, "inv_sigma_wm")
    pm_precision = covariances(inv_sigma_pm, count, "inv_sigma_pm")
    qmx, qmy = np.zeros((6 * count, 6)), np.zeros((6 * count, 6))
    covariance, precision = np.zeros((6 * count, 6 * count)), np.zeros((6 * count, 6 * count))
    for i, bi in enumerate(b_array):
        rb, pb = bi[:3, :3], bi[:3, 3]
        rotation = slice(6 * i, 6 * i + 3)
        position = slice(6 * i + 3, 6 * i + 6)
        qmx[rotation, :3] = -np.eye(3)
        qmx[position, 3:] = -np.eye(3)
        qmy[rotation, :3] = rb.T
        qmy[position, :3], qmy[position, 3:] = -rb.T @ skew(pb), rb.T
        precision[rotation, rotation], precision[position, position] = wm_precision[i], pm_precision[i]
        covariance[rotation, rotation] = np.linalg.inv(wm_precision[i])
        covariance[position, position] = np.linalg.inv(pm_precision[i])
    estimated, mapping = _weighted_mapping(np.hstack((qmx, qmy)), covariance, precision)
    return estimated[:6, :6], estimated[6:12, 6:12], mapping

