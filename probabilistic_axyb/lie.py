from __future__ import annotations

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(rotation_vector: np.ndarray) -> np.ndarray:
    """SO(3) exponential map (MATLAB ``LargeSO3``)."""
    vector = np.asarray(rotation_vector, dtype=float).reshape(3)
    theta = np.linalg.norm(vector)
    if theta < 1e-14:
        return np.eye(3) + skew(vector)
    axis_hat = skew(vector / theta)
    return np.eye(3) + np.sin(theta) * axis_hat + (1.0 - np.cos(theta)) * (axis_hat @ axis_hat)


def log_so3(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map, robust at zero and pi."""
    matrix = np.asarray(rotation, dtype=float)[:3, :3]
    cosine = np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cosine))
    if theta < 1e-8:
        return 0.5 * np.array(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
        )
    if np.pi - theta < 1e-6:
        symmetric = (matrix + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        index = int(np.argmax(axis))
        if axis[index] > 1e-8:
            for j in range(3):
                if j != index:
                    axis[j] = symmetric[index, j] / axis[index]
        axis /= np.linalg.norm(axis)
        return theta * axis
    scale = theta / (2.0 * np.sin(theta))
    return scale * np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]]
    )


def inv_se3(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def right_update(transform: np.ndarray, rotation_step: np.ndarray, translation_step: np.ndarray) -> np.ndarray:
    increment = np.eye(4)
    increment[:3, :3] = exp_so3(rotation_step)
    increment[:3, 3] = np.asarray(translation_step).reshape(3)
    return transform @ increment


def interpolate_se3(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = first[:3, :3] @ exp_so3(fraction * log_so3(first[:3, :3].T @ second[:3, :3]))
    result[:3, 3] = (1.0 - fraction) * first[:3, 3] + fraction * second[:3, 3]
    return result


def project_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=float))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def inverse_right_jacobian_so3(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    theta = np.linalg.norm(vector)
    hat = skew(vector)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * hat + (hat @ hat) / 12.0
    half = theta / 2.0
    coefficient = (1.0 - np.cos(half) / (np.sin(half) / half)) / (theta * theta)
    return np.eye(3) - 0.5 * hat + coefficient * (hat @ hat)

