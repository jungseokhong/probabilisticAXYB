from __future__ import annotations

import numpy as np

from .lie import exp_so3


def random_so3(rng: np.random.Generator | None = None) -> np.ndarray:
    generator = rng or np.random.default_rng()
    quaternion = generator.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def random_se3(position_scale: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    generator = rng or np.random.default_rng()
    result = np.eye(4)
    result[:3, :3] = random_so3(generator)
    result[:3, 3] = position_scale * generator.normal(size=3)
    return result


def add_noise_se3(
    transform: np.ndarray,
    rotation_std: float,
    translation_std: float,
    *,
    side: str = "right",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    generator = rng or np.random.default_rng()
    noise = np.eye(4)
    noise[:3, :3] = exp_so3(generator.normal(scale=rotation_std, size=3))
    noise[:3, 3] = generator.normal(scale=translation_std, size=3)
    if side == "right":
        return np.asarray(transform) @ noise
    if side == "left":
        return noise @ np.asarray(transform)
    raise ValueError("side must be 'left' or 'right'")


def invert_covariances(covariances: np.ndarray) -> np.ndarray:
    array = np.asarray(covariances, dtype=float)
    if array.ndim == 2:
        return np.linalg.inv(array)
    if array.ndim != 3:
        raise ValueError("covariances must contain 3-by-3 matrices")
    matlab_layout = array.shape[1:] != (3, 3) and array.shape[:2] == (3, 3)
    native = np.moveaxis(array, 2, 0) if matlab_layout else array
    inverted = np.linalg.inv(native)
    return np.moveaxis(inverted, 0, 2) if matlab_layout else inverted
