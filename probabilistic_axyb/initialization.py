from __future__ import annotations

import numpy as np

from ._validation import transforms
from .lie import inv_se3, log_so3, project_so3

# Relative rotations must span at least two dimensions for X and Y to be
# separable. Measured axis sets are either well above 0.3 or below 1e-4, so the
# threshold sits in an empty region rather than on a judgement call.
_EXCITATION_TOLERANCE = 1e-3


def rotation_excitation(a: np.ndarray) -> float:
    """Ratio of the second to the first singular value of the relative-rotation axes.

    ``AX=YB`` determines ``X`` and ``Y`` only when the measured motions rotate
    about at least two non-parallel axes. This returns ``0.0`` for a stationary
    or single-axis sequence and grows towards ``1`` as the axes spread out.
    """
    a_array = transforms(a, "a")
    reference = inv_se3(a_array[0])
    axes = []
    for ai in a_array[1:]:
        vector = log_so3((reference @ ai)[:3, :3])
        angle = np.linalg.norm(vector)
        if angle > _EXCITATION_TOLERANCE:
            axes.append(vector / angle)
    if len(axes) < 2:
        return 0.0
    values = np.linalg.svd(np.array(axes), compute_uv=False)
    return float(values[1] / values[0])


def solve_axyb(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form Shah initializer for transformation pairs satisfying ``AX=YB``.

    Inputs may use either ``(n, 4, 4)`` (recommended) or MATLAB's
    ``(4, 4, n)`` layout.
    """
    a_array = transforms(a, "a")
    b_array = transforms(b, "b")
    if len(a_array) != len(b_array):
        raise ValueError("a and b must contain the same number of transforms")
    if len(a_array) < 3:
        raise ValueError("at least three transform pairs are required")
    if rotation_excitation(a_array) < _EXCITATION_TOLERANCE:
        raise ValueError(
            "transform motions are degenerate; rotations share a single axis, so "
            "the rotational parts of X and Y cannot be identified separately"
        )

    correlation = sum(np.kron(bi[:3, :3], ai[:3, :3]) for ai, bi in zip(a_array, b_array))
    u, _, vt = np.linalg.svd(correlation)
    vx = vt[0].reshape((3, 3), order="F")
    vy = u[:, 0].reshape((3, 3), order="F")
    # The sign of a singular vector is arbitrary. When the reshaped block is a
    # reflection, negating the pair recovers the rotation; projecting it as-is
    # would return the nearest rotation to a reflection instead.
    if np.linalg.det(vx) < 0.0:
        vx, vy = -vx, -vy
    rx = project_so3(vx)
    ry = project_so3(vy)

    system = np.empty((3 * len(a_array), 6))
    target = np.empty(3 * len(a_array))
    for i, (ai, bi) in enumerate(zip(a_array, b_array)):
        rows = slice(3 * i, 3 * i + 3)
        system[rows] = np.hstack((-ai[:3, :3], np.eye(3)))
        target[rows] = ai[:3, 3] - ry @ bi[:3, 3]
    translation, _, rank, _ = np.linalg.lstsq(system, target, rcond=None)
    if rank < 6:
        raise ValueError("transform motions are degenerate; translation cannot be identified")

    x = np.eye(4)
    y = np.eye(4)
    x[:3, :3], x[:3, 3] = rx, translation[:3]
    y[:3, :3], y[:3, 3] = ry, translation[3:]
    return x, y

