from __future__ import annotations

import numpy as np

from ._validation import transforms
from .lie import project_so3


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

    correlation = sum(np.kron(bi[:3, :3], ai[:3, :3]) for ai, bi in zip(a_array, b_array))
    u, _, vt = np.linalg.svd(correlation)
    rx = project_so3(vt[0].reshape((3, 3), order="F"))
    ry = project_so3(u[:, 0].reshape((3, 3), order="F"))

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

