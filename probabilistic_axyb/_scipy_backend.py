"""Sparse trust-region least-squares backend, imported only when requested."""

from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
except ImportError as error:  # pragma: no cover - exercised without the extra installed
    raise ImportError(
        "The default 'scipy' backend requires SciPy. Reinstall the package with "
        "`pip install probabilistic-axyb`, or explicitly use backend='numpy'."
    ) from error

from .lie import inv_se3, log_so3, right_update


def _square_roots(precisions):
    # np.linalg.cholesky returns L for P=L@L.T; L.T@v gives v.T@P@v.
    return tuple(np.swapaxes(np.linalg.cholesky(item), 1, 2) for item in precisions)


def _unpack(parameters, x_base, y_base, c_base=None):
    x = right_update(x_base, parameters[:3], parameters[3:6])
    y = right_update(y_base, parameters[6:9], parameters[9:12])
    if c_base is None:
        return x, y, None
    c = np.empty_like(c_base)
    for i in range(len(c_base)):
        offset = 12 + 6 * i
        c[i] = right_update(c_base[i], parameters[offset : offset + 3], parameters[offset + 3 : offset + 6])
    return x, y, c


def solve(a, b, x0, y0, c0, precisions, noise_configuration, max_evaluations, tolerance):
    roots = _square_roots(precisions)
    count = len(a)

    def residual(parameters):
        x, y, c = _unpack(parameters, x0, y0, c0)
        values = np.empty(12 * count)
        inv_x = inv_se3(x)
        for i in range(count):
            inv_c = inv_se3(c[i])
            if noise_configuration == 1:
                n_noise = c[i] @ inv_x @ inv_se3(a[i])
            else:
                n_noise = x @ inv_c @ a[i]
            m_noise = inv_c @ y @ b[i]
            offset = 12 * i
            values[offset : offset + 3] = roots[0][i] @ log_so3(n_noise[:3, :3])
            values[offset + 3 : offset + 6] = roots[1][i] @ n_noise[:3, 3]
            values[offset + 6 : offset + 9] = roots[2][i] @ log_so3(m_noise[:3, :3])
            values[offset + 9 : offset + 12] = roots[3][i] @ m_noise[:3, 3]
        return values

    # Each measurement depends on X, Y, and only its corresponding latent C.
    sparsity = lil_matrix((12 * count, 12 + 6 * count), dtype=int)
    for i in range(count):
        rows = slice(12 * i, 12 * i + 12)
        sparsity[rows, :12] = 1
        sparsity[rows, 12 + 6 * i : 18 + 6 * i] = 1
    result = least_squares(
        residual,
        np.zeros(12 + 6 * count),
        jac_sparsity=sparsity.tocsr(),
        method="trf",
        x_scale="jac",
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_evaluations,
    )
    x, y, c = _unpack(result.x, x0, y0, c0)
    return x, y, c, -0.5 * float(result.fun @ result.fun), result.nfev, result.success


def solve_noiseless_a(a, b, x0, y0, wm_precision, pm_precision, max_evaluations, tolerance):
    wm_root, pm_root = _square_roots((wm_precision, pm_precision))
    count = len(a)

    def residual(parameters):
        x, y, _ = _unpack(parameters, x0, y0)
        inv_x = inv_se3(x)
        values = np.empty(6 * count)
        for i in range(count):
            noise = inv_x @ inv_se3(a[i]) @ y @ b[i]
            offset = 6 * i
            values[offset : offset + 3] = wm_root[i] @ log_so3(noise[:3, :3])
            values[offset + 3 : offset + 6] = pm_root[i] @ noise[:3, 3]
        return values

    result = least_squares(
        residual,
        np.zeros(12),
        method="trf",
        x_scale="jac",
        ftol=tolerance,
        xtol=tolerance,
        gtol=tolerance,
        max_nfev=max_evaluations,
    )
    x, y, _ = _unpack(result.x, x0, y0)
    return x, y, -0.5 * float(result.fun @ result.fun), result.nfev, result.success
