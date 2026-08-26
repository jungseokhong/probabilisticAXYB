"""Numba kernels for the analytic-gradient solver.

This module is imported lazily: Numba remains an optional dependency.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError as error:  # pragma: no cover - exercised without the extra installed
    raise ImportError(
        "The 'numba' backend requires Numba. Install it with "
        "`pip install probabilistic-axyb[numba]`."
    ) from error


@njit(cache=True)
def _skew(vector):
    x, y, z = vector
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


@njit(cache=True)
def _inv_se3(transform):
    result = np.eye(4)
    rotation = transform[:3, :3].T.copy()
    translation = transform[:3, 3].copy()
    result[:3, :3] = rotation
    result[:3, 3] = -rotation @ translation
    return result


@njit(cache=True)
def _log_so3(rotation):
    cosine = (rotation[0, 0] + rotation[1, 1] + rotation[2, 2] - 1.0) / 2.0
    cosine = min(1.0, max(-1.0, cosine))
    theta = np.arccos(cosine)
    vee = np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-6:
        symmetric = (rotation + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        index = np.argmax(axis)
        if axis[index] > 1e-8:
            for j in range(3):
                if j != index:
                    axis[j] = symmetric[index, j] / axis[index]
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * vee


@njit(cache=True)
def _inverse_right_jacobian(vector):
    theta = np.linalg.norm(vector)
    hat = _skew(vector)
    if theta < 1e-7:
        return np.eye(3) - 0.5 * hat + (hat @ hat) / 12.0
    half = theta / 2.0
    coefficient = (1.0 - np.cos(half) / (np.sin(half) / half)) / (theta * theta)
    return np.eye(3) - 0.5 * hat + coefficient * (hat @ hat)


@njit(cache=True)
def _noise_terms(x, y, a, b, c, noise_configuration):
    inv_c = _inv_se3(c)
    if noise_configuration == 1:
        n_noise = c @ _inv_se3(x) @ _inv_se3(a)
    else:
        n_noise = x @ inv_c @ a
    return n_noise, inv_c @ y @ b


@njit(cache=True)
def cost(x, y, a, b, c, wn_precision, pn_precision, wm_precision, pm_precision, noise_configuration):
    total = 0.0
    for i in range(a.shape[0]):
        n_noise, m_noise = _noise_terms(x, y, a[i], b[i], c[i], noise_configuration)
        wn, pn = _log_so3(n_noise[:3, :3]), n_noise[:3, 3].copy()
        wm, pm = _log_so3(m_noise[:3, :3]), m_noise[:3, 3].copy()
        total -= 0.5 * (
            wn @ wn_precision[i] @ wn
            + pn @ pn_precision[i] @ pn
            + wm @ wm_precision[i] @ wm
            + pm @ pm_precision[i] @ pm
        )
    return total


@njit(cache=True)
def gradient(x, y, a, b, c, wn_precision, pn_precision, wm_precision, pm_precision, noise_configuration):
    count = a.shape[0]
    rx, ry, py = x[:3, :3].copy(), y[:3, :3].copy(), y[:3, 3].copy()
    gwx, gwy, gqx, gqy = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    gwc, gqc = np.zeros((count, 3)), np.zeros((count, 3))
    total = 0.0
    for i in range(count):
        rc = c[i, :3, :3].copy()
        pa, pb, pc = a[i, :3, 3].copy(), b[i, :3, 3].copy(), c[i, :3, 3].copy()
        n_noise, m_noise = _noise_terms(x, y, a[i], b[i], c[i], noise_configuration)
        wn, pn = _log_so3(n_noise[:3, :3]), n_noise[:3, 3].copy()
        wm, pm = _log_so3(m_noise[:3, :3]), m_noise[:3, 3].copy()
        inv_d1, inv_d2 = _inverse_right_jacobian(wn), _inverse_right_jacobian(wm)
        if noise_configuration == 1:
            first = -wn @ wn_precision[i] @ inv_d1 @ rc - pn @ pn_precision[i] @ _skew(pc - pn) @ rc
            second = wm @ wm_precision[i] @ inv_d2 - pm @ pm_precision[i] @ _skew(rc.T @ (ry @ pb))
            gwx -= first
            gwy -= second @ rc.T @ ry
            gwc[i] = first + second - pm @ pm_precision[i] @ _skew(rc.T @ (py - pc))
            gqx += pn @ pn_precision[i] @ rc
            gqy -= pm @ pm_precision[i] @ rc.T @ ry
            gqc[i] = -pn @ pn_precision[i] @ rc + pm @ pm_precision[i]
        else:
            first = wn @ wn_precision[i] @ inv_d1 @ rx - pn @ pn_precision[i] @ _skew(rx @ rc.T @ (pa - pc)) @ rx
            second = wm @ wm_precision[i] @ inv_d2 - pm @ pm_precision[i] @ _skew(rc.T @ ry @ pb)
            gwx -= first
            gwy -= second @ rc.T @ ry
            gwc[i] = first + second - pm @ pm_precision[i] @ _skew(rc.T @ (py - pc))
            gqx -= pn @ pn_precision[i] @ rx
            gqy -= pm @ pm_precision[i] @ rc.T @ ry
            gqc[i] = pn @ pn_precision[i] @ rx + pm @ pm_precision[i]
        total -= 0.5 * (
            wn @ wn_precision[i] @ wn
            + pn @ pn_precision[i] @ pn
            + wm @ wm_precision[i] @ wm
            + pm @ pm_precision[i] @ pm
        )
    return gwx, gwy, gwc, gqx, gqy, gqc, total


@njit(cache=True)
def cost_noiseless_a(x, y, a, b, wm_precision, pm_precision):
    total = 0.0
    inv_x = _inv_se3(x)
    for i in range(a.shape[0]):
        noise = inv_x @ _inv_se3(a[i]) @ y @ b[i]
        wm, pm = _log_so3(noise[:3, :3]), noise[:3, 3].copy()
        total -= 0.5 * (wm @ wm_precision[i] @ wm + pm @ pm_precision[i] @ pm)
    return total


@njit(cache=True)
def gradient_noiseless_a(x, y, a, b, wm_precision, pm_precision):
    inv_x = _inv_se3(x)
    gwx, gwy, gqx, gqy = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    total = 0.0
    for i in range(a.shape[0]):
        noise = inv_x @ _inv_se3(a[i]) @ y @ b[i]
        rm = noise[:3, :3].copy()
        wm, pm = _log_so3(rm), noise[:3, 3].copy()
        h = wm @ wm_precision[i] @ _inverse_right_jacobian(wm)
        weighted_p = pm @ pm_precision[i]
        rb_transpose = b[i, :3, :3].T.copy()
        pb = b[i, :3, 3].copy()
        gwxy = rm @ rb_transpose
        gwx += h - weighted_p @ _skew(pm)
        gwy += -h @ gwxy + weighted_p @ gwxy @ _skew(pb)
        gqx += weighted_p
        gqy -= weighted_p @ gwxy
        total -= 0.5 * (wm @ wm_precision[i] @ wm + pm @ pm_precision[i] @ pm)
    return gwx, gwy, gqx, gqy, total
