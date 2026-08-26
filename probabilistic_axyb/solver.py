from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

import numpy as np

from ._validation import covariances, transform, transforms
from .lie import (
    interpolate_se3,
    inv_se3,
    inverse_right_jacobian_so3,
    log_so3,
    project_so3,
    right_update,
    skew,
)


@dataclass(frozen=True)
class SolverResult:
    """Detailed solver output. ``c``, ``n_noise``, and ``m_noise`` are absent for configuration 3."""

    x: np.ndarray
    y: np.ndarray
    log_likelihood: float
    iterations: int
    converged: bool
    c: np.ndarray | None = None
    n_noise: np.ndarray | None = None
    m_noise: np.ndarray | None = None


def available_backends() -> tuple[str, ...]:
    """Return solver backends usable in the current Python environment."""
    result = ["numpy"]
    if find_spec("scipy") is not None:
        result.append("scipy")
    if find_spec("numba") is not None:
        result.append("numba")
    return tuple(result)


def _check_backend(backend: str) -> str:
    normalized = backend.lower()
    if normalized not in ("numpy", "scipy", "numba"):
        raise ValueError("backend must be 'numpy', 'scipy', or 'numba'")
    return normalized


def _prepare_common(a: np.ndarray, b: np.ndarray, x0: np.ndarray, y0: np.ndarray):
    a_array = transforms(a, "a")
    b_array = transforms(b, "b")
    if len(a_array) != len(b_array):
        raise ValueError("a and b must contain the same number of transforms")
    if not len(a_array):
        raise ValueError("at least one transform pair is required")
    return a_array, b_array, transform(x0, "x0"), transform(y0, "y0")


def _noise_terms(x, y, a, b, c, noise_configuration):
    inv_c = inv_se3(c)
    if noise_configuration == 1:
        n_noise = c @ inv_se3(x) @ inv_se3(a)
    elif noise_configuration == 2:
        n_noise = x @ inv_c @ a
    else:
        raise ValueError("noise_configuration must be 1 or 2")
    m_noise = inv_c @ y @ b
    return n_noise, m_noise


def compute_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    inv_sigma_wn: np.ndarray,
    inv_sigma_pn: np.ndarray,
    inv_sigma_wm: np.ndarray,
    inv_sigma_pm: np.ndarray,
    noise_configuration: int,
) -> float:
    """Evaluate equations (15)/(22), matching MATLAB ``computeLogL``."""
    a_array, b_array, x_array, y_array = _prepare_common(a, b, x, y)
    c_array = transforms(c, "c")
    count = len(a_array)
    if len(c_array) != count:
        raise ValueError("c must contain one transform per measurement pair")
    precisions = [
        covariances(value, count, name)
        for value, name in (
            (inv_sigma_wn, "inv_sigma_wn"),
            (inv_sigma_pn, "inv_sigma_pn"),
            (inv_sigma_wm, "inv_sigma_wm"),
            (inv_sigma_pm, "inv_sigma_pm"),
        )
    ]
    total = 0.0
    for i in range(count):
        nn, mn = _noise_terms(x_array, y_array, a_array[i], b_array[i], c_array[i], noise_configuration)
        wn, pn = log_so3(nn[:3, :3]), nn[:3, 3]
        wm, pm = log_so3(mn[:3, :3]), mn[:3, 3]
        total -= 0.5 * (
            wn @ precisions[0][i] @ wn
            + pn @ precisions[1][i] @ pn
            + wm @ precisions[2][i] @ wm
            + pm @ precisions[3][i] @ pm
        )
    return float(total)


def compute_log_likelihood_noiseless_a(
    x: np.ndarray,
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    inv_sigma_wm: np.ndarray,
    inv_sigma_pm: np.ndarray,
) -> float:
    """Evaluate equation (29), matching MATLAB ``computeLogL_noiselessA``."""
    a_array, b_array, x_array, y_array = _prepare_common(a, b, x, y)
    count = len(a_array)
    wm_precision = covariances(inv_sigma_wm, count, "inv_sigma_wm")
    pm_precision = covariances(inv_sigma_pm, count, "inv_sigma_pm")
    total = 0.0
    inv_x = inv_se3(x_array)
    for i in range(count):
        noise = inv_x @ inv_se3(a_array[i]) @ y_array @ b_array[i]
        rotation, position = log_so3(noise[:3, :3]), noise[:3, 3]
        total -= 0.5 * (rotation @ wm_precision[i] @ rotation + position @ pm_precision[i] @ position)
    return float(total)


def _gradient(x, y, a, b, c, precisions, noise_configuration):
    count = len(a)
    rx, ry = x[:3, :3], y[:3, :3]
    py = y[:3, 3]
    gwx, gwy = np.zeros(3), np.zeros(3)
    gqx, gqy = np.zeros(3), np.zeros(3)
    gwc, gqc = np.zeros((count, 3)), np.zeros((count, 3))
    total = 0.0
    for i in range(count):
        ra, rc = a[i, :3, :3], c[i, :3, :3]
        pa, pb, pc = a[i, :3, 3], b[i, :3, 3], c[i, :3, 3]
        nn, mn = _noise_terms(x, y, a[i], b[i], c[i], noise_configuration)
        wn, pn = log_so3(nn[:3, :3]), nn[:3, 3]
        wm, pm = log_so3(mn[:3, :3]), mn[:3, 3]
        wn_precision, pn_precision, wm_precision, pm_precision = (item[i] for item in precisions)
        inv_d1 = inverse_right_jacobian_so3(wn)
        inv_d2 = inverse_right_jacobian_so3(wm)

        if noise_configuration == 1:
            u = pc - pn
            first = -wn @ wn_precision @ inv_d1 @ rc - pn @ pn_precision @ skew(u) @ rc
            u = rc.T @ (ry @ pb)
            second = wm @ wm_precision @ inv_d2 - pm @ pm_precision @ skew(u)
            gwx -= first
            gwy -= second @ rc.T @ ry
            u = rc.T @ (py - pc)
            gwc[i] = first + second - pm @ pm_precision @ skew(u)
            gqx += pn @ pn_precision @ rc
            gqy -= pm @ pm_precision @ rc.T @ ry
            gqc[i] = -pn @ pn_precision @ rc + pm @ pm_precision
        else:
            u = rx @ rc.T @ (pa - pc)
            first = wn @ wn_precision @ inv_d1 @ rx - pn @ pn_precision @ skew(u) @ rx
            u = rc.T @ ry @ pb
            second = wm @ wm_precision @ inv_d2 - pm @ pm_precision @ skew(u)
            gwx -= first
            gwy -= second @ rc.T @ ry
            u = rc.T @ (py - pc)
            gwc[i] = first + second - pm @ pm_precision @ skew(u)
            gqx -= pn @ pn_precision @ rx
            gqy -= pm @ pm_precision @ rc.T @ ry
            gqc[i] = pn @ pn_precision @ rx + pm @ pm_precision

        total -= 0.5 * (
            wn @ wn_precision @ wn
            + pn @ pn_precision @ pn
            + wm @ wm_precision @ wm
            + pm @ pm_precision @ pm
        )
    return gwx, gwy, gwc, gqx, gqy, gqc, float(total)


def _apply_update(x, y, c, gradient, scale, step_rotation, step_translation):
    gwx, gwy, gwc, gqx, gqy, gqc, _ = gradient
    count = len(c)
    x_new = right_update(x, scale * step_rotation * gwx / count, scale * step_translation * gqx / count)
    y_new = right_update(y, scale * step_rotation * gwy / count, scale * step_translation * gqy / count)
    c_new = np.empty_like(c)
    for i in range(count):
        c_new[i] = right_update(c[i], scale * step_rotation * gwc[i], scale * step_translation * gqc[i])
    return x_new, y_new, c_new


def solve_axyb_prob(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    inv_sigma_wn: np.ndarray,
    inv_sigma_pn: np.ndarray,
    inv_sigma_wm: np.ndarray,
    inv_sigma_pm: np.ndarray,
    noise_configuration: int,
    step_rotation: float = 0.05,
    step_translation: float = 0.05,
    *,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    backend: str = "scipy",
    return_result: bool = False,
) -> tuple[np.ndarray, np.ndarray] | SolverResult:
    """Maximum-likelihood calibration for paper noise configuration 1 or 2.

    This is the Python counterpart of MATLAB ``solveAXYB_prob``. ``numpy`` and
    ``numba`` use the same analytic-gradient/backtracking algorithm. ``scipy``
    uses sparse trust-region least squares and interprets ``max_iterations`` as
    its maximum number of objective evaluations.
    """
    if noise_configuration not in (1, 2):
        raise ValueError("noise_configuration must be 1 or 2")
    if step_rotation <= 0 or step_translation <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("steps and tolerance must be positive; max_iterations must be at least one")
    backend = _check_backend(backend)
    a_array, b_array, x, y = _prepare_common(a, b, x0, y0)
    count = len(a_array)
    precisions = tuple(
        covariances(value, count, name)
        for value, name in (
            (inv_sigma_wn, "inv_sigma_wn"),
            (inv_sigma_pn, "inv_sigma_pn"),
            (inv_sigma_wm, "inv_sigma_wm"),
            (inv_sigma_pm, "inv_sigma_pm"),
        )
    )
    c = np.empty_like(a_array)
    for i in range(count):
        denominator = sum(np.trace(item[i]) for item in precisions)
        fraction = np.trace(precisions[2][i] + precisions[3][i]) / denominator
        c[i] = interpolate_se3(a_array[i] @ x, y @ b_array[i], float(fraction))

    if backend == "scipy":
        from ._scipy_backend import solve

        x, y, c, current, iteration, converged = solve(
            a_array,
            b_array,
            x,
            y,
            c,
            precisions,
            noise_configuration,
            max_iterations,
            tolerance,
        )
        n_noise, m_noise = np.empty_like(c), np.empty_like(c)
        for i in range(count):
            n_noise[i], m_noise[i] = _noise_terms(x, y, a_array[i], b_array[i], c[i], noise_configuration)
        result = SolverResult(x, y, current, iteration, converged, c, n_noise, m_noise)
        return result if return_result else (x, y)

    if backend == "numba":
        from . import _numba_backend

        def cost_function(x_value, y_value, c_value):
            return float(
                _numba_backend.cost(
                    x_value, y_value, a_array, b_array, c_value, *precisions, noise_configuration
                )
            )

        def gradient_function(x_value, y_value, c_value):
            return _numba_backend.gradient(
                x_value, y_value, a_array, b_array, c_value, *precisions, noise_configuration
            )
    else:
        def cost_function(x_value, y_value, c_value):
            return compute_log_likelihood(
                x_value,
                y_value,
                a_array,
                b_array,
                c_value,
                *precisions,
                noise_configuration,
            )

        def gradient_function(x_value, y_value, c_value):
            return _gradient(
                x_value, y_value, a_array, b_array, c_value, precisions, noise_configuration
            )

    converged = False
    current = cost_function(x, y, c)
    scale = 1.0
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        gradient = gradient_function(x, y, c)
        accepted = False
        trial_scale = scale
        for _ in range(24):
            candidate = _apply_update(x, y, c, gradient, trial_scale, step_rotation, step_translation)
            candidate_cost = cost_function(*candidate)
            if candidate_cost >= current - 1e-13:
                accepted = True
                break
            trial_scale *= 0.5
        if not accepted:
            break
        improvement = candidate_cost - current
        x, y, c, current = *candidate, candidate_cost
        scale = min(1.0, trial_scale * 1.25)
        if improvement <= tolerance * (1.0 + abs(current)):
            converged = True
            break
        if iteration % 200 == 0:
            x[:3, :3] = project_so3(x[:3, :3])
            y[:3, :3] = project_so3(y[:3, :3])
            for item in c:
                item[:3, :3] = project_so3(item[:3, :3])

    n_noise, m_noise = np.empty_like(c), np.empty_like(c)
    for i in range(count):
        n_noise[i], m_noise[i] = _noise_terms(x, y, a_array[i], b_array[i], c[i], noise_configuration)
    result = SolverResult(x, y, current, iteration, converged, c, n_noise, m_noise)
    return result if return_result else (x, y)


def _gradient_noiseless_a(x, y, a, b, wm_precision, pm_precision):
    inv_x = inv_se3(x)
    gwx, gwy, gqx, gqy = (np.zeros(3) for _ in range(4))
    total = 0.0
    for i in range(len(a)):
        noise = inv_x @ inv_se3(a[i]) @ y @ b[i]
        rm = noise[:3, :3]
        wm, pm = log_so3(rm), noise[:3, 3]
        h = wm @ wm_precision[i] @ inverse_right_jacobian_so3(wm)
        weighted_p = pm @ pm_precision[i]
        gwx += h - weighted_p @ skew(pm)
        gwxy = rm @ b[i, :3, :3].T
        gwy += -h @ gwxy + weighted_p @ gwxy @ skew(b[i, :3, 3])
        gqx += weighted_p
        gqy -= weighted_p @ gwxy
        total -= 0.5 * (wm @ wm_precision[i] @ wm + pm @ pm_precision[i] @ pm)
    return gwx, gwy, gqx, gqy, float(total)


def solve_axyb_prob_noiseless_a(
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    inv_sigma_wm: np.ndarray,
    inv_sigma_pm: np.ndarray,
    step_rotation: float = 0.05,
    step_translation: float = 0.05,
    *,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    backend: str = "scipy",
    return_result: bool = False,
) -> tuple[np.ndarray, np.ndarray] | SolverResult:
    """Maximum-likelihood calibration for noise configuration 3 (noiseless A)."""
    if step_rotation <= 0 or step_translation <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("steps and tolerance must be positive; max_iterations must be at least one")
    backend = _check_backend(backend)
    a_array, b_array, x, y = _prepare_common(a, b, x0, y0)
    count = len(a_array)
    wm_precision = covariances(inv_sigma_wm, count, "inv_sigma_wm")
    pm_precision = covariances(inv_sigma_pm, count, "inv_sigma_pm")
    if backend == "scipy":
        from ._scipy_backend import solve_noiseless_a

        x, y, current, iteration, converged = solve_noiseless_a(
            a_array,
            b_array,
            x,
            y,
            wm_precision,
            pm_precision,
            max_iterations,
            tolerance,
        )
        result = SolverResult(x, y, current, iteration, converged)
        return result if return_result else (x, y)

    if backend == "numba":
        from . import _numba_backend

        def cost_function(x_value, y_value):
            return float(
                _numba_backend.cost_noiseless_a(
                    x_value, y_value, a_array, b_array, wm_precision, pm_precision
                )
            )

        def gradient_function(x_value, y_value):
            return _numba_backend.gradient_noiseless_a(
                x_value, y_value, a_array, b_array, wm_precision, pm_precision
            )
    else:
        def cost_function(x_value, y_value):
            return compute_log_likelihood_noiseless_a(
                x_value, y_value, a_array, b_array, wm_precision, pm_precision
            )

        def gradient_function(x_value, y_value):
            return _gradient_noiseless_a(
                x_value, y_value, a_array, b_array, wm_precision, pm_precision
            )

    current = cost_function(x, y)
    scale, converged, iteration = 1.0, False, 0
    for iteration in range(1, max_iterations + 1):
        gradient = gradient_function(x, y)
        gwx, gwy, gqx, gqy, _ = gradient
        trial_scale, accepted = scale, False
        for _ in range(24):
            candidate_x = right_update(
                x, trial_scale * step_rotation * gwx / count, trial_scale * step_translation * gqx / count
            )
            candidate_y = right_update(
                y, trial_scale * step_rotation * gwy / count, trial_scale * step_translation * gqy / count
            )
            candidate_cost = cost_function(candidate_x, candidate_y)
            if candidate_cost >= current - 1e-13:
                accepted = True
                break
            trial_scale *= 0.5
        if not accepted:
            break
        improvement = candidate_cost - current
        x, y, current = candidate_x, candidate_y, candidate_cost
        scale = min(1.0, trial_scale * 1.25)
        if improvement <= tolerance * (1.0 + abs(current)):
            converged = True
            break
        if iteration % 200 == 0:
            x[:3, :3] = project_so3(x[:3, :3])
            y[:3, :3] = project_so3(y[:3, :3])
    result = SolverResult(x, y, current, iteration, converged)
    return result if return_result else (x, y)
