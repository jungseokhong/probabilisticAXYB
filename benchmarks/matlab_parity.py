"""Run deterministic MATLAB/Python equation and solver parity checks."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from probabilistic_axyb import (
    compute_log_likelihood,
    compute_log_likelihood_noiseless_a,
    compute_uncertainty,
    compute_uncertainty_noiseless_a,
    exp_so3,
    inv_se3,
    log_so3,
    random_se3,
    solve_axyb_prob,
    solve_axyb_prob_noiseless_a,
)
from probabilistic_axyb.lie import interpolate_se3
from probabilistic_axyb.solver import _gradient, _gradient_noiseless_a


ROOT = Path(__file__).resolve().parents[1]
MATLAB_SCRIPT_DIR = ROOT / "tests" / "matlab"


def _matlab_layout(array):
    return np.moveaxis(array, 0, 2)


def make_fixture(count=10, seed=2026):
    rng = np.random.default_rng(seed)
    x_true, y_true = random_se3(rng=rng), random_se3(rng=rng)
    a = np.stack([random_se3(rng=rng) for _ in range(count)])
    b = np.stack([inv_se3(y_true) @ item @ x_true for item in a])
    x0, y0 = x_true.copy(), y_true.copy()
    x0[:3, :3] = x0[:3, :3] @ exp_so3(np.array([0.08, -0.05, 0.04]))
    y0[:3, :3] = y0[:3, :3] @ exp_so3(np.array([-0.04, 0.07, 0.05]))
    x0[:3, 3] += np.array([0.08, -0.04, 0.06])
    y0[:3, 3] += np.array([-0.05, 0.07, -0.03])

    precisions = []
    for family in range(4):
        matrices = np.empty((count, 3, 3))
        for i in range(count):
            basis = rng.normal(size=(3, 3))
            matrices[i] = basis.T @ basis + (1.0 + 0.25 * family) * np.eye(3)
        precisions.append(matrices)

    c = np.empty_like(a)
    for i in range(count):
        denominator = sum(np.trace(item[i]) for item in precisions)
        fraction = np.trace(precisions[2][i] + precisions[3][i]) / denominator
        c[i] = interpolate_se3(a[i] @ x0, y0 @ b[i], float(fraction))
    return x_true, y_true, a, b, x0, y0, c, tuple(precisions)


def save_fixture(path, fixture):
    x_true, y_true, a, b, x0, y0, c, precisions = fixture
    savemat(
        path,
        {
            "X_true": x_true,
            "Y_true": y_true,
            "A": _matlab_layout(a),
            "B": _matlab_layout(b),
            "X0": x0,
            "Y0": y0,
            "C": _matlab_layout(c),
            "invSig_wN": _matlab_layout(precisions[0]),
            "invSig_pN": _matlab_layout(precisions[1]),
            "invSig_wM": _matlab_layout(precisions[2]),
            "invSig_pM": _matlab_layout(precisions[3]),
            "step_R": 0.05,
            "step_p": 0.05,
        },
    )


def maximum_absolute(first, second):
    return float(np.max(np.abs(np.asarray(first).squeeze() - np.asarray(second).squeeze())))


def transform_error(reference, estimate):
    delta = inv_se3(reference) @ estimate
    return float(np.linalg.norm(log_so3(delta[:3, :3]))), float(np.linalg.norm(delta[:3, 3]))


def python_equation_outputs(fixture):
    _, _, a, b, x0, y0, c, precisions = fixture
    result = {}
    for configuration in (1, 2):
        gradient = _gradient(x0, y0, a, b, c, precisions, configuration)
        result[f"J_conf{configuration}"] = compute_log_likelihood(
            x0, y0, a, b, c, *precisions, configuration
        )
        result[f"J_grad_conf{configuration}"] = gradient[6]
        for name, value in zip(("GwX", "GwY", "GwC", "GqX", "GqY", "GqC"), gradient[:6]):
            result[f"{name}_conf{configuration}"] = value
        cov_x, cov_y, mapping = compute_uncertainty(x0, y0, c, *precisions, configuration)
        result[f"covX_conf{configuration}"] = cov_x
        result[f"covY_conf{configuration}"] = cov_y
        result[f"Z_conf{configuration}"] = mapping

    gradient = _gradient_noiseless_a(x0, y0, a, b, precisions[2], precisions[3])
    result["J_conf3"] = compute_log_likelihood_noiseless_a(
        x0, y0, a, b, precisions[2], precisions[3]
    )
    result["J_grad_conf3"] = gradient[4]
    for name, value in zip(("GwX", "GwY", "GqX", "GqY"), gradient[:4]):
        result[f"{name}_conf3"] = value
    cov_x, cov_y, mapping = compute_uncertainty_noiseless_a(
        x0, y0, b, precisions[2], precisions[3]
    )
    result["covX_conf3"], result["covY_conf3"], result["Z_conf3"] = cov_x, cov_y, mapping
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--keep-files", action="store_true")
    args = parser.parse_args()
    fixture = make_fixture()
    temporary = tempfile.TemporaryDirectory(prefix="probabilistic-axyb-parity-")
    directory = Path(temporary.name)
    input_path, output_path = directory / "input.mat", directory / "matlab_output.mat"
    save_fixture(input_path, fixture)
    command = (
        f"addpath('{MATLAB_SCRIPT_DIR.as_posix()}'); "
        f"run_parity('{input_path.as_posix()}', '{output_path.as_posix()}');"
    )
    subprocess.run([args.matlab, "-batch", command], cwd=ROOT, check=True)
    matlab = loadmat(output_path)
    python = python_equation_outputs(fixture)

    equation_errors = []
    print("Equation-level maximum absolute differences:")
    for name, value in python.items():
        difference = maximum_absolute(value, matlab[name])
        equation_errors.append(difference)
        print(f"  {name:<12} {difference:.6e}")

    x_true, y_true, a, b, x0, y0, _, precisions = fixture
    solver_precision = np.repeat(np.eye(3)[None], len(a), axis=0)
    solver_precisions = (solver_precision,) * 4
    print("\nFull-solver accuracy against known noiseless X/Y (identity precisions):")
    print("implementation  conf  log_likelihood    X rot       X trans     Y rot       Y trans")
    solver_accuracy = []
    for configuration in (1, 2):
        for implementation, backend in (("Python/SciPy", "scipy"), ("Python/NumPy", "numpy")):
            result = solve_axyb_prob(
                a, b, x0, y0, *solver_precisions, configuration,
                max_iterations=5000, tolerance=1e-10, backend=backend, return_result=True
            )
            x_error, y_error = transform_error(x_true, result.x), transform_error(y_true, result.y)
            solver_accuracy.append((implementation, configuration, max(*x_error, *y_error)))
            print(
                f"{implementation:<15} {configuration:>4d} {result.log_likelihood:>15.6e} "
                f"{x_error[0]:>11.3e} {x_error[1]:>11.3e} {y_error[0]:>11.3e} {y_error[1]:>11.3e}"
            )
        x_matlab, y_matlab = matlab[f"X_solver_conf{configuration}"], matlab[f"Y_solver_conf{configuration}"]
        x_error, y_error = transform_error(x_true, x_matlab), transform_error(y_true, y_matlab)
        solver_accuracy.append(("MATLAB", configuration, max(*x_error, *y_error)))
        print(
            f"{'MATLAB':<15} {configuration:>4d} {float(matlab[f'J_solver_conf{configuration}'].item()):>15.6e} "
            f"{x_error[0]:>11.3e} {x_error[1]:>11.3e} {y_error[0]:>11.3e} {y_error[1]:>11.3e}"
        )

    for implementation, backend in (("Python/SciPy", "scipy"), ("Python/NumPy", "numpy")):
        result = solve_axyb_prob_noiseless_a(
            a, b, x0, y0, solver_precision, solver_precision,
            max_iterations=5000, tolerance=1e-10, backend=backend, return_result=True
        )
        x_error, y_error = transform_error(x_true, result.x), transform_error(y_true, result.y)
        solver_accuracy.append((implementation, 3, max(*x_error, *y_error)))
        print(
            f"{implementation:<15} {3:>4d} {result.log_likelihood:>15.6e} "
            f"{x_error[0]:>11.3e} {x_error[1]:>11.3e} {y_error[0]:>11.3e} {y_error[1]:>11.3e}"
        )
    x_error = transform_error(x_true, matlab["X_solver_conf3"])
    y_error = transform_error(y_true, matlab["Y_solver_conf3"])
    solver_accuracy.append(("MATLAB", 3, max(*x_error, *y_error)))
    print(
        f"{'MATLAB':<15} {3:>4d} {float(matlab['J_solver_conf3'].item()):>15.6e} "
        f"{x_error[0]:>11.3e} {x_error[1]:>11.3e} {y_error[0]:>11.3e} {y_error[1]:>11.3e}"
    )

    maximum_error = max(equation_errors)
    print(f"\nMaximum equation-level difference: {maximum_error:.6e}")
    if maximum_error > 1e-8:
        raise SystemExit("MATLAB/Python equation parity exceeded tolerance 1e-8")
    for implementation, configuration, error in solver_accuracy:
        tolerance = 1e-8 if implementation == "Python/SciPy" else 1e-2
        if error > tolerance:
            raise SystemExit(
                f"{implementation} configuration {configuration} solver error {error:g} "
                f"exceeded tolerance {tolerance:g}"
            )
    if args.keep_files:
        kept = ROOT / "matlab_parity_artifacts"
        kept.mkdir(exist_ok=True)
        save_fixture(kept / "input.mat", fixture)
        (kept / "matlab_output.mat").write_bytes(output_path.read_bytes())
        print(f"Artifacts copied to {kept}")
    temporary.cleanup()


if __name__ == "__main__":
    main()
