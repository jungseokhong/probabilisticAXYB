"""Reproducible NumPy/SciPy/Numba solver benchmark."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from probabilistic_axyb import (
    available_backends,
    exp_so3,
    inv_se3,
    log_so3,
    random_se3,
    solve_axyb_prob,
)


@dataclass
class Measurement:
    backend: str
    run: str
    seconds: float
    iterations: int
    log_likelihood: float
    rotation_error: float
    translation_error: float


def make_problem(count: int, seed: int):
    rng = np.random.default_rng(seed)
    x_true, y_true = random_se3(rng=rng), random_se3(rng=rng)
    a = np.stack([random_se3(rng=rng) for _ in range(count)])
    b = np.stack([inv_se3(y_true) @ item @ x_true for item in a])
    precision = np.repeat(np.eye(3)[None], count, axis=0)
    x0, y0 = x_true.copy(), y_true.copy()
    x0[:3, :3] = x0[:3, :3] @ exp_so3(0.12 * rng.normal(size=3))
    y0[:3, :3] = y0[:3, :3] @ exp_so3(0.12 * rng.normal(size=3))
    x0[:3, 3] += 0.12 * rng.normal(size=3)
    y0[:3, 3] += 0.12 * rng.normal(size=3)
    return x_true, y_true, a, b, precision, x0, y0


def measure(backend, label, problem, max_iterations, tolerance):
    x_true, y_true, a, b, precision, x0, y0 = problem
    started = perf_counter()
    result = solve_axyb_prob(
        a,
        b,
        x0,
        y0,
        precision,
        precision,
        precision,
        precision,
        1,
        max_iterations=max_iterations,
        tolerance=tolerance,
        backend=backend,
        return_result=True,
    )
    seconds = perf_counter() - started
    x_delta, y_delta = inv_se3(x_true) @ result.x, inv_se3(y_true) @ result.y
    rotation_error = max(
        np.linalg.norm(log_so3(x_delta[:3, :3])), np.linalg.norm(log_so3(y_delta[:3, :3]))
    )
    translation_error = max(np.linalg.norm(x_delta[:3, 3]), np.linalg.norm(y_delta[:3, 3]))
    return Measurement(
        backend,
        label,
        seconds,
        result.iterations,
        result.log_likelihood,
        rotation_error,
        translation_error,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurements", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    problem = make_problem(args.measurements, args.seed)
    measurements = []
    for backend in available_backends():
        measurements.append(measure(backend, "cold", problem, args.max_iterations, args.tolerance))
        for index in range(args.repeats):
            measurements.append(
                measure(backend, f"warm-{index + 1}", problem, args.max_iterations, args.tolerance)
            )

    print(f"pairs={args.measurements}, max_iterations={args.max_iterations}, tolerance={args.tolerance:g}")
    print("backend  run       seconds   iterations   log_likelihood   rot_error(rad)   trans_error")
    for item in measurements:
        print(
            f"{item.backend:<8} {item.run:<8} {item.seconds:>8.4f} {item.iterations:>12d} "
            f"{item.log_likelihood:>16.6e} {item.rotation_error:>16.6e} {item.translation_error:>13.6e}"
        )
    print("\nWarm-run means:")
    for backend in available_backends():
        warm = [item.seconds for item in measurements if item.backend == backend and item.run.startswith("warm")]
        print(f"  {backend:<8} {np.mean(warm):.4f} s")


if __name__ == "__main__":
    main()

