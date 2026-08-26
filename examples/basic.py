"""Python equivalent of ``main_exmaple1.m``."""

import numpy as np

from probabilistic_axyb import exp_so3, inv_se3, random_se3, solve_axyb_prob


def main() -> None:
    rng = np.random.default_rng(7)
    x_true, y_true = random_se3(rng=rng), random_se3(rng=rng)
    a = np.stack([random_se3(rng=rng) for _ in range(20)])
    b = np.stack([inv_se3(y_true) @ ai @ x_true for ai in a])
    precision = np.repeat(np.eye(3)[None, ...], len(a), axis=0)

    x0, y0 = x_true.copy(), y_true.copy()
    x0[:3, :3] = x0[:3, :3] @ exp_so3(0.2 * rng.normal(size=3))
    x0[:3, 3] += 0.2 * rng.normal(size=3)
    y0[:3, :3] = y0[:3, :3] @ exp_so3(0.2 * rng.normal(size=3))
    y0[:3, 3] += 0.2 * rng.normal(size=3)

    x_est, y_est = solve_axyb_prob(
        a, b, x0, y0, precision, precision, precision, precision, noise_configuration=1
    )
    print("X Frobenius error:", np.linalg.norm(x_true - x_est))
    print("Y Frobenius error:", np.linalg.norm(y_true - y_est))


if __name__ == "__main__":
    main()

