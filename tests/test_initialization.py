import unittest

import numpy as np

from probabilistic_axyb import (
    exp_so3,
    inv_se3,
    log_so3,
    random_se3,
    rotation_excitation,
    solve_axyb,
)


def transform_error(reference, estimate):
    delta = inv_se3(reference) @ estimate
    return np.linalg.norm(log_so3(delta[:3, :3])), np.linalg.norm(delta[:3, 3])


def make_problem(seed, count=20, axis=None):
    rng = np.random.default_rng(seed)
    x_true, y_true = random_se3(rng=rng), random_se3(rng=rng)
    if axis is None:
        a = np.stack([random_se3(rng=rng) for _ in range(count)])
    else:
        a = np.empty((count, 4, 4))
        for i, angle in enumerate(np.linspace(-1.5, 1.5, count)):
            a[i] = np.eye(4)
            a[i][:3, :3] = exp_so3(angle * np.asarray(axis, dtype=float))
            a[i][:3, 3] = rng.normal(size=3)
    b = np.stack([inv_se3(y_true) @ item @ x_true for item in a])
    return x_true, y_true, a, b


class InitializationTests(unittest.TestCase):
    def test_recovers_solution_when_singular_vector_sign_is_negative(self):
        # Seed 0 makes the reshaped leading right-singular vector a reflection.
        # Projecting it directly returns the nearest rotation to a reflection,
        # which used to leave the initializer roughly 100 degrees off.
        x_true, y_true, a, b = make_problem(0)
        correlation = sum(np.kron(bi[:3, :3], ai[:3, :3]) for ai, bi in zip(a, b))
        _, _, vt = np.linalg.svd(correlation)
        self.assertLess(np.linalg.det(vt[0].reshape((3, 3), order="F")), 0.0)

        x, y = solve_axyb(a, b)
        for error in (*transform_error(x_true, x), *transform_error(y_true, y)):
            self.assertLess(error, 1e-8)

    def test_rejects_single_axis_rotation(self):
        _, _, a, b = make_problem(5, axis=(0.0, 0.0, 1.0))
        self.assertLess(rotation_excitation(a), 1e-6)
        with self.assertRaisesRegex(ValueError, "single axis"):
            solve_axyb(a, b)

    def test_rejects_stationary_motion(self):
        _, _, a, b = make_problem(7)
        a = np.repeat(a[:1], len(a), axis=0)
        b = np.repeat(b[:1], len(b), axis=0)
        self.assertEqual(rotation_excitation(a), 0.0)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            solve_axyb(a, b)

    def test_excitation_is_high_for_well_conditioned_motion(self):
        _, _, a, _ = make_problem(0)
        self.assertGreater(rotation_excitation(a), 0.3)


if __name__ == "__main__":
    unittest.main()
