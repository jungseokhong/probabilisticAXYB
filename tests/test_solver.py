import unittest

import numpy as np

from probabilistic_axyb import (
    compute_log_likelihood_noiseless_a,
    compute_uncertainty,
    compute_uncertainty_noiseless_a,
    exp_so3,
    inv_se3,
    log_so3,
    random_se3,
    solve_axyb,
    solve_axyb_prob,
    solve_axyb_prob_noiseless_a,
)


def transform_error(reference, estimate):
    delta = inv_se3(reference) @ estimate
    return np.linalg.norm(log_so3(delta[:3, :3])), np.linalg.norm(delta[:3, 3])


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2023)
        cls.x_true, cls.y_true = random_se3(rng=rng), random_se3(rng=rng)
        cls.a = np.stack([random_se3(rng=rng) for _ in range(20)])
        cls.b = np.stack([inv_se3(cls.y_true) @ item @ cls.x_true for item in cls.a])
        cls.precision = np.repeat(np.eye(3)[None, ...], len(cls.a), axis=0)

    def test_closed_form_initializer_recovers_noiseless_problem(self):
        x, y = solve_axyb(self.a, self.b)
        for error in (*transform_error(self.x_true, x), *transform_error(self.y_true, y)):
            self.assertLess(error, 1e-8)

    def test_configuration_1_improves_and_recovers_solution(self):
        rng = np.random.default_rng(4)
        x0, y0 = self.x_true.copy(), self.y_true.copy()
        x0[:3, :3] = x0[:3, :3] @ exp_so3(0.15 * rng.normal(size=3))
        y0[:3, :3] = y0[:3, :3] @ exp_so3(0.15 * rng.normal(size=3))
        x0[:3, 3] += 0.15 * rng.normal(size=3)
        y0[:3, 3] += 0.15 * rng.normal(size=3)
        result = solve_axyb_prob(
            self.a,
            self.b,
            x0,
            y0,
            self.precision,
            self.precision,
            self.precision,
            self.precision,
            1,
            max_iterations=2500,
            return_result=True,
        )
        self.assertGreater(result.log_likelihood, -1e-7)
        for error in (*transform_error(self.x_true, result.x), *transform_error(self.y_true, result.y)):
            self.assertLess(error, 2e-3)

        cov_x, cov_y, mapping = compute_uncertainty(
            result.x,
            result.y,
            result.c,
            self.precision,
            self.precision,
            self.precision,
            self.precision,
            1,
        )
        self.assertEqual((cov_x.shape, cov_y.shape, mapping.shape), ((6, 6), (6, 6), (132, 240)))

    def test_configuration_2_recovers_noiseless_problem(self):
        rng = np.random.default_rng(14)
        x0, y0 = self.x_true.copy(), self.y_true.copy()
        x0[:3, :3] = x0[:3, :3] @ exp_so3(0.1 * rng.normal(size=3))
        y0[:3, :3] = y0[:3, :3] @ exp_so3(0.1 * rng.normal(size=3))
        x0[:3, 3] += 0.1 * rng.normal(size=3)
        y0[:3, 3] += 0.1 * rng.normal(size=3)
        result = solve_axyb_prob(
            self.a,
            self.b,
            x0,
            y0,
            self.precision,
            self.precision,
            self.precision,
            self.precision,
            2,
            max_iterations=2500,
            return_result=True,
        )
        self.assertGreater(result.log_likelihood, -1e-7)
        for error in (*transform_error(self.x_true, result.x), *transform_error(self.y_true, result.y)):
            self.assertLess(error, 2e-3)

    def test_configuration_3_recovers_noiseless_problem_and_uncertainty(self):
        rng = np.random.default_rng(8)
        x0, y0 = self.x_true.copy(), self.y_true.copy()
        x0[:3, :3] = x0[:3, :3] @ exp_so3(0.12 * rng.normal(size=3))
        y0[:3, :3] = y0[:3, :3] @ exp_so3(0.12 * rng.normal(size=3))
        x0[:3, 3] += 0.12 * rng.normal(size=3)
        y0[:3, 3] += 0.12 * rng.normal(size=3)
        initial = compute_log_likelihood_noiseless_a(x0, y0, self.a, self.b, self.precision, self.precision)
        result = solve_axyb_prob_noiseless_a(
            self.a,
            self.b,
            x0,
            y0,
            self.precision,
            self.precision,
            max_iterations=2500,
            return_result=True,
        )
        self.assertGreater(result.log_likelihood, initial)
        for error in (*transform_error(self.x_true, result.x), *transform_error(self.y_true, result.y)):
            self.assertLess(error, 2e-3)
        cov_x, cov_y, mapping = compute_uncertainty_noiseless_a(
            result.x, result.y, self.b, self.precision, self.precision
        )
        self.assertEqual(cov_x.shape, (6, 6))
        self.assertEqual(cov_y.shape, (6, 6))
        self.assertEqual(mapping.shape, (12, 120))
        self.assertGreaterEqual(np.linalg.eigvalsh(cov_x).min(), -1e-10)

    def test_matlab_tensor_layout_is_accepted(self):
        x, y = solve_axyb(np.moveaxis(self.a, 0, 2), np.moveaxis(self.b, 0, 2))
        self.assertLess(transform_error(self.x_true, x)[0], 1e-8)
        self.assertLess(transform_error(self.y_true, y)[0], 1e-8)

    def test_ambiguous_native_layout_prefers_python_convention(self):
        a, b = self.a[:4], self.b[:4]
        x, y = solve_axyb(a, b)
        self.assertLess(transform_error(self.x_true, x)[0], 1e-8)
        self.assertLess(transform_error(self.y_true, y)[0], 1e-8)

        three_a, three_b = self.a[:3], self.b[:3]
        precisions = np.stack((np.eye(3), 2 * np.eye(3), 3 * np.eye(3)))
        initial = compute_log_likelihood_noiseless_a(
            self.x_true, self.y_true, three_a, three_b, precisions, precisions
        )
        self.assertAlmostEqual(initial, 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
