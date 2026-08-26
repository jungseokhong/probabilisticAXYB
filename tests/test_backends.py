import unittest

import numpy as np

from probabilistic_axyb import (
    available_backends,
    exp_so3,
    inv_se3,
    random_se3,
    solve_axyb_prob,
    solve_axyb_prob_noiseless_a,
)


class OptionalBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(81)
        cls.x_true, cls.y_true = random_se3(rng=rng), random_se3(rng=rng)
        cls.a = np.stack([random_se3(rng=rng) for _ in range(8)])
        cls.b = np.stack([inv_se3(cls.y_true) @ item @ cls.x_true for item in cls.a])
        cls.precision = np.repeat(np.eye(3)[None], len(cls.a), axis=0)
        cls.x0, cls.y0 = cls.x_true.copy(), cls.y_true.copy()
        cls.x0[:3, :3] = cls.x0[:3, :3] @ exp_so3(np.array([0.04, -0.03, 0.02]))
        cls.y0[:3, :3] = cls.y0[:3, :3] @ exp_so3(np.array([-0.03, 0.02, 0.04]))
        cls.x0[:3, 3] += 0.04
        cls.y0[:3, 3] -= 0.04

    def solve_probabilistic(self, backend, noise_configuration=1):
        return solve_axyb_prob(
            self.a,
            self.b,
            self.x0,
            self.y0,
            self.precision,
            self.precision,
            self.precision,
            self.precision,
            noise_configuration,
            max_iterations=800,
            tolerance=1e-9,
            backend=backend,
            return_result=True,
        )

    @unittest.skipUnless("numba" in available_backends(), "Numba is not installed")
    def test_numba_matches_numpy_algorithm(self):
        for configuration in (1, 2):
            with self.subTest(noise_configuration=configuration):
                numpy_result = self.solve_probabilistic("numpy", configuration)
                numba_result = self.solve_probabilistic("numba", configuration)
                self.assertAlmostEqual(numpy_result.log_likelihood, numba_result.log_likelihood, places=12)
                self.assertTrue(np.allclose(numpy_result.x, numba_result.x, atol=1e-12))
                self.assertTrue(np.allclose(numpy_result.y, numba_result.y, atol=1e-12))

        arguments = (
            self.a,
            self.b,
            self.x0,
            self.y0,
            self.precision,
            self.precision,
        )
        numpy_result = solve_axyb_prob_noiseless_a(
            *arguments, max_iterations=800, tolerance=1e-9, backend="numpy", return_result=True
        )
        numba_result = solve_axyb_prob_noiseless_a(
            *arguments, max_iterations=800, tolerance=1e-9, backend="numba", return_result=True
        )
        self.assertAlmostEqual(numpy_result.log_likelihood, numba_result.log_likelihood, places=12)
        self.assertTrue(np.allclose(numpy_result.x, numba_result.x, atol=1e-12))
        self.assertTrue(np.allclose(numpy_result.y, numba_result.y, atol=1e-12))

    @unittest.skipUnless("scipy" in available_backends(), "SciPy is not installed")
    def test_scipy_converges_for_all_noise_configurations(self):
        configuration_1 = self.solve_probabilistic("scipy", 1)
        configuration_2 = self.solve_probabilistic("scipy", 2)
        configuration_3 = solve_axyb_prob_noiseless_a(
            self.a,
            self.b,
            self.x0,
            self.y0,
            self.precision,
            self.precision,
            max_iterations=200,
            tolerance=1e-10,
            backend="scipy",
            return_result=True,
        )
        self.assertTrue(configuration_1.converged)
        self.assertTrue(configuration_2.converged)
        self.assertTrue(configuration_3.converged)
        self.assertGreater(configuration_1.log_likelihood, -1e-14)
        self.assertGreater(configuration_2.log_likelihood, -1e-14)
        self.assertGreater(configuration_3.log_likelihood, -1e-14)

    @unittest.skipUnless("scipy" in available_backends(), "SciPy is not installed")
    def test_scipy_is_the_default(self):
        default_result = solve_axyb_prob_noiseless_a(
            self.a,
            self.b,
            self.x0,
            self.y0,
            self.precision,
            self.precision,
            max_iterations=200,
            tolerance=1e-10,
            return_result=True,
        )
        scipy_result = solve_axyb_prob_noiseless_a(
            self.a,
            self.b,
            self.x0,
            self.y0,
            self.precision,
            self.precision,
            max_iterations=200,
            tolerance=1e-10,
            backend="scipy",
            return_result=True,
        )
        self.assertTrue(np.allclose(default_result.x, scipy_result.x, atol=1e-13))
        self.assertTrue(np.allclose(default_result.y, scipy_result.y, atol=1e-13))

    def test_invalid_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            solve_axyb_prob_noiseless_a(
                self.a,
                self.b,
                self.x0,
                self.y0,
                self.precision,
                self.precision,
                backend="unknown",
            )


if __name__ == "__main__":
    unittest.main()
