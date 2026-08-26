import unittest

import numpy as np

from probabilistic_axyb import exp_so3, inv_se3, log_so3, project_so3, random_se3


class LieGroupTests(unittest.TestCase):
    def test_exp_log_round_trip(self):
        for vector in (np.zeros(3), np.array([0.1, -0.2, 0.3]), np.array([2.4, 0.2, -0.1])):
            self.assertTrue(np.allclose(exp_so3(log_so3(exp_so3(vector))), exp_so3(vector), atol=1e-10))

    def test_inverse(self):
        transform = random_se3(rng=np.random.default_rng(1))
        self.assertTrue(np.allclose(transform @ inv_se3(transform), np.eye(4), atol=1e-12))

    def test_projection(self):
        projected = project_so3(np.diag([1.0, 1.0, -1.0]))
        self.assertAlmostEqual(np.linalg.det(projected), 1.0, places=12)
        self.assertTrue(np.allclose(projected.T @ projected, np.eye(3), atol=1e-12))


if __name__ == "__main__":
    unittest.main()

