"""Probabilistic hand-eye and robot-world calibration for ``AX = YB``."""

from .initialization import rotation_excitation, solve_axyb
from .lie import exp_so3, inv_se3, log_so3, project_so3
from .solver import (
    SolverResult,
    available_backends,
    compute_log_likelihood,
    compute_log_likelihood_noiseless_a,
    solve_axyb_prob,
    solve_axyb_prob_noiseless_a,
)
from .synthetic import add_noise_se3, invert_covariances, random_se3, random_so3
from .uncertainty import compute_uncertainty, compute_uncertainty_noiseless_a

__all__ = [
    "SolverResult",
    "available_backends",
    "add_noise_se3",
    "compute_log_likelihood",
    "compute_log_likelihood_noiseless_a",
    "compute_uncertainty",
    "compute_uncertainty_noiseless_a",
    "exp_so3",
    "inv_se3",
    "invert_covariances",
    "log_so3",
    "project_so3",
    "random_se3",
    "random_so3",
    "rotation_excitation",
    "solve_axyb",
    "solve_axyb_prob",
    "solve_axyb_prob_noiseless_a",
]
