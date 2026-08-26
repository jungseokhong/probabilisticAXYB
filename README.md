# Probabilistic Framework for Hand-Eye and Robot-World Calibration AX = YB
<br>

MATLAB implementation of Probabilistic Framework for Hand-Eye and Robot-World Calibration AX = YB (IEEE T-RO 2023).

Paper Link: https://ieeexplore.ieee.org/abstract/document/9931998


## Overview

This is a MATLAB code of probabilistic solver for hand-eye and robot-world calibration AX = YB, the detailed algorithm of which is presented in the paper entitled "Probabilistic Framework for Hand–Eye and Robot–World Calibration AX=YB" (IEEE T-RO 2023). The algorithm incorporates different individual noise distributions of measurements A_i and B_i, and also provides calibration uncertainty as an error covariance matrix.

* The code has been uploaded for the published version of the paper. (10/20/2023)
* An instruction file ``instruction.docx`` has been added. (10/20/2023)
* Codes from https://github.com/ihtishamaliktk/RWHE-Calib were used in experiments. Please allow us some time to clean up the codes and properly cite them.

## Instruction
* Please read ``instruction.docx`` for more details.

1. See the three system noise configurations presented in the paper and select one that best fits your system.
2. Calibration functions are different between noise configurations 1,2 and noise configuration 3.
	* For noise configurations 1 and 2, call	
		```
		[X, Y] = solveAXYB_prob(A, B, X0, Y0, invSig_wN, invSig_pN, invSig_wM, invSig_pM, noiseConf, step_R, step_p)
		```
	* For noise configuration 3, call		
		```
		[X, Y] = solveAXYB_prob_noiselessA(A, B, X0, Y0, invSig_wM, invSig_pM, step_R, step_p)
		```		
Here ``X, Y`` are the calibration results, and ``A, B`` are the measurements pairs in size of ``4 X 4 X n`` each (n is the number of measurement pairs). ``invSig_wN, invSig_pN, invSig_wM, invSig_pM`` are the inverses of rotational and translational noise covariances of A and B, each of which is in size of ``3 X 3 X n``. ``step_R, step_p`` are stepsizes for rotation and translation, respectively.

## Demos
* The script ``main_example1.m`` demonstrates a simple example of how to use the solver.
* The scripts ``main_fig5.m``, ``main_fig6.mm``, ... generate the figures in the paper.

## Python version

The `python` branch contains a NumPy implementation of the calibration workflow
described above and in `instruction.docx`. It includes:

* maximum-likelihood solvers for noise configurations 1, 2, and 3;
* the analytic gradients used by the MATLAB implementation;
* a closed-form Shah initializer for `AX = YB` (the Python replacement for
  using `solveAXYB_sgo` merely to obtain an initial estimate), with an
  excitation check that rejects motions from which `X` and `Y` cannot be
  separated;
* calibration uncertainty for all three noise configurations;
* Lie-group, covariance, noise, and synthetic-data utilities; and
* support for both Python `(n, 4, 4)` and MATLAB `(4, 4, n)` tensor layouts.

### Installation

Python 3.10 or newer and NumPy are required. With `uv` installed:

```bash
uv sync
uv run python -m examples.basic
uv run python -m unittest discover -s tests -v
```

SciPy is installed by default. Install the optional Numba backend with:

```bash
uv sync --extra speed
# or: python -m pip install -e '.[speed]'
```

The repository does not require `uv`; a regular virtual environment works too:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m examples.basic
python -m unittest discover -s tests -v
```

### Usage

Arrays should normally have shape `(n, 4, 4)`, with one homogeneous transform
per measurement. A single `3 x 3` precision matrix may be supplied when every
measurement has the same noise model; it is broadcast across all pairs.

```python
import numpy as np

from probabilistic_axyb import solve_axyb, solve_axyb_prob

# A and B have shape (n, 4, 4). Use a closed-form initial estimate.
X0, Y0 = solve_axyb(A, B)

# These are inverse covariance matrices (precisions), as in the MATLAB API.
inv_Sigma_wN = np.eye(3)
inv_Sigma_pN = np.eye(3)
inv_Sigma_wM = np.eye(3)
inv_Sigma_pM = np.eye(3)

X, Y = solve_axyb_prob(
    A,
    B,
    X0,
    Y0,
    inv_Sigma_wN,
    inv_Sigma_pN,
    inv_Sigma_wM,
    inv_Sigma_pM,
    noise_configuration=1,  # or 2
    step_rotation=0.05,
    step_translation=0.05,
)
```

For noise configuration 3, in which `A` is treated as noiseless:

```python
from probabilistic_axyb import solve_axyb_prob_noiseless_a

X, Y = solve_axyb_prob_noiseless_a(
    A,
    B,
    X0,
    Y0,
    inv_Sigma_wM,
    inv_Sigma_pM,
)
```

Pass `return_result=True` to either solver to also receive convergence status,
iteration count, final log likelihood, and (for configurations 1/2) the latent
`C`, `N`, and `M` transformations. See `examples/basic.py` for an executable
counterpart to `main_exmaple1.m`.

### Choosing a noise model on real data

Identifiability first. `AX = YB` separates `X` from `Y` only when the measured
motions rotate about at least two non-parallel axes; a stationary or single-axis
sequence produces a plausible-looking residual alongside an arbitrarily wrong
`Y`. `solve_axyb` now raises on such input, and `rotation_excitation(A)` reports
the margin directly (roughly `0.3`-`0.5` for well-conditioned motion, below
`1e-4` for single-axis motion).

Note that a rank check on the translation system does not catch this. Sensor
jitter makes `numpy.linalg.lstsq` report full rank even for a target that never
moved; only the condition number gives it away. Nor is such a target useless. It
is degenerate for the *joint* problem, but when several targets share one `Y`,
the moving ones determine `Y` and then `X = A^-1 Y B` holds at every pose of the
stationary one, recovered directly with no optimisation.

Noise model second. Only the *ratio* of the `A` and `B` covariances is
identifiable, so one side must be anchored at a trusted value and the other
estimated. Two points matter when picking that anchor on hardware:

* Repeat-to-repeat and within-window variance measure **precision, not
  accuracy**. Returning to the same commanded joint configuration and observing
  a few microns of forward-kinematics spread says the encoders repeat; it says
  nothing about link lengths, joint offsets, or gravity-induced deflection,
  which are identical on every repeat and therefore invisible in a repeat
  covariance. The same holds for motion capture, whose specification is accuracy
  including volume-dependent bias.
* Covariances estimated per pose from a handful of repeats are rank deficient by
  construction (two samples give a rank-one `3x3`). Inverting them, even with an
  eigenvalue floor, yields near-hard constraints along essentially arbitrary
  directions. Pool the residuals across all poses of a target instead, and pass
  a single broadcast `3x3` per block.

To measure the combined error without committing to any calibration estimate,
compare `A_i^-1 A_j` against `B_i^-1 B_j`: they are conjugate, so their rotation
angles and screw-axis displacements must agree. Whatever they disagree by is
measurement error that no modelling choice can flatter.

`examples/calibrate_rby1.py` applies all of this to a robot dataset in which `A`
is forward kinematics and `B` is motion capture: it screens targets by
excitation, reports the invariant disagreement, anchors the mocap covariance at
its specification, re-estimates the forward-kinematics covariance from the
residual, cross-checks the shared robot-world transform across targets, and then
recovers `X` for the deferred targets against that pinned `Y`.

```bash
python -m examples.calibrate_rby1 <dataset directory>
```

Note that treating motion capture as the *noiseless* measurement does not need a
separate solver. `AX = YB` implies `B X^-1 = Y^-1 A`, so passing the swapped
pair to `solve_axyb_prob_noiseless_a` and inverting the results places the noise
on `A` in the link frame.


### Optional solver backends

Both solver functions accept `backend="numpy"`, `backend="numba"`, or
`backend="scipy"`:

```python
from probabilistic_axyb import available_backends, solve_axyb_prob

print(available_backends())
X, Y = solve_axyb_prob(
    A,
    B,
    X0,
    Y0,
    inv_Sigma_wN,
    inv_Sigma_pN,
    inv_Sigma_wM,
    inv_Sigma_pM,
    noise_configuration=1,
    backend="scipy",
)
```

* `scipy` is the default and uses sparse trust-region least squares. It generally
  converges in far fewer objective evaluations and is the recommended backend.
* `numpy` selects the reference analytic-gradient implementation.
* `numba` compiles the same analytic-gradient/backtracking algorithm. Its first
  call includes compilation, while subsequent calls are substantially faster
  and numerically match the NumPy backend.
  `step_rotation` and `step_translation` apply only to the NumPy and Numba
  algorithms.

Requesting Numba when it is not installed raises an `ImportError` with the
corresponding installation command.

### Backend benchmark

Run the reproducible benchmark with:

```bash
python -m benchmarks.benchmark_backends
```

On an AMD Ryzen 9 5900X with Python 3.10, NumPy 2.2.6, SciPy 1.15.3, and
Numba 0.67.0, the default 20-pair noiseless benchmark produced:

| Backend | Cold start | Warm mean | Iterations/evaluations | Max. translation error |
|---|---:|---:|---:|---:|
| NumPy | 6.154 s | 6.183 s | 1811 | `1.02e-4` |
| SciPy | 0.281 s | 0.163 s | 6 | `1.65e-11` |
| Numba | 9.751 s | 0.894 s | 1811 | `1.02e-4` |

The Numba cold result includes compilation using an empty cache. Timings are
hardware-dependent; the benchmark also prints rotation/translation errors and
final log likelihood so speed is not compared independently of solution quality.

### MATLAB parity verification

When MATLAB is installed, run the original `.m` functions and Python side by
side on the same deterministic fixture with:

```bash
python -m benchmarks.matlab_parity
```

The parity fixture uses distinct anisotropic precision matrices for every
measurement. MATLAB R2026a and Python agreed as follows:

* likelihoods: maximum difference `4.44e-16`;
* analytic-gradient blocks: maximum difference `1.69e-14`;
* uncertainty covariances and mappings: maximum difference `5.66e-15`;
* maximum difference across every equation-level output: `1.69e-14`.

The harness separately runs all three full solvers with README-style identity
precisions against known noiseless `X` and `Y`. SciPy recovered the transforms
to between `1e-15` and `5.4e-12`. The original fixed-step MATLAB solver reached
approximately `5e-6` error for configurations 1/2 and `3.8e-3` for configuration
3. The optimizer outputs are not expected to be identical: the equations are
the same, but SciPy uses trust-region least squares, and the Python NumPy/Numba
path adds backtracking and convergence controls to the translated gradients.
