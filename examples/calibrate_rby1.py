"""Calibrate an RB-Y1 mocap/FK dataset with probabilistic AX=YB.

Each target NPZ holds ``A = base_T_link`` (robot FK from ``/tf``) and
``B = map_T_rigid`` (mocap), so solving ``A X = Y B`` recovers the marker
mounting transform ``X = link_T_rigid`` and the robot-world transform
``Y = base_T_map``. ``Y`` is physically shared by every target, which makes
cross-target agreement the strongest available end-to-end check.

Targets are handled in two stages. Those with enough rotational excitation solve
the joint problem and determine ``Y``. Stationary or single-axis targets cannot
determine ``Y`` on their own -- their least-squares solution slides freely along
an unconstrained direction and lands tens of metres away -- but once ``Y`` is
pinned by the first group, ``X = A^-1 Y B`` holds at every pose and is recovered
directly. Such a target is degenerate for the joint problem, not uninformative.

Noise model: mocap is anchored at its specification and forward kinematics is
estimated from the residual. Only the *ratio* of the two covariances is
identifiable, so anchoring one side is required, and mocap is the side with a
trustworthy number. The per-pair precision matrices stored in the NPZ files are
deliberately ignored: estimated from 2-5 repeats each, they are rank deficient by
construction and act as near-hard constraints along arbitrary directions.

Usage::

    python -m examples.calibrate_rby1 <dataset directory>
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from probabilistic_axyb import (
    compute_uncertainty,
    exp_so3,
    inv_se3,
    log_so3,
    project_so3,
    rotation_excitation,
    solve_axyb,
    solve_axyb_prob,
    solve_axyb_prob_noiseless_a,
)

# Mocap manufacturer specification. This is the anchor: it is never re-estimated.
MOCAP_SIGMA_POSITION = 0.3e-3  # metres
MOCAP_SIGMA_ROTATION = np.deg2rad(0.5)  # radians

# Starting guess for forward kinematics, from the X/Y-independent invariant
# analysis below. The iteration is not sensitive to this, it only needs to start
# in the right order of magnitude.
FK_SIGMA_POSITION = 2.0e-3
FK_SIGMA_ROTATION = np.deg2rad(0.3)

NOISE_CONFIGURATION = 2
ITERATIONS = 5
# Below this, relative rotations share a single axis and X, Y are not separable.
MIN_EXCITATION = 1e-3


def isotropic(sigma: float) -> np.ndarray:
    """Precision matrix for an isotropic standard deviation."""
    return np.eye(3) / (sigma * sigma)


def pose_residuals(a, b, x, y):
    """Rotation (rad) and translation (m) of ``(A X)^-1 (Y B)`` per pair."""
    rotation = np.empty(len(a))
    translation = np.empty(len(a))
    for i in range(len(a)):
        delta = inv_se3(a[i] @ x) @ (y @ b[i])
        rotation[i] = np.linalg.norm(log_so3(delta[:3, :3]))
        translation[i] = np.linalg.norm(delta[:3, 3])
    return rotation, translation


def invariant_disagreement(a, b, minimum_angle=np.deg2rad(5.0)):
    """Consistency of A against B without reference to X or Y.

    For any pair ``(i, j)``, ``A_i^-1 A_j`` and ``B_i^-1 B_j`` are conjugate, so
    their rotation angles and their displacements along the screw axis must
    agree. Any disagreement is measurement error, attributable to no particular
    calibration estimate. This is the one error figure in this script that no
    modelling choice can flatter.
    """
    angles = []
    screws = []
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            delta_a = inv_se3(a[i]) @ a[j]
            delta_b = inv_se3(b[i]) @ b[j]
            wa, wb = log_so3(delta_a[:3, :3]), log_so3(delta_b[:3, :3])
            na, nb = np.linalg.norm(wa), np.linalg.norm(wb)
            if na < minimum_angle or nb < minimum_angle:
                continue
            angles.append(abs(na - nb))
            screws.append(abs(delta_a[:3, 3] @ (wa / na) - delta_b[:3, 3] @ (wb / nb)))
    if not angles:
        return float("nan"), float("nan")
    return float(np.median(angles)), float(np.median(screws))


def pooled_covariance(noise):
    """Rotation and position covariance pooled over every pose of one target.

    Pooling is the point: one 3x3 estimated from ~100 samples, rather than the
    per-pose covariances the NPZ files carry, which come from 2-5 repeats and are
    rank deficient by construction.
    """
    rotations = np.array([log_so3(item[:3, :3]) for item in noise])
    positions = np.array([item[:3, 3] for item in noise])
    return np.cov(rotations, rowvar=False), np.cov(positions, rowvar=False)


def regularized_precision(covariance, floor):
    """Invert a covariance with an eigenvalue floor expressed as a sigma.

    The floor is the mocap anchor: the residual cannot resolve a forward
    kinematics error smaller than the instrument that observed it, so claiming
    one would be an artefact of the estimator rather than a measurement.
    """
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, floor * floor)
    return vectors @ np.diag(1.0 / values) @ vectors.T


def effective_sigma(precision):
    """Isotropic-equivalent sigma of a precision matrix."""
    return float(np.sqrt(np.trace(np.linalg.inv(precision)) / 3.0))


def se3_mean(items, iterations=20):
    """Mean of a set of rigid transforms, with the rotation averaged on SO(3)."""
    items = list(items)
    rotation = project_so3(np.mean([item[:3, :3] for item in items], axis=0))
    for _ in range(iterations):
        delta = np.mean([log_so3(rotation.T @ item[:3, :3]) for item in items], axis=0)
        rotation = rotation @ exp_so3(delta)
        if np.linalg.norm(delta) < 1e-15:
            break
    mean = np.eye(4)
    mean[:3, :3] = rotation
    mean[:3, 3] = np.mean([item[:3, 3] for item in items], axis=0)
    return mean


def recover_x_given_y(a, b, y):
    """Estimate X for a target whose motion cannot determine Y on its own.

    With Y already pinned by the moving targets, ``X = A^-1 Y B`` holds at every
    pose independently, so no optimisation is needed: X is the mean of those
    per-pose estimates, and their scatter says whether it can be trusted. A
    stationary or single-axis target is degenerate only for the *joint* problem;
    once Y is known it carries perfectly ordinary information about X.
    """
    estimates = [inv_se3(a[i]) @ y @ b[i] for i in range(len(a))]
    x = se3_mean(estimates)
    inv_x = inv_se3(x)
    rotation = np.array(
        [np.linalg.norm(log_so3((inv_x @ item)[:3, :3])) for item in estimates]
    )
    position = np.array([np.linalg.norm(item[:3, 3] - x[:3, 3]) for item in estimates])
    return x, rotation, position


def calibrate(a, b, iterations=ITERATIONS, verbose=True):
    """Solve one target, re-estimating the FK covariance from the residual."""
    x, y = solve_axyb(a, b)

    mocap_rotation = isotropic(MOCAP_SIGMA_ROTATION)
    mocap_position = isotropic(MOCAP_SIGMA_POSITION)
    fk_rotation = isotropic(FK_SIGMA_ROTATION)
    fk_position = isotropic(FK_SIGMA_POSITION)

    result = None
    for iteration in range(iterations):
        result = solve_axyb_prob(
            a,
            b,
            x,
            y,
            fk_rotation,
            fk_position,
            mocap_rotation,
            mocap_position,
            noise_configuration=NOISE_CONFIGURATION,
            return_result=True,
        )
        x, y = result.x, result.y
        # N is the FK-side noise, right-multiplicative on A in configuration 2.
        rotation_covariance, position_covariance = pooled_covariance(result.n_noise)
        # Floor at the mocap anchor: FK cannot be credibly sharper than the
        # instrument used to observe it.
        fk_rotation = regularized_precision(rotation_covariance, MOCAP_SIGMA_ROTATION)
        fk_position = regularized_precision(position_covariance, MOCAP_SIGMA_POSITION)
        if verbose:
            # Report the sigma actually in force, i.e. after the floor, so the
            # printed number matches the weighting the solver received.
            sigma_w = np.rad2deg(effective_sigma(fk_rotation))
            sigma_p = effective_sigma(fk_position) * 1e3
            floored = "" if sigma_p > MOCAP_SIGMA_POSITION * 1e3 * 1.01 else "  (at mocap floor)"
            rotation, translation = pose_residuals(a, b, x, y)
            print(
                f"      iter {iteration + 1}: FK sigma {sigma_w:6.3f} deg / {sigma_p:6.3f} mm"
                f"   residual {np.rad2deg(rotation).mean():6.3f} deg / {translation.mean() * 1e3:6.3f} mm"
                f"   {'converged' if result.converged else 'NOT CONVERGED'}{floored}"
            )

    covariance_x, covariance_y, _ = compute_uncertainty(
        x,
        y,
        result.c,
        fk_rotation,
        fk_position,
        mocap_rotation,
        mocap_position,
        noise_configuration=NOISE_CONFIGURATION,
    )
    return x, y, covariance_x, covariance_y, (fk_rotation, fk_position), result


def mocap_noiseless(a, b, fk_rotation, fk_position):
    """Sensitivity check treating mocap, not FK, as the exact measurement.

    ``A X = Y B`` implies ``B X^-1 = Y^-1 A``, so feeding the swapped pair to the
    configuration-3 solver puts mocap in the noiseless slot and leaves FK noisy in
    the link frame. The returned transforms are inverted back.
    """
    x0, y0 = solve_axyb(b, a)
    x, y = solve_axyb_prob_noiseless_a(b, a, x0, y0, fk_rotation, fk_position)
    return inv_se3(x), inv_se3(y)


def format_sigma(covariance):
    """One-sigma rotation (deg) and translation (mm) from a 6x6 covariance."""
    rotation = np.rad2deg(np.sqrt(np.trace(covariance[:3, :3]) / 3.0))
    translation = np.sqrt(np.trace(covariance[3:, 3:]) / 3.0) * 1e3
    return rotation, translation


def convention_check(a, b, x, y):
    """Residual of A X = Y B against the arrangements it is most often confused with.

    Getting the order wrong is a silent failure -- every arrangement produces *some*
    answer -- so the check is worth running once per dataset rather than trusting the
    documentation.
    """
    candidates = {
        "A X = Y B  (this convention)": [inv_se3(a[i] @ x) @ (y @ b[i]) for i in range(len(a))],
        "X A = Y B": [inv_se3(x @ a[i]) @ (y @ b[i]) for i in range(len(a))],
        "A X = B Y": [inv_se3(a[i] @ x) @ (b[i] @ y) for i in range(len(a))],
        "A^-1 X = Y B": [inv_se3(inv_se3(a[i]) @ x) @ (y @ b[i]) for i in range(len(a))],
        "A X = Y B^-1": [inv_se3(a[i] @ x) @ (y @ inv_se3(b[i])) for i in range(len(a))],
    }
    out = {}
    for label, deltas in candidates.items():
        rotation = [np.rad2deg(np.linalg.norm(log_so3(d[:3, :3]))) for d in deltas]
        translation = [np.linalg.norm(d[:3, 3]) * 1e3 for d in deltas]
        out[label] = (float(np.sqrt(np.mean(np.square(rotation)))),
                      float(np.sqrt(np.mean(np.square(translation)))))
    return out


def error_scales(data, x, y):
    """The three quantities that get called 'error' but differ by orders of magnitude.

    Reported together because reading one for another is the easiest mistake to make
    with this dataset: sensor jitter, the robot failing to return to the same pose,
    and forward kinematics disagreeing with motion capture are three different things.
    """
    a, b, uid = data["A"], data["B"], data["unique_id"]
    per_frame = float(np.sqrt(np.trace(data["B_within_cov_position"], axis1=1, axis2=2)).mean())
    frames = float(data["B_sample_count"].mean())
    window_mean = per_frame / np.sqrt(max(frames, 1.0))

    residual = np.stack([(inv_se3(a[i] @ x) @ (y @ b[i]))[:3, 3] for i in range(len(a))])
    centred = residual.copy()
    for u in np.unique(uid):
        mask = uid == u
        centred[mask] -= centred[mask].mean(axis=0)
    repeat = float(np.sqrt((np.linalg.norm(centred, axis=1) ** 2).mean()))

    _, screw = invariant_disagreement(a, b)
    return {
        "mocap_frame_jitter_mm": per_frame * 1e3,
        "mocap_window_mean_mm": window_mean * 1e3,
        "non_repeatability_mm": repeat * 1e3,
        "fk_vs_mocap_mm": screw * 1e3,
        "frames_per_window": frames,
    }


def noise_split(result):
    """Split the loop-closure error into its FK side and its mocap side.

    In configuration 2 the solver posits a latent true pose ``C_i`` and writes the
    error as ``N_i`` (forward kinematics) composed with ``M_i`` (motion capture).
    Comparing their magnitudes says which instrument the residual is actually made of.
    """
    n_translation = np.linalg.norm(result.n_noise[:, :3, 3], axis=1)
    m_translation = np.linalg.norm(result.m_noise[:, :3, 3], axis=1)
    return (float(np.sqrt((n_translation ** 2).mean()) * 1e3),
            float(np.sqrt((m_translation ** 2).mean()) * 1e3))


def latent_placement(result, a, b, x, y):
    """How far the inferred true pose C_i sits from each instrument's view of it.

    ``C_i`` is ``base_T_rigid`` -- where the marker cluster actually was. The robot
    says ``A_i X``, motion capture says ``Y B_i``, and C lands between them in
    proportion to what the noise model says each is worth. This is the noise model's
    effect made visible.
    """
    to_robot = [np.linalg.norm((inv_se3(result.c[i]) @ (a[i] @ x))[:3, 3]) for i in range(len(a))]
    to_mocap = [np.linalg.norm((inv_se3(result.c[i]) @ (y @ b[i]))[:3, 3]) for i in range(len(a))]
    return (float(np.sqrt(np.mean(np.square(to_robot))) * 1e3),
            float(np.sqrt(np.mean(np.square(to_mocap))) * 1e3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path, help="directory holding the target NPZ files")
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--skip-sensitivity", action="store_true")
    arguments = parser.parse_args()

    files = sorted(arguments.dataset.glob("*.npz"))
    if not files:
        parser.error(f"no NPZ files in {arguments.dataset}")

    print("=" * 96)
    print("Excitation screening  (AX=YB needs rotations about at least two non-parallel axes)")
    print("=" * 96)
    usable = []
    degenerate = []
    for path in files:
        data = np.load(path)
        a, b = data["A"], data["B"]
        excitation = rotation_excitation(a)
        if excitation < MIN_EXCITATION:
            reason = "stationary" if excitation == 0.0 else "single-axis rotation"
            print(f"  {path.stem:20s} excitation {excitation:9.2e}  deferred to stage 2 ({reason})")
            degenerate.append((path.stem, a, b))
            continue
        print(f"  {path.stem:20s} excitation {excitation:9.2e}  ok  n={len(a)}")
        usable.append((path.stem, a, b))

    if not usable:
        parser.error("every target is degenerate; Y cannot be determined from this dataset")

    print()
    print("=" * 96)
    print("Transform convention  (getting the order wrong fails silently, so check it)")
    print("=" * 96)
    probe_name, probe_a, probe_b = usable[0]
    probe_x, probe_y = solve_axyb(probe_a, probe_b)
    for label, (rotation, translation) in convention_check(probe_a, probe_b, probe_x, probe_y).items():
        print(f"  {label:30s} rot {rotation:8.3f} deg   pos {translation:10.3f} mm")
    print(f"  (checked on {probe_name}; only the true arrangement closes the loop)")

    print()
    print("=" * 96)
    print("Measurement consistency  (independent of X and Y, so no modelling choice can flatter it)")
    print("=" * 96)
    print(f"  {'target':20s} {'median rot':>12} {'median screw':>14}    versus mocap spec")
    for name, a, b in usable:
        angle, screw = invariant_disagreement(a, b)
        print(
            f"  {name:20s} {np.rad2deg(angle):9.3f} deg {screw * 1e3:11.3f} mm"
            f"    {np.rad2deg(angle) / np.rad2deg(MOCAP_SIGMA_ROTATION):5.1f}x / "
            f"{screw / MOCAP_SIGMA_POSITION:5.1f}x"
        )

    print()
    print("=" * 96)
    print(f"Calibration  (configuration {NOISE_CONFIGURATION}; mocap anchored at "
          f"{MOCAP_SIGMA_POSITION * 1e3:.1f} mm / {np.rad2deg(MOCAP_SIGMA_ROTATION):.1f} deg, FK from residual)")
    print("=" * 96)
    solutions = {}
    diagnostics = {}
    for name, a, b in usable:
        print(f"  {name}:")
        x, y, covariance_x, covariance_y, fk, result = calibrate(a, b, arguments.iterations)
        sigma_x = format_sigma(covariance_x)
        sigma_y = format_sigma(covariance_y)
        solutions[name] = (x, y, a, b, fk, sigma_x, sigma_y)
        diagnostics[name] = {
            "scales": error_scales(np.load(arguments.dataset / f"{name}.npz"), x, y),
            "split": noise_split(result),
            "latent": latent_placement(result, a, b, x, y),
        }
        rotation, translation = pose_residuals(a, b, x, y)
        print(f"      residual rms   {np.rad2deg(np.sqrt((rotation ** 2).mean())):6.3f} deg"
              f" / {np.sqrt((translation ** 2).mean()) * 1e3:6.3f} mm")
        print(f"      X = link_T_rigid  t = {np.round(x[:3, 3] * 1e3, 2)} mm"
              f"   1-sigma {sigma_x[0]:.4f} deg / {sigma_x[1]:.4f} mm")
        print(f"      Y = base_T_map    t = {np.round(y[:3, 3] * 1e3, 2)} mm"
              f"   1-sigma {sigma_y[0]:.4f} deg / {sigma_y[1]:.4f} mm")

    print()
    print("=" * 96)
    print("Cross-target agreement of Y = base_T_map  (Y is physically shared: disagreement is system error)")
    print("=" * 96)
    names = list(solutions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            delta = inv_se3(solutions[names[i]][1]) @ solutions[names[j]][1]
            print(f"  {names[i]:20s} vs {names[j]:20s}"
                  f"  {np.rad2deg(np.linalg.norm(log_so3(delta[:3, :3]))):6.3f} deg"
                  f"  {np.linalg.norm(delta[:3, 3]) * 1e3:8.2f} mm")
    centre = np.mean([solutions[name][1][:3, 3] for name in names], axis=0)
    spread = np.max([np.linalg.norm(solutions[name][1][:3, 3] - centre) for name in names])
    reported = float(np.mean([solutions[name][6][1] for name in names]))
    print(f"\n  mean Y translation {np.round(centre * 1e3, 2)} mm, max deviation {spread * 1e3:.2f} mm")
    print(f"  mean reported 1-sigma {reported:.2f} mm")
    if spread * 1e3 > reported:
        print(f"  -> spread exceeds the reported sigma by {spread * 1e3 / reported:.1f}x. The reported")
        print("     covariance assumes the noise model is correct and the error is zero-mean; the")
        print("     dominant FK error is neither. Quote the cross-target spread as the error bar.")

    print()
    print("=" * 96)
    print("Error scales  (three different things, all called 'error', orders of magnitude apart)")
    print("=" * 96)
    print(f"  {'target':20s} {'mocap jitter':>13} {'on window mean':>15} {'non-repeat':>12} {'FK vs mocap':>13}")
    for name in names:
        s = diagnostics[name]["scales"]
        print(f"  {name:20s} {s['mocap_frame_jitter_mm']:10.4f} mm {s['mocap_window_mean_mm']:12.4f} mm"
              f" {s['non_repeatability_mm']:9.4f} mm {s['fk_vs_mocap_mm']:10.4f} mm")
    reference = diagnostics[names[0]]["scales"]
    print(f"\n  Mocap jitter averages down over ~{reference['frames_per_window']:.0f} frames per window, so what")
    print("  actually enters B is the third-decimal figure. Non-repeatability is ~100x that:")
    print("  the robot does not return to the same pose, and only mocap can see it -- FK")
    print("  reports the same pose either way, because the encoders return to the same counts.")

    print()
    print("=" * 96)
    print("What the residual is made of  (configuration 2 splits it into FK and mocap parts)")
    print("=" * 96)
    print(f"  {'target':20s} {'|N| FK side':>12} {'|M| mocap side':>15} {'C to robot':>12} {'C to mocap':>12}")
    for name in names:
        n_rms, m_rms = diagnostics[name]["split"]
        to_robot, to_mocap = diagnostics[name]["latent"]
        print(f"  {name:20s} {n_rms:9.3f} mm {m_rms:12.3f} mm {to_robot:9.3f} mm {to_mocap:9.3f} mm")
    print("\n  C is base_T_rigid, the inferred true pose of the marker cluster. It sits between")
    print("  the robot's view (A X) and mocap's view (Y B), pulled toward whichever the noise")
    print("  model calls more trustworthy. That pull is the noise model's entire effect.")

    if degenerate:
        consensus = se3_mean([solutions[name][1] for name in names])
        print()
        print("=" * 96)
        print("Stage 2: X for the deferred targets, with Y pinned by the targets above")
        print("=" * 96)
        print(f"  consensus Y translation {np.round(consensus[:3, 3] * 1e3, 2)} mm"
              f"  (mean of {len(names)} targets)")
        for name, a, b in degenerate:
            x, rotation, position = recover_x_given_y(a, b, consensus)
            print(f"  {name:20s} X = link_T_rigid  t = {np.round(x[:3, 3] * 1e3, 2)} mm  (n={len(a)})")
            print(f"  {'':20s} per-pose scatter  {np.rad2deg(rotation).std():6.3f} deg rms"
                  f" (max {np.rad2deg(rotation).max():.3f})"
                  f" / {position.std() * 1e3:6.3f} mm rms (max {position.max() * 1e3:.3f})")
        print()
        print(f"  Scatter above is per-pose spread about the mean, not an accuracy figure. Error in the")
        print(f"  consensus Y transfers into X at close to 1:1, so the real translational uncertainty on")
        print(f"  these X values is of the order of the {spread * 1e3:.1f} mm Y spread, not the sub-millimetre")
        print(f"  scatter. The scatter only confirms that a single rigid X explains every pose.")

    if arguments.skip_sensitivity:
        return
    print()
    print("=" * 96)
    print("Sensitivity: mocap treated as noiseless instead  (how far a defensible model change moves X, Y)")
    print("=" * 96)
    for name, (x, y, a, b, fk, sigma_x, sigma_y) in solutions.items():
        x_alt, y_alt = mocap_noiseless(a, b, *fk)
        delta_x = inv_se3(x) @ x_alt
        delta_y = inv_se3(y) @ y_alt
        shift_x = np.linalg.norm(delta_x[:3, 3]) * 1e3
        shift_y = np.linalg.norm(delta_y[:3, 3]) * 1e3
        print(f"  {name:20s} X shift {np.rad2deg(np.linalg.norm(log_so3(delta_x[:3, :3]))):6.3f} deg"
              f" / {shift_x:7.3f} mm  ({shift_x / sigma_x[1]:4.1f}x sigma)"
              f"   Y shift {np.rad2deg(np.linalg.norm(log_so3(delta_y[:3, :3]))):6.3f} deg"
              f" / {shift_y:7.3f} mm  ({shift_y / sigma_y[1]:4.1f}x sigma)")

    print()
    print("Note: most FK error is systematic (kinematic model bias), so folding it into a")
    print("zero-mean covariance downweights A correctly and yields honest uncertainty, but")
    print("leaves bias in X and Y. Removing that bias requires kinematic parameter")
    print("identification, not reweighting.")


if __name__ == "__main__":
    main()
