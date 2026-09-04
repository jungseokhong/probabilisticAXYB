"""Turn a calibrated AX=YB dataset into supervised forward-kinematics error data.

The label is the calibration residual expressed as a body-frame twist::

    xi = log( (A X)^-1 (Y B) )   in R^6, rotation first then translation

so the calibration is not optional -- ``A`` (forward kinematics) and ``B`` (motion
capture) live in unrelated frames, and ``X`` and ``Y`` are exactly what relates
them. Without them there is no error to measure.

Each configuration's repeats are split into two parts:

``xi_mean``
    the mean over repeats -- the *systematic* error, a deterministic function of
    configuration and the part worth predicting or correcting;
``xi_within_cov``
    the covariance over repeats -- the genuinely random part, which on this
    hardware sits at the motion-capture noise floor and is nearly constant.

Both are written out. Train on the first; keep the second as a diagnostic.

Usage::

    python -m examples.build_error_dataset <dataset directory> [-o out.npz]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from probabilistic_axyb import inv_se3, log_so3

from examples.calibrate_rby1 import (
    MIN_EXCITATION,
    MOCAP_SIGMA_POSITION,
    MOCAP_SIGMA_ROTATION,
    calibrate,
    recover_x_given_y,
    rotation_excitation,
    se3_mean,
)


def twist(delta: np.ndarray) -> np.ndarray:
    """Body-frame 6-vector for a rigid transform: rotation (3) then translation (3)."""
    return np.concatenate((log_so3(delta[:3, :3]), delta[:3, 3]))


def errors(a: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-pose error twists for one target."""
    return np.stack([twist(inv_se3(a[i] @ x) @ (y @ b[i])) for i in range(len(a))])


def gravity_proxy(q: np.ndarray) -> np.ndarray:
    """Cheap per-joint gravitational-load feature.

    Not a dynamics model. ``cos`` of the accumulated angle down the chain tracks how
    much of each joint's downstream mass is cantilevered horizontally, which is the
    quantity that drives deflection. The torso evidence -- 0.018 mm of scatter within
    a configuration against 0.379 mm between them, correlating up to 0.88 with arm
    position -- says load-induced flex is the dominant error mechanism here, so the
    model is given that driver directly rather than being asked to infer it from
    angles alone.
    """
    return np.cos(np.cumsum(q, axis=-1))


class Merged:
    """Several captures of one target, concatenated, behaving like one NPZ.

    Configuration ids are offset by 10000 per capture so they stay unique, and a
    ``capture`` array records which capture each pose came from. Calibrating on the
    merged poses yields ONE X per link and ONE Y for every capture, i.e. a single
    label frame -- valid when markers and mocap have not moved between captures,
    which the cross-capture calibration matrix checks.
    """

    def __init__(self, parts, captures):
        self.files = sorted(set.intersection(*(set(d.files) for d in parts)))
        self._data = {}
        counts = [len(d["A"]) for d in parts]
        for key in self.files:
            arrays = [np.asarray(d[key]) for d in parts]
            if key == "unique_id":
                arrays = [a + 10000 * c for a, c in zip(arrays, captures)]
            per_pose = all(a.ndim >= 1 and a.shape[0] == n for a, n in zip(arrays, counts)) \
                and all(a.shape[1:] == arrays[0].shape[1:] for a in arrays)
            self._data[key] = np.concatenate(arrays) if per_pose else arrays[0]
        self._data["capture"] = np.concatenate(
            [np.full(len(d["A"]), i) for i, d in zip(captures, parts)])
        self.files.append("capture")

    def __getitem__(self, key):
        return self._data[key]


def load_targets(directories):
    """Split targets into those that can determine Y and those that cannot.

    With several directories, each target is the concatenation of its captures.
    """
    directories = list(directories)
    names = sorted(set.intersection(*(
        {p.stem for p in d.glob("*.npz")} for d in directories)))
    solvable, deferred = [], []
    for name in names:
        parts = [np.load(d / f"{name}.npz", allow_pickle=False) for d in directories]
        data = parts[0] if len(parts) == 1 else Merged(parts, list(range(len(parts))))
        record = (name, data)
        (solvable if rotation_excitation(data["A"]) >= MIN_EXCITATION else deferred).append(record)
    return solvable, deferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path, nargs="+",
                        help="one dataset directory, or several captures to calibrate jointly")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("error_dataset.npz"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--calibrate-on", type=int, default=0, metavar="N",
        help="solve X, Y on N randomly chosen configurations per target (0 = all). "
             "Labels are still computed for every pose. Measured on Aug 27: 100 "
             "configurations reproduce the full-data Y to 0.7 mm / 0.04 deg, below "
             "the ~1.5 mm cross-calibration floor, at a fraction of the solve time.",
    )
    parser.add_argument("--calibrate-seed", type=int, default=0)
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="target to drop from the consensus Y (repeatable); it still gets an X via stage 2",
    )
    parser.add_argument(
        "--urdf", type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "models" / "rby1_no_world.urdf",
        help="URDF for gravity-torque features (skipped with a note if absent)",
    )
    arguments = parser.parse_args()

    solvable, deferred = load_targets(arguments.dataset)
    if len(arguments.dataset) > 1:
        print(f"Pooling {len(arguments.dataset)} captures: one X per link, one Y, one label frame")
    if not solvable:
        parser.error("no target has enough excitation to determine Y")

    print("Calibrating (labels are defined by X and Y, so this has to run first)")
    transforms, contributors, per_target_y = {}, [], {}
    fk_side, trust = {}, {}
    subset_rng = np.random.default_rng(arguments.calibrate_seed)
    for name, data in solvable:
        excluded = name in arguments.exclude
        a_cal, b_cal = data["A"], data["B"]
        mask = np.ones(len(data["A"]), bool)
        if arguments.calibrate_on:
            # Calibration saturates well before all poses are used; the solve runs on
            # a random subset of configurations (all repeats of each) and X, Y are
            # then applied to every pose for the labels.
            chosen = subset_rng.permutation(np.unique(data["unique_id"]))[: arguments.calibrate_on]
            mask = np.isin(data["unique_id"], chosen)
            a_cal, b_cal = data["A"][mask], data["B"][mask]
        x, y, _, _, _, result = calibrate(a_cal, b_cal, arguments.iterations, verbose=False, covariance=False)
        transforms[name] = x
        per_target_y[name] = y
        # The FK-attributed half of the split, kept as an alternative label. It is
        # the purer measure of forward-kinematics error, at the cost of depending on
        # the mocap anchor -- see the note printed at the end.
        n_twists = np.full((len(data["A"]), 6), np.nan)   # xi_N exists only where the solver ran
        n_twists[mask] = np.stack([twist(item) for item in result.n_noise])
        fk_side[name] = n_twists
        n_rms = float(np.sqrt((np.linalg.norm(result.n_noise[:, :3, 3], axis=1) ** 2).mean()))
        m_rms = float(np.sqrt((np.linalg.norm(result.m_noise[:, :3, 3], axis=1) ** 2).mean()))
        trust[name] = n_rms / m_rms if m_rms > 0 else float("inf")
        if not excluded:
            contributors.append(y)
        print(f"  {name:20s} X solved on {len(a_cal)} poses{'   [excluded from consensus Y]' if excluded else ''}")
    y_consensus = se3_mean(contributors)
    print(f"  consensus Y translation {np.round(y_consensus[:3, 3] * 1e3, 2)} mm"
          f"  from {len(contributors)} target(s)")

    for name, data in deferred:
        transforms[name], _, _ = recover_x_given_y(data["A"], data["B"], y_consensus)
        print(f"  {name:20s} X recovered against the pinned Y (stage 2)")

    # ---- assemble rows -------------------------------------------------------
    rows = []
    joint_name = None
    for name, data in solvable + deferred:
        a, b = data["A"], data["B"]
        xi = errors(a, b, transforms[name], y_consensus)
        uid = data["unique_id"]
        if "joint_position" in data.files:
            q = data["joint_position"]
            if joint_name is None and "joint_name" in data.files:
                joint_name = data["joint_name"]
        else:
            q = np.full((len(a), 0), np.nan)
        xi_n = fk_side.get(name)
        for u in np.unique(uid):
            index = np.flatnonzero(uid == u)
            group = xi[index]
            group_n = xi_n[index] if xi_n is not None else None
            rows.append({
                "target": name,
                "capture": int(data["capture"][index[0]]) if "capture" in data.files else 0,
                "unique_id": int(u),
                "repeats": len(index),
                "q": q[index[0]] if q.shape[1] else q[0],
                "A_nominal": data["A"][index].mean(axis=0),
                "xi_N_mean": group_n.mean(axis=0) if group_n is not None else np.full(6, np.nan),
                "xi_N_within_cov": (np.cov(group_n, rowvar=False) if group_n is not None
                                    and len(index) > 1 else np.full((6, 6), np.nan)),
                "trust_ratio": trust.get(name, np.nan),
                "xi": group,
                "xi_mean": group.mean(axis=0),
                "xi_within_cov": np.cov(group, rowvar=False) if len(index) > 1 else np.zeros((6, 6)),
            })

    have_joints = bool(rows) and rows[0]["q"].size > 0
    if not have_joints:
        print("\n  NOTE: no 'joint_position' in this dataset. Rebuild it with the patched")
        print("        build_probabilistic_axyb_dataset.py to get joint-space features;")
        print("        A_nominal is written either way so the file stays usable.")

    q_all = np.stack([r["q"] for r in rows]) if have_joints else np.zeros((len(rows), 0))
    gravity_torque = None
    full_joint_names, full_joint_median = None, None
    if have_joints:
        full_joint_names = [str(n) for n in joint_name]
        full_joint_median = np.median(q_all, axis=0)          # q_all is still the full vector here
    if have_joints:
        # URDF-computed static gravity torque per arm joint, from the FULL joint
        # vector (the torso configuration matters even when it is held fixed).
        # This is the physical load driving the deflection; the geometric proxy
        # stays as a fallback for datasets without a URDF.
        try:
            from examples.gravity_torque import torques_for_dataset

            arm = [str(n) for n, s in zip(joint_name, q_all.std(axis=0)) if s > 1e-3]
            gravity_torque = torques_for_dataset(q_all, joint_name, arm, arguments.urdf)
            print(f"\n  Gravity torque computed from {arguments.urdf.name} for: {', '.join(arm)}")
        except (FileNotFoundError, OSError) as error:
            print(f"\n  NOTE: no URDF at {arguments.urdf} ({error}); keeping the geometric proxy only.")
    if have_joints:
        # Keep only joints that actually moved. A joint held fixed for the whole
        # capture contributes a constant column, which carries no information and
        # costs a degree of freedom the fit does not have to spare.
        moving = q_all.std(axis=0) > 1e-3
        if joint_name is not None and len(joint_name) == len(moving):
            kept = [str(n) for n, m in zip(joint_name, moving) if m]
            dropped = len(moving) - len(kept)
            print(f"\n  Joints: {len(kept)} moving, {dropped} held fixed and dropped")
            print(f"    kept: {', '.join(kept)}")
            joint_name = np.asarray(kept)
        q_all = q_all[:, moving]
    payload = {
        "target": np.asarray([r["target"] for r in rows]),
        "capture": np.asarray([r["capture"] for r in rows]),
        "capture_names": np.asarray([str(d) for d in arguments.dataset]),
        "unique_id": np.asarray([r["unique_id"] for r in rows]),
        "repeats": np.asarray([r["repeats"] for r in rows]),
        "q": q_all,
        "sin_q": np.sin(q_all),
        "cos_q": np.cos(q_all),
        "gravity_proxy": gravity_proxy(q_all) if have_joints else q_all,
        **({"gravity_torque": gravity_torque} if gravity_torque is not None else {}),
        # The full-body posture the torques were computed with (median over the
        # capture, every joint). Inference must reproduce this posture for the
        # fixed joints, or the live torque feature drifts from the training one.
        **({"posture_joint_name": np.asarray(full_joint_names),
            "posture_joint_median": full_joint_median} if full_joint_median is not None else {}),
        "A_nominal": np.stack([r["A_nominal"] for r in rows]),
        "xi_mean": np.stack([r["xi_mean"] for r in rows]),
        "xi_within_cov": np.stack([r["xi_within_cov"] for r in rows]),
        "xi_N_mean": np.stack([r["xi_N_mean"] for r in rows]),
        "xi_N_within_cov": np.stack([r["xi_N_within_cov"] for r in rows]),
        "trust_ratio": np.asarray([r["trust_ratio"] for r in rows]),
        "mocap_sigma_position_m": np.asarray(MOCAP_SIGMA_POSITION),
        "mocap_sigma_rotation_rad": np.asarray(MOCAP_SIGMA_ROTATION),
        "Y": y_consensus,
        "X_names": np.asarray(list(transforms)),
        "X": np.stack([transforms[k] for k in transforms]),
    }
    if joint_name is not None:
        payload["joint_name"] = joint_name
    np.savez_compressed(arguments.output, **payload)

    csv_path = arguments.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        # Name the joint columns after the joints themselves. q_all has already had
        # the fixed joints removed, so the rows must be taken from it rather than
        # from the unfiltered per-row vector, or the columns shift out of alignment.
        if joint_name is not None and len(joint_name) == q_all.shape[1]:
            joint_columns = [f"q_{name}" for name in joint_name]
        else:
            joint_columns = [f"q{index}" for index in range(q_all.shape[1])]
        writer.writerow(
            ["target", "capture", "unique_id", "repeats"]
            + [f"xi_mean_{k}" for k in ("wx", "wy", "wz", "px", "py", "pz")]
            + ["xi_mean_rot_deg", "xi_mean_pos_mm", "within_rot_deg", "within_pos_mm"]
            + [f"xi_N_mean_{k}" for k in ("wx", "wy", "wz", "px", "py", "pz")]
            + ["xi_N_mean_rot_deg", "xi_N_mean_pos_mm", "trust_ratio"]
            + joint_columns
        )
        for i, r in enumerate(rows):
            cov = r["xi_within_cov"]
            xn = r["xi_N_mean"]
            writer.writerow(
                [r["target"], r["capture"], r["unique_id"], r["repeats"]]
                + [f"{v:.9g}" for v in r["xi_mean"]]
                + [f"{np.degrees(np.linalg.norm(r['xi_mean'][:3])):.6g}",
                   f"{np.linalg.norm(r['xi_mean'][3:]) * 1e3:.6g}",
                   f"{np.degrees(np.sqrt(max(np.trace(cov[:3, :3]), 0))):.6g}",
                   f"{np.sqrt(max(np.trace(cov[3:, 3:]), 0)) * 1e3:.6g}"]
                + [f"{v:.9g}" for v in xn]
                + [f"{np.degrees(np.linalg.norm(xn[:3])):.6g}",
                   f"{np.linalg.norm(xn[3:]) * 1e3:.6g}",
                   f"{r['trust_ratio']:.4g}"]
                + [f"{v:.9g}" for v in (q_all[i] if have_joints else [])]
            )

    # ---- report --------------------------------------------------------------
    print(f"\nWrote {arguments.output} and {csv_path}"
          f"  ({len(rows)} rows = {len(np.unique(payload['unique_id']))} configurations"
          f" x {len(transforms)} targets)")

    print("\nSystematic vs random, per target  (the whole premise: systematic should dominate)")
    print(f"  {'target':20s} {'systematic (between)':>22} {'random (within)':>18} {'ratio':>7}")
    summary = {}
    for name in transforms:
        mask = payload["target"] == name
        between = np.linalg.norm(payload["xi_mean"][mask][:, 3:], axis=1).std() * 1e3
        within = np.sqrt(np.maximum(
            np.trace(payload["xi_within_cov"][mask][:, 3:, 3:], axis1=1, axis2=2), 0)).mean() * 1e3
        summary[name] = (float(between), float(within))
        ratio = between / within if within > 1e-12 else float("inf")
        print(f"  {name:20s} {between:19.3f} mm {within:15.3f} mm {ratio:7.1f}x")

    print("\nLabel sanity  (RMS |xi| translation, consensus Y against each target's own Y)")
    print(f"  {'target':20s} {'own Y':>9} {'consensus Y':>13}   difference")
    for name, data in solvable:
        own = errors(data["A"], data["B"], transforms[name], per_target_y[name])
        shared = errors(data["A"], data["B"], transforms[name], y_consensus)
        r_own = np.sqrt((np.linalg.norm(own[:, 3:], axis=1) ** 2).mean()) * 1e3
        r_shared = np.sqrt((np.linalg.norm(shared[:, 3:], axis=1) ** 2).mean()) * 1e3
        print(f"  {name:20s} {r_own:6.3f} mm {r_shared:10.3f} mm   {r_shared - r_own:+6.3f}")
    print("  The consensus figure is larger by construction, and that is the point: one")
    print("  shared Y stops per-link error being absorbed into a per-link Y. The gap is")
    print("  the per-link error being exposed -- exactly the signal a model should learn.")

    meta = {
        "note": "labels are defined by these X and Y; re-calibrating invalidates a model trained on them",
        "consensus_Y_translation_mm": [float(v) for v in y_consensus[:3, 3] * 1e3],
        "targets_in_consensus": [n for n, _ in solvable if n not in arguments.exclude],
        "targets_deferred_to_stage_2": [n for n, _ in deferred],
        "configurations": int(len(np.unique(payload["unique_id"]))),
        "has_joint_positions": bool(have_joints),
        "systematic_vs_random_mm": summary,
    }
    meta_path = arguments.output.with_name(arguments.output.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {meta_path}")
    print("\nTwo labels are written. Which to use, per target:")
    print(f"  {'target':20s} {'|N|/|M|':>9}   verdict")
    for name in transforms:
        ratio = trust.get(name)
        if ratio is None or not np.isfinite(ratio):
            print(f"  {name:20s} {'--':>9}   stage 2: no split available, xi_mean only")
            continue
        if ratio >= 5:
            verdict = "FK dominates; xi_mean and xi_N_mean agree"
        elif ratio >= 2:
            verdict = "FK leads but mocap is visible in xi_mean"
        else:
            verdict = "mocap is as large as FK; xi_mean is NOT mostly FK error"
        print(f"  {name:20s} {ratio:9.1f}x   {verdict}")
    print("\n  xi_mean   total residual. Depends only on X and Y, and plugs straight into")
    print("            a correction: base_T_markers = A(q) X exp(xi). Default choice.")
    print("  xi_N_mean the FK-attributed half. Purer, but it moves if you change the")
    print(f"            mocap anchor, which this run set to {MOCAP_SIGMA_POSITION * 1e3:.2f} mm /"
          f" {np.degrees(MOCAP_SIGMA_ROTATION):.2f} deg.")
    print("\nSplit on 'unique_id', never on rows: repeats of one configuration are")
    print("near-duplicates and a row-wise split leaks them across the boundary.")


if __name__ == "__main__":
    main()
