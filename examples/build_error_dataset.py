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


def load_targets(directory: pathlib.Path):
    """Split targets into those that can determine Y and those that cannot."""
    solvable, deferred = [], []
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        record = (path.stem, data)
        (solvable if rotation_excitation(data["A"]) >= MIN_EXCITATION else deferred).append(record)
    return solvable, deferred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("error_dataset.npz"))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--exclude", action="append", default=[],
        help="target to drop from the consensus Y (repeatable); it still gets an X via stage 2",
    )
    arguments = parser.parse_args()

    solvable, deferred = load_targets(arguments.dataset)
    if not solvable:
        parser.error("no target has enough excitation to determine Y")

    print("Calibrating (labels are defined by X and Y, so this has to run first)")
    transforms, contributors, per_target_y = {}, [], {}
    for name, data in solvable:
        excluded = name in arguments.exclude
        x, y, _, _, _, _ = calibrate(data["A"], data["B"], arguments.iterations, verbose=False)
        transforms[name] = x
        per_target_y[name] = y
        if not excluded:
            contributors.append(y)
        print(f"  {name:20s} X solved{'   [excluded from consensus Y]' if excluded else ''}")
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
        for u in np.unique(uid):
            index = np.flatnonzero(uid == u)
            group = xi[index]
            rows.append({
                "target": name,
                "unique_id": int(u),
                "repeats": len(index),
                "q": q[index[0]] if q.shape[1] else q[0],
                "A_nominal": data["A"][index].mean(axis=0),
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
    payload = {
        "target": np.asarray([r["target"] for r in rows]),
        "unique_id": np.asarray([r["unique_id"] for r in rows]),
        "repeats": np.asarray([r["repeats"] for r in rows]),
        "q": q_all,
        "sin_q": np.sin(q_all),
        "cos_q": np.cos(q_all),
        "gravity_proxy": gravity_proxy(q_all) if have_joints else q_all,
        "A_nominal": np.stack([r["A_nominal"] for r in rows]),
        "xi_mean": np.stack([r["xi_mean"] for r in rows]),
        "xi_within_cov": np.stack([r["xi_within_cov"] for r in rows]),
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
        writer.writerow(
            ["target", "unique_id", "repeats"]
            + [f"xi_mean_{k}" for k in ("wx", "wy", "wz", "px", "py", "pz")]
            + ["xi_mean_rot_deg", "xi_mean_pos_mm", "within_rot_deg", "within_pos_mm"]
            + [f"q{i}" for i in range(q_all.shape[1])]
        )
        for i, r in enumerate(rows):
            cov = r["xi_within_cov"]
            writer.writerow(
                [r["target"], r["unique_id"], r["repeats"]]
                + [f"{v:.9g}" for v in r["xi_mean"]]
                + [f"{np.degrees(np.linalg.norm(r['xi_mean'][:3])):.6g}",
                   f"{np.linalg.norm(r['xi_mean'][3:]) * 1e3:.6g}",
                   f"{np.degrees(np.sqrt(max(np.trace(cov[:3, :3]), 0))):.6g}",
                   f"{np.sqrt(max(np.trace(cov[3:, 3:]), 0)) * 1e3:.6g}"]
                + [f"{v:.9g}" for v in (r["q"] if have_joints else [])]
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
    print("Split on 'unique_id', never on rows: repeats of one configuration are")
    print("near-duplicates and a row-wise split leaks them across the boundary.")


if __name__ == "__main__":
    main()
