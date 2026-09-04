"""Cross-capture generalisation: train on one capture's labels, predict another's.

The one evaluation no within-dataset split can give. The two captures were taken on
different days and carry *independent calibrations*, so their labels disagree by the
cross-calibration floor (~1-1.5 mm workspace) even for a perfect model -- treat that
floor, not zero, as the best achievable score.

Trains per-target ridge on ALL of the old capture (lambda by grouped 5-fold on the
old data only), standardises features by old-capture statistics, then evaluates once
on every row of the new capture.

Usage::

    python -m examples.cross_capture_test old/error_dataset.npz new/error_dataset.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from examples.fit_error_model import joint_features, ridge as ridge_fit, predict as ridge_predict

CHARACTERISTIC_LENGTH = 0.4


def scaled_labels(data):
    return np.hstack([data["xi_mean"][:, :3] * CHARACTERISTIC_LENGTH, data["xi_mean"][:, 3:]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=pathlib.Path, help="training capture (error_dataset.npz)")
    parser.add_argument("new", type=pathlib.Path, help="evaluation capture (error_dataset.npz)")
    parser.add_argument("--features", default="joint-torque")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=None)
    arguments = parser.parse_args()

    old = np.load(arguments.old, allow_pickle=False)
    new = np.load(arguments.new, allow_pickle=False)

    # Captures may disagree on which joints counted as "moving" (a joint that
    # jittered above threshold in one run and not the other). Align on the
    # intersection by NAME -- positional feature columns would silently mismatch.
    joints_old = [str(n) for n in old["joint_name"]]
    joints_new = [str(n) for n in new["joint_name"]]
    common = [j for j in joints_old if j in joints_new]
    if not common:
        parser.error("no common joints between captures")
    dropped = sorted(set(joints_old) ^ set(joints_new))
    if dropped:
        print(f"aligning on {len(common)} common joints; dropping {dropped}\n")

    def aligned(data, joints):
        index = [joints.index(j) for j in common]
        blocks = [data["sin_q"][:, index], data["cos_q"][:, index]]
        if "torque" in arguments.features and "gravity_torque" in data.files:
            blocks.append(data["gravity_torque"][:, index] / 10.0)
        return np.hstack(blocks)

    x_old, x_new = aligned(old, joints_old), aligned(new, joints_new)
    mu, sd = x_old.mean(0), x_old.std(0) + 1e-9        # old-capture statistics only
    x_old, x_new = (x_old - mu) / sd, (x_new - mu) / sd
    y_old, y_new = scaled_labels(old), scaled_labels(new)

    names = sorted(set(old["target"].tolist()) & set(new["target"].tolist()))
    rng = np.random.default_rng(0)
    report = {}
    print(f"train: {arguments.old} ({len(np.unique(old['unique_id']))} configs)")
    print(f"eval : {arguments.new} ({len(np.unique(new['unique_id']))} configs)\n")
    print(f"  {'target':20s} {'old-mean base':>14} {'model':>9} {'skill':>7} {'mag corr':>9}")
    for name in names:
        m_old, m_new = old["target"] == name, new["target"] == name
        uid = old["unique_id"][m_old]
        folds = rng.permutation(np.unique(uid)) % 5
        fold_of = dict(zip(np.unique(uid), folds))
        group = np.array([fold_of[u] for u in uid])
        best, berr = None, np.inf
        for lam in (0.1, 1.0, 10.0, 100.0):
            errs = []
            for k in range(5):
                hold = group == k
                c = ridge_fit(x_old[m_old][~hold], y_old[m_old][~hold], lam)
                errs.append(((ridge_predict(c, x_old[m_old][hold]) - y_old[m_old][hold]) ** 2).mean())
            if np.mean(errs) < berr:
                berr, best = np.mean(errs), lam
        c = ridge_fit(x_old[m_old], y_old[m_old], best)
        pred = ridge_predict(c, x_new[m_new])

        truth = y_new[m_new]
        base_vec = y_old[m_old].mean(0)                # what the old capture alone would say
        resid = np.sqrt((np.linalg.norm(truth[:, 3:] - pred[:, 3:], axis=1) ** 2).mean()) * 1e3
        base = np.sqrt((np.linalg.norm(truth[:, 3:] - base_vec[3:], axis=1) ** 2).mean()) * 1e3
        mag_t = np.linalg.norm(truth[:, 3:], axis=1) * 1e3
        mag_p = np.linalg.norm(pred[:, 3:], axis=1) * 1e3
        corr = float(np.corrcoef(mag_p, mag_t)[0, 1])
        skill = 1 - (resid / base) ** 2
        report[name] = {"base_mm": float(base), "model_mm": float(resid),
                        "skill": float(skill), "mag_corr": corr, "lambda": best}
        print(f"  {name:20s} {base:11.3f} mm {resid:6.3f} mm {skill:+7.2f} {corr:+9.2f}")

    mean_skill = float(np.mean([r["skill"] for r in report.values()]))
    print(f"\n  mean skill {mean_skill:+.2f}   (score against the ~1-1.5 mm cross-calibration floor,")
    print("                              not against zero)")
    out = arguments.output or arguments.new.parent / f"cross_capture_{arguments.features}.json"
    out.write_text(json.dumps({"train": str(arguments.old), "eval": str(arguments.new),
                               "features": arguments.features, "mean_skill": mean_skill,
                               "targets": report}, indent=1))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
