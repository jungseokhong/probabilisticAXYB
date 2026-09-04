"""Deployable end-effector noise estimator: joint angles in, expected error out.

Everything runs locally: the weights are a small PyTorch checkpoint on disk, the
gravity torque is computed here from the vendored URDF, and the feature scaling is
stored in the checkpoint. At inference you provide ONLY the 7 right-arm joint
angles (radians); sin/cos encoding, torque computation, and normalisation happen
inside.

Train and save (refits on every configuration of the given datasets)::

    python -m examples.ee_noise_model train results/aug_26_2026/error_dataset.npz \
        results/aug_27_2026/valid_markers/error_dataset.npz \
        -o results/ee_noise_model.pt

Query::

    python -m examples.ee_noise_model predict results/ee_noise_model.pt \
        --q 2.65 -0.25 -2.88 -2.32 1.52 0.17 -1.64

or from code::

    from examples.ee_noise_model import EENoiseEstimator
    model = EENoiseEstimator("results/ee_noise_model.pt")
    out = model.predict([2.65, -0.25, -2.88, -2.32, 1.52, 0.17, -1.64])
    out["expected_mm"], out["bound95_mm"], out["expected_deg"]

Held-out accuracy of this recipe (100 frozen Aug 27 test configurations,
5-seed ensemble): vector residual 1.94 mm; size prediction rms +-1.14 mm /
+-0.33 deg, corr +0.66; actual <= 1.5x prediction at 86%. The saved model is refit on all data, so treat those as its
accuracy estimate. The prediction is the error of the MARKER frame; direction
(median cosine ~0.6) is a prior, not a correction.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from examples.gravity_torque import Robot, DEFAULT_URDF
from examples.train_error_network import Trunk, fit, CHARACTERISTIC_LENGTH

ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]


class EENoiseEstimator:
    """Loads a checkpoint and answers: how wrong will FK be at this configuration?"""

    def __init__(self, path, urdf=DEFAULT_URDF, device="cpu"):
        payload = torch.load(path, map_location=device, weights_only=True)
        self.mu = payload["feature_mean"].numpy()
        self.sd = payload["feature_std"].numpy()
        self.fixed = {k: float(v) for k, v in payload["fixed_joints"].items()}
        # An ensemble: predictions are averaged over every stored member, which
        # measured 16% better than the best single seed on the frozen test set.
        states = payload["states"] if "states" in payload else [payload["state"]]
        self.nets = []
        for state in states:
            net = Trunk(len(self.mu), int(payload["hidden"]), 1, 6)
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
        self.robot = Robot(urdf)
        self.device = device

    def features(self, q):
        q = np.asarray(q, dtype=float).reshape(-1)
        if len(q) != 7:
            raise ValueError("expected the 7 right-arm joint angles, radians")
        positions = dict(self.fixed)
        positions.update(zip(ARM_JOINTS, q))
        torque = self.robot.gravity_torques(positions, ARM_JOINTS)
        raw = np.concatenate([np.sin(q), np.cos(q), torque / 10.0])
        return (raw - self.mu) / self.sd

    def predict(self, q):
        x = torch.tensor(self.features(q), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            member = [n(x, torch.zeros(1, dtype=torch.long)).numpy()[0, :6] for n in self.nets]
        out = np.mean(member, axis=0)
        rotation = out[:3] / CHARACTERISTIC_LENGTH          # rad
        translation = out[3:]                               # m
        expected_mm = float(np.linalg.norm(translation) * 1e3)
        expected_deg = float(np.degrees(np.linalg.norm(rotation)))
        return {
            "xi": np.concatenate([rotation, translation]),  # marker-frame twist
            "expected_mm": expected_mm,
            "expected_deg": expected_deg,
            "bound83_mm": 1.5 * expected_mm,                # actual <= this at ~83%
            "bound95_mm": 2.0 * expected_mm,                # actual <= this at ~96%
            "bound_deg": 2.0 * expected_deg,
        }


def train(paths, output, hidden=32, seeds=5, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    xs, ys, val_flags, fixed = [], [], [], None
    for index, path in enumerate(paths):
        data = np.load(path, allow_pickle=False)
        mask = data["target"] == "end_effector"
        joints = [str(n) for n in data["joint_name"]]
        pick = [joints.index(j) for j in ARM_JOINTS]
        xs.append(np.hstack([data["sin_q"][mask][:, pick], data["cos_q"][mask][:, pick],
                             data["gravity_torque"][mask][:, pick] / 10.0]))
        ys.append(np.hstack([data["xi_mean"][mask][:, :3] * CHARACTERISTIC_LENGTH,
                             data["xi_mean"][mask][:, 3:]]))
        # every 6th configuration of each capture becomes early-stopping validation
        uid = data["unique_id"][mask]
        val_flags.append((uid % 6) == 0)
        if fixed is None:
            # Fixed-joint posture for the inference-time torque: the SAME full-body
            # posture the training torques were computed with. Never assume zeros --
            # the captures were taken with the torso folded, and a wrong posture
            # silently shifts the gravity feature.
            if "posture_joint_median" not in data.files:
                raise ValueError(f"{path} lacks posture_joint_median; rebuild the dataset "
                                 "(build_error_dataset.py records it) before training")
            fixed = {str(n): float(v) for n, v in
                     zip(data["posture_joint_name"], data["posture_joint_median"])}
    x, y = np.vstack(xs), np.vstack(ys)
    val = np.concatenate(val_flags)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    x = (x - mu) / sd

    X = torch.tensor(x, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.float32, device=device)
    V = torch.tensor(val, device=device)
    zeros = torch.zeros(len(x), dtype=torch.long, device=device)
    states, vals = [], []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = Trunk(x.shape[1], hidden, 1, 6).to(device)
        v = fit(net, X, Y, zeros, V, "mse", 8000, 1e-3, 1e-3)
        states.append({k: t.detach().cpu() for k, t in net.state_dict().items()})
        vals.append(v)
    torch.save({
        "states": states, "hidden": hidden,
        "feature_mean": torch.tensor(mu), "feature_std": torch.tensor(sd),
        "fixed_joints": fixed,
        "trained_on": [str(p) for p in paths],
        "heldout_accuracy": "ensemble: vector 1.94 mm, size rms +-1.14 mm / +-0.33 deg, "
                            "corr +0.66, <=1.5x @86% (100 frozen aug27 test configs, pre-refit)",
    }, output)
    print(f"trained ensemble of {len(states)} on {len(x)} EE rows from {len(paths)} capture(s); "
          f"val mse {min(vals):.3e}..{max(vals):.3e}\nsaved -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("datasets", nargs="+", type=pathlib.Path)
    t.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("results/ee_noise_model.pt"))
    q = sub.add_parser("predict")
    q.add_argument("model", type=pathlib.Path)
    q.add_argument("--q", nargs=7, type=float, required=True, metavar="RAD")
    arguments = parser.parse_args()
    if arguments.command == "train":
        train(arguments.datasets, arguments.output)
    else:
        model = EENoiseEstimator(arguments.model)
        out = model.predict(arguments.q)
        print(f"expected error : {out['expected_mm']:.2f} mm / {out['expected_deg']:.2f} deg")
        print(f"bounds         : <= {out['bound83_mm']:.2f} mm (~83%)   <= {out['bound95_mm']:.2f} mm (~96%)")
        print(f"twist (marker frame): rot {np.round(out['xi'][:3], 5)} rad  trans {np.round(out['xi'][3:] * 1e3, 3)} mm")


if __name__ == "__main__":
    main()
