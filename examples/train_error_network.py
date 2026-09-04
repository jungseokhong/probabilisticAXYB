"""Train and honestly evaluate small networks for configuration-conditioned FK error.

PyTorch models behind the same frozen protocol for every candidate:

- the TEST set is 20 configurations frozen in ``eval_split.json``, never touched by
  training, early stopping, or model selection;
- 16 VALIDATION configurations drive early stopping and hyperparameter choice;
- every model, including the ridge baseline, sees exactly the same split.

Models:

``constant``    per-target mean of the training configurations (null hypothesis)
``ridge``       linear baseline (the standing model, lambda picked on val)
``per-target``  independent small MLP per link (no sharing)
``shared``      one trunk, one linear head per link (multi-task; plan option A5)
``hetero``      shared trunk, heads emit mean and log-sigma; MSE warm-up then
                beta-NLL -- per-configuration, per-axis sigma

Units: targets are scaled to metre-equivalents, ``z = [L w, p]`` with L = 0.4 m, so
rotation and translation trade off physically in one loss. Reported metrics are mm
and degrees.

Usage::

    python -m examples.train_error_network results/aug_26_2026/error_dataset.npz
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

from examples.fit_error_model import joint_features, ridge as ridge_fit, predict as ridge_predict

CHARACTERISTIC_LENGTH = 0.4          # metres per radian: unifies rotation and translation
SIGMA_FLOOR = 1e-4                   # metre-equivalent; ~0.1 mm / ~0.014 deg
LOG_FLOOR = float(np.log(SIGMA_FLOOR))


class Trunk(nn.Module):
    """Two SiLU layers and one linear head per target; heads may emit mean+log-sigma."""

    def __init__(self, n_in, hidden, n_heads, n_out):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden, n_out) for _ in range(n_heads))

    def forward(self, x, head_index):
        features = self.body(x)
        out = torch.empty(len(x), self.heads[0].out_features, device=x.device)
        for h, head in enumerate(self.heads):
            rows = head_index == h
            if rows.any():
                out[rows] = head(features[rows])
        return out


def beta_nll(mu, log_sigma, y, beta=0.5):
    log_sigma = log_sigma.clamp(LOG_FLOOR, 3.0)
    inv_var = torch.exp(-2 * log_sigma)
    weight = torch.exp(2 * beta * log_sigma).detach()      # stops sigma excusing bad means
    return (weight * (0.5 * (mu - y) ** 2 * inv_var + log_sigma)).mean()


def fit(model, x, y, heads, val_rows, kind, epochs, lr, weight_decay, warmup=800):
    train_rows = ~val_rows
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, patience = np.inf, None, 0
    n_out = y.shape[1]
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad()
        out = model(x[train_rows], heads[train_rows])
        if kind == "hetero" and epoch >= warmup:
            loss = beta_nll(out[:, :n_out], out[:, n_out:], y[train_rows])
        else:
            loss = ((out[:, :n_out] - y[train_rows]) ** 2).mean()
        loss.backward()
        optimiser.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                v = model(x[val_rows], heads[val_rows])
                val = float(((v[:, :n_out] - y[val_rows]) ** 2).mean())
            if np.isfinite(val) and val < best - 1e-12:
                best, patience = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience > 40 and epoch > warmup:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def metrics(y_true, y_pred, targets, names, sigma=None):
    """Physical-unit test metrics; inputs are metre-equivalent 6-vectors (numpy)."""
    L = CHARACTERISTIC_LENGTH
    rows = {}
    for t in names:
        m = targets == t
        pos_t = np.linalg.norm(y_true[m][:, 3:], axis=1) * 1e3
        pos_p = np.linalg.norm(y_pred[m][:, 3:], axis=1) * 1e3
        rows[t] = {
            "pos_resid_mm": float(np.sqrt((np.linalg.norm(
                y_true[m][:, 3:] - y_pred[m][:, 3:], axis=1) ** 2).mean()) * 1e3),
            "rot_resid_deg": float(np.degrees(np.sqrt((np.linalg.norm(
                (y_true[m][:, :3] - y_pred[m][:, :3]) / L, axis=1) ** 2).mean()))),
            "mag_corr": float(np.corrcoef(pos_p, pos_t)[0, 1]) if len(pos_t) > 2 else np.nan,
            "cover15": float((pos_t <= 1.5 * np.maximum(pos_p, 0.1)).mean()),
        }
    if sigma is not None:
        z = np.abs(y_true - y_pred) / np.maximum(sigma, SIGMA_FLOOR)
        rows["_sigma"] = {"within_1s": float((z <= 1).mean()), "within_2s": float((z <= 2).mean())}
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=pathlib.Path)
    parser.add_argument("--split", type=pathlib.Path, default=None)
    parser.add_argument("--features", default="joint-torque")
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cpu", action="store_true")
    arguments = parser.parse_args()

    device = "cpu" if arguments.cpu or not torch.cuda.is_available() else "cuda"
    data = np.load(arguments.dataset, allow_pickle=False)
    split_path = arguments.split or arguments.dataset.parent / "eval_split.json"
    split = json.loads(split_path.read_text())
    uid = data["unique_id"]
    in_test = np.isin(uid, split["test"])
    in_val = np.isin(uid, split["val"])
    in_train = np.isin(uid, split["train"])

    x_np = joint_features(data, np.ones(len(uid), bool), arguments.features)
    mu_f, sd_f = x_np[in_train].mean(0), x_np[in_train].std(0) + 1e-9   # train-only scaling
    x_np = (x_np - mu_f) / sd_f
    L = CHARACTERISTIC_LENGTH
    y_np = np.hstack([data["xi_mean"][:, :3] * L, data["xi_mean"][:, 3:]])

    names = sorted(set(data["target"].tolist()))
    heads_np = np.array([names.index(t) for t in data["target"]])

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    heads = torch.tensor(heads_np, device=device)
    fit_rows = torch.tensor(in_train | in_val, device=device)
    val_local = torch.tensor(in_val[in_train | in_val], device=device)

    print(f"device {device} | features {arguments.features} ({x_np.shape[1]} dims) | "
          f"configs {len(split['train'])}/{len(split['val'])}/{len(split['test'])} train/val/test "
          f"({in_train.sum()}/{in_val.sum()}/{in_test.sum()} rows)\n")

    results = {}
    tgt_test = data["target"][in_test]

    # constant: per-target mean of train configurations
    pred = np.zeros_like(y_np)
    for h, n in enumerate(names):
        pred[heads_np == h] = y_np[(heads_np == h) & in_train].mean(0)
    results["constant"] = metrics(y_np[in_test], pred[in_test], tgt_test, names)

    # ridge: lambda on val, refit on train+val
    pred = np.zeros_like(y_np)
    for h, n in enumerate(names):
        tr, va = (heads_np == h) & in_train, (heads_np == h) & in_val
        best, berr = None, np.inf
        for lam in (0.01, 0.1, 1.0, 10.0, 100.0):
            c = ridge_fit(x_np[tr], y_np[tr], lam)
            e = float(((ridge_predict(c, x_np[va]) - y_np[va]) ** 2).mean())
            if e < berr:
                berr, best = e, lam
        c = ridge_fit(x_np[tr | va], y_np[tr | va], best)
        pred[heads_np == h] = ridge_predict(c, x_np[heads_np == h])
    results["ridge"] = metrics(y_np[in_test], pred[in_test], tgt_test, names)

    def select(builder, kind, label):
        best_val, keep = np.inf, None
        for seed in range(arguments.seeds):
            torch.manual_seed(seed)
            model = builder().to(device)
            v = fit(model, x[fit_rows], y[fit_rows], heads[fit_rows], val_local, kind,
                    arguments.epochs, arguments.lr, arguments.weight_decay)
            if v < best_val:
                model.eval()
                with torch.no_grad():
                    out = model(x, heads).cpu().numpy()
                best_val, keep = v, out
        sigma = None
        if keep.shape[1] == 12:
            sigma = np.exp(np.clip(keep[:, 6:], LOG_FLOOR, 3.0))[in_test]
        results[label] = metrics(y_np[in_test], keep[in_test, :6], tgt_test, names, sigma)

    select(lambda: Trunk(x_np.shape[1], arguments.hidden, len(names), 6), "mse", "shared")
    select(lambda: Trunk(x_np.shape[1], arguments.hidden, len(names), 12), "hetero", "hetero")

    # per-target: independent nets
    pred = np.zeros_like(y_np)
    for h, n in enumerate(names):
        rows_np = heads_np == h
        rows = torch.tensor(rows_np & (in_train | in_val), device=device)
        vloc = torch.tensor(in_val[rows_np & (in_train | in_val)], device=device)
        zeros = torch.zeros(int(rows.sum()), dtype=torch.long, device=device)
        best_val, keep = np.inf, None
        for seed in range(arguments.seeds):
            torch.manual_seed(seed)
            model = Trunk(x_np.shape[1], 32, 1, 6).to(device)
            v = fit(model, x[rows], y[rows], zeros, vloc, "mse",
                    arguments.epochs, arguments.lr, arguments.weight_decay)
            if v < best_val:
                model.eval()
                with torch.no_grad():
                    keep = model(x[torch.tensor(rows_np, device=device)],
                                 torch.zeros(int(rows_np.sum()), dtype=torch.long, device=device)).cpu().numpy()
                best_val = v
        pred[rows_np] = keep[:, :6]
    results["per-target"] = metrics(y_np[in_test], pred[in_test], tgt_test, names)

    # report ------------------------------------------------------------------
    print(f"TEST-SET results ({len(split['test'])} configurations nothing ever saw)\n")
    print(f"  {'model':12s} {'EE pos resid':>13} {'EE rot resid':>13} {'EE mag corr':>12} "
          f"{'EE cover@1.5x':>14} {'torso resid':>12}")
    for label in ("constant", "ridge", "per-target", "shared", "hetero"):
        r = results[label]
        ee, torso = r["end_effector"], r["link_torso_5"]
        print(f"  {label:12s} {ee['pos_resid_mm']:10.3f} mm {ee['rot_resid_deg']:10.3f}° "
              f"{ee['mag_corr']:12.2f} {100*ee['cover15']:12.0f} % {torso['pos_resid_mm']:9.3f} mm")
    print()
    print(f"  {'model':12s} {'mean pos resid':>15} {'mean skill vs constant':>24}")
    const = results["constant"]
    for label in ("ridge", "per-target", "shared", "hetero"):
        r = results[label]
        resids = [r[n]["pos_resid_mm"] for n in names]
        skills = [1 - (r[n]["pos_resid_mm"] / const[n]["pos_resid_mm"]) ** 2 for n in names]
        print(f"  {label:12s} {np.mean(resids):12.3f} mm {np.mean(skills):+24.2f}")
    if "_sigma" in results["hetero"]:
        s = results["hetero"]["_sigma"]
        print(f"\n  hetero sigma calibration: {100*s['within_1s']:.0f}% within 1σ (target ~68), "
              f"{100*s['within_2s']:.0f}% within 2σ (target ~95)")

    out_path = arguments.dataset.parent / f"network_results_{arguments.features}.json"
    out_path.write_text(json.dumps({m: {t: ({k: float(v) for k, v in r.items()} if isinstance(r, dict) else r)
                                        for t, r in res.items()} for m, res in results.items()}, indent=1))
    print(f"\nfull per-target metrics -> {out_path}")


if __name__ == "__main__":
    main()
