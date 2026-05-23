"""Phase 4: train and evaluate the two-stage multi-user cluster controller.

Architecture (handles variable N by construction):
  global head : cluster aggregate features -> (mu*, chi*, beta_A*)
  per-user shared head : (bob features + Alice ctx + predicted mu, chi) -> beta_i*

Closed-loop evaluation reconstructs each held-out cluster from its user rows,
chains the two heads to produce (mu, beta_A, {beta_i}, chi), and evaluates the
analytic cluster_key_rate; reports feasible-gated achieved key vs the oracle
rate stored in the dataset.

Compares KAN (pykan) vs regularised MLP vs Linear on the multi-axis trade-off,
mean+/-std over seeds. Run on the server for the KAN row.

Usage:
  python scripts/train_cluster.py --models mlp,linear --seeds 3 --quick   # local
  python scripts/train_cluster.py --models kan,mlp,linear --seeds 5 --device cuda
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import (SystemParams, make_expectation, NodeState, cluster_key_rate,  # noqa
                    build_link_state)

RESULTS = Path(__file__).resolve().parents[1] / "results"
G_FEATS = ["gA", "sA", "wA", "N", "g_mean", "g_min", "s_mean", "d_eve"]
G_LABELS = ["mu", "chi", "beta_A"]
U_FEATS = ["g_i", "s_i", "w_i", "gA", "sA", "mu", "chi"]
U_LABELS = ["beta_i"]
MU_LO, MU_HI = 0.05, 0.95
CHI_LO, CHI_HI = 0.0, 1.0
BETA_LO, BETA_HI = 0.3, 4.5


def _read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return [dict((k, float(v)) for k, v in r.items()) for r in csv.DictReader(f)]


def standardize(X, ref=None):
    if ref is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1
        return (X - mu) / sd, (mu, sd)
    mu, sd = ref
    return (X - mu) / sd


# ---- models (compact regressors; mirrors train_kan.py) ----
class LinearModel:
    name = "Linear"; interpretable = "closed-form"
    def fit(self, X, y, seed=0):
        A = np.hstack([X, np.ones((X.shape[0], 1))])
        self.W, *_ = np.linalg.lstsq(A, y, rcond=None)
    def predict(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))]) @ self.W
    @property
    def n_params(self): return int(self.W.size)


class MLPModel:
    name = "MLP"; interpretable = "no"
    def __init__(self, hidden=32, epochs=1500, lr=1e-2, wd=1e-3, device="cpu",
                 zero_init_output=False):
        self.h, self.E, self.lr, self.wd, self.dev = hidden, epochs, lr, wd, device
        self.zero_init_output = zero_init_output
    def fit(self, X, y, seed=0):
        import torch
        torch.manual_seed(seed); self.t = torch
        d_in, d_out = X.shape[1], y.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, self.h), torch.nn.SiLU(),
            torch.nn.Linear(self.h, self.h), torch.nn.SiLU(),
            torch.nn.Linear(self.h, d_out)).to(self.dev)
        if self.zero_init_output:
            # for residual learning: start with predict==0 so initial total==base
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)
        n = X.shape[0]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).numpy()
        nv = max(1, int(0.2 * n)); vi, ti = perm[:nv], perm[nv:]
        Xt = torch.tensor(X, dtype=torch.float32, device=self.dev)
        Yt = torch.tensor(y, dtype=torch.float32, device=self.dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        loss = torch.nn.MSELoss(); best, state, bad = float("inf"), None, 0
        for _ in range(self.E):
            self.net.train(); opt.zero_grad()
            l = loss(self.net(Xt[ti]), Yt[ti]); l.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = loss(self.net(Xt[vi]), Yt[vi]).item()
            if v < best - 1e-6:
                best, state, bad = v, {k: p.detach().clone() for k, p in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 250: break
        if state is not None: self.net.load_state_dict(state)
    def predict(self, X):
        Xt = self.t.tensor(X, dtype=self.t.float32, device=self.dev)
        with self.t.no_grad(): return self.net(Xt).cpu().numpy()
    @property
    def n_params(self): return sum(p.numel() for p in self.net.parameters())


class KANModel:
    name = "KAN"; interpretable = "yes"
    def __init__(self, hidden=3, steps=40, device="cpu", lamb=0.001):
        self.h, self.s, self.dev, self.lamb = hidden, steps, device, lamb
    def fit(self, X, y, seed=0):
        import torch
        from kan import KAN
        self.t = torch; torch.manual_seed(seed)
        d_in, d_out = X.shape[1], y.shape[1]
        self.y_med = float(np.median(y))     # NaN fallback target
        self.m = KAN(width=[d_in, self.h, d_out], grid=5, k=3, seed=seed, device=self.dev)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.dev)
        Yt = torch.tensor(y, dtype=torch.float32, device=self.dev)
        # lamb adds L1+entropy regularisation in pykan; helps LBFGS stability
        self.m.fit({"train_input": Xt, "train_label": Yt,
                    "test_input": Xt, "test_label": Yt},
                   opt="LBFGS", steps=self.s, lamb=self.lamb)
    def predict(self, X):
        Xt = self.t.tensor(X, dtype=self.t.float32, device=self.dev)
        with self.t.no_grad():
            out = self.m(Xt).cpu().numpy()
        # safety: replace any NaN/Inf with the training median (defensive)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = self.y_med
        return out
    @property
    def n_params(self): return sum(p.numel() for p in self.m.parameters())


class ResidualModel:
    """Train ``top`` on the residual y - base.predict(X), with base = LinearModel.
    Predict = base + top. Combines Linear's smoothing bias (robust to the sharp
    cluster feasibility boundary) with the top model's nonlinear flexibility."""
    def __init__(self, top):
        self.top = top
        self.base = LinearModel()
        self.name = top.name + "+resid"
        self.interpretable = f"{top.interpretable} (residual over Linear)"
    def fit(self, X, y, seed=0):
        self.base.fit(X, y, seed=seed)
        self.top.fit(X, y - self.base.predict(X), seed=seed)
    def predict(self, X):
        return self.base.predict(X) + self.top.predict(X)
    @property
    def n_params(self):
        return self.base.n_params + self.top.n_params


def make(key, seed, device, epochs, residual=False):
    if key == "linear":
        return LinearModel()
    if key == "mlp":
        base = MLPModel(epochs=epochs, device=device, zero_init_output=residual)
    elif key == "kan":
        base = KANModel(device=device)
    else:
        raise KeyError(key)
    return ResidualModel(base) if residual else base


def clip_global(y):
    y = y.copy()
    y[:, 0] = np.clip(y[:, 0], MU_LO, MU_HI)
    y[:, 1] = np.clip(y[:, 1], CHI_LO, CHI_HI)
    y[:, 2] = np.clip(y[:, 2], BETA_LO, BETA_HI)
    return y


def closed_loop_cluster(gmodel, umodel, g_test_X, g_test_raw, urows_by_cid,
                        g_scale, u_scale, expect, p, beta_margin: float = 0.0):
    """For each test cluster: predict global -> assemble user features ->
    predict per-user beta_i (+ optional safety margin on beta_A and beta_i) ->
    evaluate cluster_key_rate. Returns achieved key sum, oracle sum, secure
    pair count vs oracle. The margin recipe (Phase 3) is essential for cluster
    controllers because cluster feasibility couples 4 predicted variables."""
    Xn = standardize(g_test_X, g_scale)
    pred_g = clip_global(gmodel.predict(Xn))
    achieved = 0.0; oracle = 0.0; sec_pred = sec_oracle = 0
    for i, gr in enumerate(g_test_raw):
        cid = int(gr["cluster_id"])
        mu_p, chi_p, bA_p = pred_g[i]
        bA_p = float(np.clip(bA_p + beta_margin, BETA_LO, BETA_HI))
        urs = urows_by_cid[cid]
        alice = NodeState(gr["gA"], gr["sA"], gr["wA"])
        bobs = [NodeState(u["g_i"], u["s_i"], u["w_i"]) for u in urs]
        UX_raw = np.array([[u["g_i"], u["s_i"], u["w_i"], gr["gA"], gr["sA"],
                            mu_p, chi_p] for u in urs])
        UX = standardize(UX_raw, u_scale)
        bi = np.clip(umodel.predict(UX)[:, 0] + beta_margin,
                     BETA_LO, BETA_HI).tolist()
        d_eve = float(urs[0]["d_eve"])
        res = cluster_key_rate(float(mu_p), float(bA_p), bi, float(chi_p),
                               alice, bobs, expect, d_eve=d_eve)
        ach = sum(r for r, f in zip(res["per_rate"], res["per_feasible"]) if f)
        achieved += ach; oracle += gr["rate_opt"]
        sec_pred += res["n_feasible"]; sec_oracle += int(gr["nfeas_opt"])
    return achieved, oracle, sec_pred, sec_oracle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,linear")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--margin-sweep", action="store_true",
                    help="sweep a beta safety margin (applied to beta_A and beta_i) "
                         "and report key retention vs delta")
    ap.add_argument("--residual", action="store_true",
                    help="train MLP/KAN to predict residuals over a Linear base "
                         "(combines Linear's robustness with ML's flexibility)")
    args = ap.parse_args()

    g_rows = _read_csv(RESULTS / "cluster_global.csv")
    u_rows = _read_csv(RESULTS / "cluster_user.csv")
    if not g_rows:
        print("ERROR: empty cluster_global.csv -- run generate_dataset_cluster.py first")
        return 1
    epochs = 300 if args.quick else 1500
    p = SystemParams(); expect = make_expectation(p.gh_order)

    # split clusters
    cids = sorted({int(r["cluster_id"]) for r in g_rows})
    rng = np.random.default_rng(0); rng.shuffle(cids)
    nte = max(1, int(0.2 * len(cids)))
    te_ids, tr_ids = set(cids[:nte]), set(cids[nte:])

    g_tr = [r for r in g_rows if int(r["cluster_id"]) in tr_ids]
    g_te = [r for r in g_rows if int(r["cluster_id"]) in te_ids]
    u_tr = [r for r in u_rows if int(r["cluster_id"]) in tr_ids]
    u_te = [r for r in u_rows if int(r["cluster_id"]) in te_ids]

    G_Xtr = np.array([[r[k] for k in G_FEATS] for r in g_tr])
    G_ytr = np.array([[r[k] for k in G_LABELS] for r in g_tr])
    G_Xte = np.array([[r[k] for k in G_FEATS] for r in g_te])
    G_yte = np.array([[r[k] for k in G_LABELS] for r in g_te])
    U_Xtr = np.array([[r[k] for k in U_FEATS] for r in u_tr])
    U_ytr = np.array([[r[k] for k in U_LABELS] for r in u_tr])
    U_Xte = np.array([[r[k] for k in U_FEATS] for r in u_te])
    U_yte = np.array([[r[k] for k in U_LABELS] for r in u_te])
    print(f"train clusters={len(g_tr)} test clusters={len(g_te)} "
          f"train user rows={len(u_tr)} test user rows={len(u_te)}")

    G_Xtr_n, g_scale = standardize(G_Xtr)
    U_Xtr_n, u_scale = standardize(U_Xtr)
    urows_by_cid = {}
    for r in u_te:
        urows_by_cid.setdefault(int(r["cluster_id"]), []).append(r)

    lines = ["Phase 4: multi-user cluster controller (two-stage, multi-axis trade-off)",
             "=" * 84,
             f"clusters tr/te = {len(g_tr)}/{len(g_te)};  seeds={args.seeds}"]
    hdr = (f"{'model':<8}{'MAE_mu':>13}{'MAE_chi':>13}{'MAE_bA':>13}"
           f"{'MAE_bi':>13}{'keyRet':>13}{'secP/O':>13}{'params':>9}")
    lines.append(""); lines.append(hdr); lines.append("-" * 84)

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = list(range(max(1, args.seeds)))
    seed0_models = {}     # for the margin sweep
    for key in wanted:
        runs = []
        for sd in seeds:
            try:
                gm = make(key, sd, args.device, epochs, residual=args.residual)
                um = make(key, sd, args.device, epochs, residual=args.residual)
                gm.fit(G_Xtr_n, G_ytr, seed=sd)
                um.fit(U_Xtr_n, U_ytr, seed=sd)
            except Exception as exc:
                print(f"[skip {key} seed {sd}: {exc}]"); continue
            if sd == 0:
                seed0_models[key] = (gm, um)
            # eval
            pg = clip_global(gm.predict(standardize(G_Xte, g_scale)))
            pu = np.clip(um.predict(standardize(U_Xte, u_scale)), BETA_LO, BETA_HI)
            mae_mu = float(np.abs(pg[:, 0] - G_yte[:, 0]).mean())
            mae_chi = float(np.abs(pg[:, 1] - G_yte[:, 1]).mean())
            mae_bA = float(np.abs(pg[:, 2] - G_yte[:, 2]).mean())
            mae_bi = float(np.abs(pu[:, 0] - U_yte[:, 0]).mean())
            ach, ora, sp, so = closed_loop_cluster(
                gm, um, G_Xte, g_te, urows_by_cid, g_scale, u_scale, expect, p)
            ret = ach / ora if ora > 0 else float("nan")
            runs.append((mae_mu, mae_chi, mae_bA, mae_bi, ret, sp, so,
                         gm.n_params + um.n_params))
        if not runs:
            continue
        arr = np.array([r[:8] for r in runs], float)
        m, s = arr.mean(0), arr.std(0)

        def pm(i, p=3):
            return f"{m[i]:.{p}f}+-{s[i]:.{p}f}" if len(seeds) > 1 else f"{m[i]:.{p}f}"
        sp_str = f"{int(m[5])}/{int(m[6])}"
        lines.append(f"{key:<8}{pm(0):>13}{pm(1):>13}{pm(2):>13}{pm(3):>13}"
                     f"{pm(4):>13}{sp_str:>13}{int(m[7]):>9}")

    # ---- beta safety-margin sweep on seed-0 models (essential for clusters) ----
    if args.margin_sweep and seed0_models:
        deltas = np.round(np.concatenate([
            np.arange(0.0, 0.305, 0.025), np.arange(0.4, 1.05, 0.2)]), 3)
        lines.append("")
        lines.append("beta safety-margin sweep (deploy beta_A+delta and beta_i+delta):")
        lines.append("  " + f"{'delta':>6}" + "".join(f"{k:>11}" for k in seed0_models))
        curves = {k: [] for k in seed0_models}
        for d in deltas:
            row = f"  {d:>6.2f}"
            for k, (gm, um) in seed0_models.items():
                ach, ora, sp, so = closed_loop_cluster(
                    gm, um, G_Xte, g_te, urows_by_cid, g_scale, u_scale, expect, p,
                    beta_margin=float(d))
                ret = ach / ora if ora > 0 else float("nan")
                curves[k].append((d, ret, sp, so))
                row += f"{ret:>11.3f}"
            lines.append(row)
        lines.append("  best margin per model:")
        for k in seed0_models:
            base = curves[k][0][1]
            d_star, ret_star, sp_star, so_star = max(curves[k], key=lambda t: t[1])
            gain = ret_star / base if base > 0 else float("inf")
            lines.append(f"    {k:<8} delta*={d_star:.2f}  keyRet*={ret_star:.3f}  "
                         f"secP={sp_star}/{so_star}  (delta=0 -> {base:.3f}, "
                         f"gain {gain:.2f}x)")
        with open(RESULTS / "cluster_margin_sweep.csv", "w", encoding="utf-8") as f:
            f.write("delta," + ",".join(f"{k}_keyret,{k}_secP" for k in seed0_models)
                    + "\n")
            for i, d in enumerate(deltas):
                f.write(f"{d:.3f}," + ",".join(
                    f"{curves[k][i][1]:.6g},{curves[k][i][2]}" for k in seed0_models)
                    + "\n")

    txt = "\n".join(lines)
    (RESULTS / "cluster_controller.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
