"""Phase 3: learn the parameter controller (zenith/gamma0/sigma_X/w_eq/d_eve)
-> (mu*, beta*) and compare KAN against MLP / KNN-lookup / Linear baselines on a
MULTI-AXIS trade-off, NOT just regression error.

Axes reported per model:
  * accuracy        : MAE(mu), MAE(beta), R2
  * closed-loop     : key retention = sum achieved r_norm / sum oracle r_norm on
                      held-out feasible states (the metric that actually matters),
                      plus secure-step retention
  * model size      : # parameters (or stored samples for KNN)
  * inference speed  : microseconds per prediction (relevant for onboard use)
  * interpretability: KAN/Linear yield closed-form rules; MLP/KNN do not

The point is the trade-off: a model may win on MAE yet lose on key retention,
size, or interpretability. We report all axes and let the analysis stand.

Usage:
  python scripts/train_kan.py --models mlp,knn,linear --quick   # CPU smoke (no pykan)
  python scripts/train_kan.py --models kan,mlp,knn,linear        # full (server, GPU)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import make_expectation, SystemParams  # noqa: E402
from satqkd.link import LinkState  # noqa: E402
from satqkd.optimize import evaluate  # noqa: E402
from satqkd.mldata import load_dataset, mae, r2_score  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
MU_LO, MU_HI = 0.05, 0.95
BETA_LO, BETA_HI = 0.3, 4.5
D_EVE_COL = 3  # index of d_eve in FEATURES


def clip_labels(y):
    y = y.copy()
    y[:, 0] = np.clip(y[:, 0], MU_LO, MU_HI)
    y[:, 1] = np.clip(y[:, 1], BETA_LO, BETA_HI)
    return y


# ----------------------------- models -----------------------------
class LinearModel:
    name = "Linear"
    interpretable = "closed-form (affine)"

    def fit(self, Xn, y):
        A = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        self.W, *_ = np.linalg.lstsq(A, y, rcond=None)

    def predict(self, Xn):
        A = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
        return A @ self.W

    @property
    def n_params(self):
        return int(self.W.size)


class KNNModel:
    name = "KNN-lookup"
    interpretable = "no (table)"

    def __init__(self, k=5):
        self.k = k

    def fit(self, Xn, y):
        self.Xn, self.y = Xn, y

    def predict(self, Xn):
        out = np.empty((Xn.shape[0], self.y.shape[1]))
        for i, x in enumerate(Xn):
            d = ((self.Xn - x) ** 2).sum(axis=1)
            nn = np.argsort(d)[: self.k]
            out[i] = self.y[nn].mean(axis=0)
        return out

    @property
    def n_params(self):
        return int(self.Xn.size + self.y.size)  # stored memory


class MLPModel:
    name = "MLP"
    interpretable = "no (black box)"

    def __init__(self, hidden=32, epochs=2000, lr=1e-2, device="cpu", seed=0,
                 weight_decay=1e-3, val_frac=0.2, patience=300):
        self.hidden, self.epochs, self.lr, self.device, self.seed = \
            hidden, epochs, lr, device, seed
        self.weight_decay, self.val_frac, self.patience = weight_decay, val_frac, patience

    def fit(self, Xn, y):
        import torch
        torch.manual_seed(self.seed)
        self.t = torch
        d_in, d_out = Xn.shape[1], y.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, self.hidden), torch.nn.SiLU(),
            torch.nn.Linear(self.hidden, self.hidden), torch.nn.SiLU(),
            torch.nn.Linear(self.hidden, d_out),
        ).to(self.device)
        # internal train/val split for early stopping (fair regularized baseline)
        n = Xn.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).numpy()
        n_val = max(1, int(round(self.val_frac * n)))
        vi, ti = perm[:n_val], perm[n_val:]
        X = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        Y = torch.tensor(y, dtype=torch.float32, device=self.device)
        Xt, Yt, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(self.epochs):
            self.net.train(); opt.zero_grad()
            loss = loss_fn(self.net(Xt), Yt); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                v = loss_fn(self.net(Xv), Yv).item()
            if v < best_val - 1e-6:
                best_val, best_state, bad = v, \
                    {k: p.detach().clone() for k, p in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)

    def predict(self, Xn):
        X = self.t.tensor(Xn, dtype=self.t.float32, device=self.device)
        with self.t.no_grad():
            return self.net(X).cpu().numpy()

    @property
    def n_params(self):
        return sum(p.numel() for p in self.net.parameters())


class KANModel:
    name = "KAN"
    interpretable = "yes (symbolic)"

    # compact symbolic library for clean closed-form rules
    SYM_LIB = ["x", "x^2", "x^3", "1/x", "exp", "log", "sqrt", "tanh", "sin"]

    def __init__(self, hidden=3, steps=60, grid=5, k=3, device="cpu", seed=0):
        self.hidden, self.steps, self.grid, self.k = hidden, steps, grid, k
        self.device, self.seed = device, seed

    def fit(self, Xn, y):
        import torch
        from kan import KAN
        self.t = torch
        torch.manual_seed(self.seed)
        d_in, d_out = Xn.shape[1], y.shape[1]
        self.model = KAN(width=[d_in, self.hidden, d_out], grid=self.grid,
                         k=self.k, seed=self.seed, device=self.device)
        Xt = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        Yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        self.ds = {"train_input": Xt, "train_label": Yt,
                   "test_input": Xt, "test_label": Yt}
        self.model.fit(self.ds, opt="LBFGS", steps=self.steps)

    def predict(self, Xn):
        X = self.t.tensor(Xn, dtype=self.t.float32, device=self.device)
        with self.t.no_grad():
            return self.model(X).cpu().numpy()

    @property
    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def symbolic(self):
        """Proper pykan recipe: prune -> retrain -> auto_symbolic -> REFIT -> formula.
        The refit after auto_symbolic is what was missing before (the symbolic
        affine params must be re-tuned, otherwise the formula collapses to bias)."""
        try:
            m = self.model
            try:
                m = m.prune()                       # drop dead nodes/edges
                m.fit(self.ds, opt="LBFGS", steps=max(20, self.steps // 2))
            except Exception:
                pass
            m.auto_symbolic(lib=self.SYM_LIB)        # fix best symbolic per edge
            m.fit(self.ds, opt="LBFGS", steps=max(20, self.steps // 2))  # refit affine
            self.model = m
            formula = m.symbolic_formula()[0]
            # round coefficients for readability (pykan API varies across versions)
            try:
                from kan.utils import ex_round
                formula = [ex_round(f, 3) for f in formula]
            except Exception:
                pass
            # report refit accuracy of the closed-form model
            import numpy as _np
            pred = self.predict(self.ds["train_input"].cpu().numpy())
            true = self.ds["train_label"].cpu().numpy()
            r2 = 1.0 - ((true - pred) ** 2).sum(0) / \
                (((true - true.mean(0)) ** 2).sum(0) + 1e-12)
            return (f"mu*  = {formula[0]}\n  beta* = {formula[1]}\n"
                    f"  (symbolic-model R2: mu={r2[0]:.3f}, beta={r2[1]:.3f})")
        except Exception as exc:  # pragma: no cover
            return f"(symbolic extraction failed: {exc})"


def closed_loop(model, ds, expect):
    """Achieved vs oracle secret key on held-out feasible test states.

    For each test state build a minimal LinkState from its features, predict
    (mu,beta), evaluate in the analytical environment, and accumulate r_norm
    where the predicted operating point is feasible. Oracle key is the dataset's
    per-step optimum on the same states.
    """
    pred = clip_labels(model.predict(ds.Xte))
    achieved = oracle = 0.0
    sec_pred = sec_oracle = 0
    for i in range(ds.Xte_raw.shape[0]):
        g0, sx, weq, de = ds.Xte_raw[i]
        ls = LinkState(zenith_rad=np.nan, sigma_X=float(sx), h_bar=np.nan,
                       w_eq=float(weq), gamma0=float(g0))
        mu, beta = float(pred[i, 0]), float(pred[i, 1])
        ps, qb, pe, r = evaluate(mu, beta, ls, expect, float(de))
        feasible = (qb < 1e-3 and ps > 1e-3 and pe > 0.1)
        achieved += r if feasible else 0.0
        sec_pred += int(feasible)
        # oracle: this test state is feasible by construction (feasible_only=True)
        oracle += float(ds.yte_rnorm[i])
        sec_oracle += 1
    return achieved, oracle, sec_pred, sec_oracle, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="mlp,knn,linear",
                    help="comma list from {kan,mlp,knn,linear}")
    ap.add_argument("--data", default=str(RESULTS / "dataset.csv"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=1,
                    help="run each model over this many seeds; report mean+/-std")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    expect = make_expectation(SystemParams().gh_order)
    ds = load_dataset(args.data, feasible_only=True, test_frac=0.2, seed=0)

    epochs = 300 if args.quick else 3000
    steps = 20 if args.quick else 60

    def make(key, seed):
        if key == "linear":
            return LinearModel()
        if key == "knn":
            return KNNModel(k=5)
        if key == "mlp":
            return MLPModel(hidden=32, epochs=epochs, device=args.device, seed=seed)
        if key == "kan":
            return KANModel(hidden=3, steps=steps, device=args.device, seed=seed)
        raise KeyError(key)

    seeds = list(range(max(1, args.seeds)))
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    agg = []          # aggregated per-model stats
    kan_seed0 = None  # keep a KAN instance for symbolic extraction
    for key in wanted:
        runs = []
        for sd in seeds:
            try:
                model = make(key, sd)
                t0 = time.time(); model.fit(ds.Xtr, ds.ytr); t_fit = time.time() - t0
            except Exception as exc:
                print(f"[skip {key} seed {sd}: {exc}]"); continue
            pred_te = clip_labels(model.predict(ds.Xte))
            mae_v = mae(pred_te, ds.yte); r2_v = r2_score(pred_te, ds.yte)
            t0 = time.time()
            for _ in range(50):
                model.predict(ds.Xte)
            us = (time.time() - t0) / (50 * ds.Xte.shape[0]) * 1e6
            ach, ora, sp, so, _ = closed_loop(model, ds, expect)
            runs.append({"mae_mu": mae_v[0], "mae_beta": mae_v[1],
                         "r2_mu": r2_v[0], "r2_beta": r2_v[1],
                         "retention": ach / ora if ora > 0 else float("nan"),
                         "sec_pred": sp, "us": us, "n_params": model.n_params})
            if key == "kan" and sd == 0:
                kan_seed0 = model
        if not runs:
            continue
        def ms(field):
            v = np.array([r[field] for r in runs], float)
            return float(v.mean()), float(v.std())
        agg.append({
            "name": make(key, 0).name,
            "mae_mu": ms("mae_mu"), "mae_beta": ms("mae_beta"),
            "r2_mu": ms("r2_mu"), "r2_beta": ms("r2_beta"),
            "retention": ms("retention"), "sec_pred": ms("sec_pred"),
            "us": ms("us")[0], "n_params": runs[-1]["n_params"],
            "interp": make(key, 0).interpretable, "n": len(runs),
        })

    # ---- report ----
    so = ds.Xte.shape[0]
    lines = []
    lines.append("Phase 3: controller comparison (multi-axis trade-off)")
    lines.append("=" * 84)
    lines.append(f"train/test = {ds.Xtr.shape[0]}/{ds.Xte.shape[0]} feasible states; "
                 f"features={ds.feature_names}; seeds={len(seeds)}")
    lines.append("")

    def pm(stat, p=3):
        m, s = stat
        return f"{m:.{p}f}+-{s:.{p}f}" if len(seeds) > 1 else f"{m:.{p}f}"

    hdr = (f"{'model':<11}{'MAE_mu':>14}{'MAE_b':>14}{'keyRet':>16}"
           f"{'secP/'+str(so):>14}{'params':>8}{'us':>8}  interpret")
    lines.append(hdr)
    lines.append("-" * 84)
    for r in agg:
        lines.append(
            f"{r['name']:<11}{pm(r['mae_mu']):>14}{pm(r['mae_beta']):>14}"
            f"{pm(r['retention']):>16}{pm(r['sec_pred'],1):>14}"
            f"{r['n_params']:>8}{r['us']:>8.1f}  {r['interp']}")
    lines.append("-" * 84)
    lines.append("keyRet = achieved/oracle secret key on held-out feasible states")
    lines.append(f"secP/{so} = #states kept operationally feasible (oracle = {so})")
    lines.append("R2 (mu/beta) per model:")
    for r in agg:
        lines.append(f"   {r['name']:<11} R2_mu={pm(r['r2_mu'],2)}  R2_beta={pm(r['r2_beta'],2)}")

    # KAN symbolic rules (design-rule extraction) from seed-0 model
    if kan_seed0 is not None:
        lines.append("\nKAN extracted design rule (symbolic_formula, seed 0):")
        lines.append("  " + kan_seed0.symbolic().replace("\n", "\n  "))

    txt = "\n".join(lines)
    (RESULTS / "kan_comparison.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nsaved: {RESULTS / 'kan_comparison.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
