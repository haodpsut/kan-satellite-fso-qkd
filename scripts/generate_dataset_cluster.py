"""Phase 4: supervised dataset for the multi-user cluster controller.

For many random clusters (variable N, heterogeneous geometry) we solve the joint
problem (P) with the decomposed solver and emit two label sets matching a
two-stage, variable-N controller:

  global  (one row per cluster): cluster-aggregate features -> (mu*, chi*, beta_A*)
  user    (one row per Bob)    : per-user features + (mu*, chi*, Alice ctx) -> beta_i*

The per-user rows share a single regression head (applied independently to each
user), so the controller handles any N. solve_cluster is CPU-heavy, so the full
dataset is meant to run on the server; use --quick for a local smoke.

Outputs results/cluster_global.csv and results/cluster_user.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation  # noqa: E402
from satqkd import NodeState, solve_cluster  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

GLOBAL_FEATURES = ["gA", "sA", "wA", "N", "g_mean", "g_min", "s_mean", "d_eve"]
GLOBAL_LABELS = ["mu", "chi", "beta_A"]
GLOBAL_EXTRA = ["rate_opt", "nfeas_opt"]   # oracle outcome (for closed-loop eval)
USER_FEATURES = ["g_i", "s_i", "w_i", "gA", "sA", "mu", "chi"]
USER_LABELS = ["beta_i"]


def to_node(z, p):
    ls = build_link_state(float(np.clip(z, 0.0, 60.0)), p)
    return NodeState(ls.gamma0, ls.sigma_X, ls.w_eq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=None, help="number of clusters")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    rng = np.random.default_rng(args.seed)

    n_clusters = args.n if args.n else (12 if args.quick else 250)
    # smaller solver grids keep dataset generation tractable
    mu_grid = np.linspace(0.4, 0.95, 8)
    chi_grid = np.array([0.0, 0.5, 1.0])
    inner_kw = dict(n_beta=17, max_iter=8)

    grows, urows = [], []
    feas_clusters = 0
    cid = 0
    t0 = time.time()
    # multi-user pair QKD is feasible only near overhead (both nodes need high
    # SNR), so concentrate sampling there to obtain feasible-labelled clusters.
    for c in range(n_clusters):
        base = float(rng.uniform(0.0, 8.0))
        N = int(rng.integers(2, 5))                   # 2..4 users
        d_eve = float(rng.uniform(22.0, 42.0))
        alice = to_node(base + rng.uniform(0.0, 2.0), p)
        bob_z = base + rng.uniform(0.0, 5.0, size=N)
        bobs = [to_node(z, p) for z in bob_z]

        sol = solve_cluster(alice, bobs, expect, d_eve=d_eve,
                            mu_grid=mu_grid, chi_grid=chi_grid, **inner_kw)
        if sol.n_feasible < 1:
            continue
        feas_clusters += 1
        g = [b.gamma0 for b in bobs]
        s = [b.sigma_X for b in bobs]
        rate_opt = sum(sol.betas) * 0.0  # placeholder; real rate via cluster_key_rate
        from satqkd import cluster_key_rate
        res = cluster_key_rate(sol.mu, sol.beta_A, sol.betas, sol.chi, alice, bobs,
                               expect, d_eve=d_eve)
        rate_opt = sum(r for r, f in zip(res["per_rate"], res["per_feasible"]) if f)
        grows.append([cid, alice.gamma0, alice.sigma_X, alice.w_eq, N,
                      float(np.mean(g)), float(np.min(g)), float(np.mean(s)), d_eve,
                      sol.mu, sol.chi, sol.beta_A, rate_opt, sol.n_feasible])
        for i, b in enumerate(bobs):
            urows.append([cid, b.gamma0, b.sigma_X, b.w_eq, alice.gamma0,
                          alice.sigma_X, sol.mu, sol.chi, sol.betas[i], d_eve])
        cid += 1
        if (c + 1) % max(1, n_clusters // 10) == 0:
            print(f"  {c+1}/{n_clusters} clusters ({feas_clusters} feasible) "
                  f"{time.time()-t0:.0f}s")

    def write(path, header, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.6g}" for v in r) + "\n")

    write(RESULTS / "cluster_global.csv",
          ["cluster_id"] + GLOBAL_FEATURES + GLOBAL_LABELS + GLOBAL_EXTRA, grows)
    write(RESULTS / "cluster_user.csv",
          ["cluster_id"] + USER_FEATURES + USER_LABELS + ["d_eve"], urows)
    print(f"\nfeasible clusters: {feas_clusters}/{n_clusters}")
    print(f"global rows: {len(grows)}  user rows: {len(urows)}")
    print(f"elapsed: {time.time()-t0:.0f}s")
    print(f"saved: {RESULTS/'cluster_global.csv'}, {RESULTS/'cluster_user.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
