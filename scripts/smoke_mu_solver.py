"""Smoke test for the decomposed multi-user solver (docs/decomposition.md).

Checks the solver (a) returns a feasible, positive-rate cluster solution near
overhead, (b) beats a naive fixed (mu,beta,chi) baseline, and (c) the mean-field
inner loop is consistent with a brute-force joint grid on a tiny 2-user cluster.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import (SystemParams, build_link_state, make_expectation,  # noqa: E402
                    NodeState, cluster_key_rate, solve_cluster, inner_meanfield)


def to_node(ls):
    return NodeState(ls.gamma0, ls.sigma_X, ls.w_eq)


def main() -> int:
    p = SystemParams()
    expect = make_expectation(p.gh_order)

    # a 4-user cluster spread over a near-overhead region
    alice = to_node(build_link_state(5.0, p))
    bobs = [to_node(build_link_state(z, p)) for z in (0.0, 3.0, 8.0, 12.0)]

    print("=" * 64)
    print("multi-user solver smoke test (N=4 cluster)")
    print("=" * 64)

    t0 = time.time()
    sol = solve_cluster(alice, bobs, expect, d_eve=26.0)
    dt = time.time() - t0
    print(f"\nsolved in {dt:.1f}s")
    print(f"  mu*={sol.mu:.3f}  chi*={sol.chi:.2f}  beta_A*={sol.beta_A:.3f}")
    print(f"  beta_i*={[round(b,2) for b in sol.betas]}")
    print(f"  total_rate={sol.total_rate:.3e} bits/s  n_feasible={sol.n_feasible}/4")

    # naive fixed baseline
    base = cluster_key_rate(0.5, 2.5, [2.5] * 4, 1.0, alice, bobs, expect, d_eve=26.0)
    print(f"\nnaive fixed (mu=0.5,beta=2.5,chi=1): total={base['total_rate']:.3e}  "
          f"n_feasible={base['n_feasible']}/4")
    gain = sol.total_rate / base["total_rate"] if base["total_rate"] > 0 else float("inf")
    print(f"solver gain over naive: {gain:.2f}x")

    # consistency: 2-user cluster, mean-field vs brute-force joint grid
    print("\n[consistency] mean-field vs brute-force joint (2-user, fixed mu,chi)")
    a2 = to_node(build_link_state(2.0, p))
    b2 = [to_node(build_link_state(z, p)) for z in (0.0, 6.0)]
    mu, chi = 0.7, 0.5
    bA_mf, betas_mf, res_mf = inner_meanfield(mu, chi, a2, b2, expect, d_eve=26.0)
    bg = np.linspace(0.3, 4.5, 22)
    best = (-1.0, None)
    for bA in bg:
        for b0 in bg:
            for b1 in bg:
                r = cluster_key_rate(mu, bA, [b0, b1], chi, a2, b2, expect, d_eve=26.0)
                if r["total_rate"] > best[0]:
                    best = (r["total_rate"], (bA, b0, b1))
    print(f"  mean-field : total={res_mf['total_rate']:.4e} "
          f"(beta_A={bA_mf:.2f}, betas={[round(b,2) for b in betas_mf]})")
    print(f"  brute-force: total={best[0]:.4e} (beta={[round(x,2) for x in best[1]]})")
    ratio = res_mf["total_rate"] / best[0] if best[0] > 0 else 0.0
    print(f"  mean-field / brute-force = {ratio:.3f} (close to 1 = good)")

    ok = (sol.total_rate > 0 and sol.n_feasible >= 1 and gain >= 1.0
          and ratio > 0.9)
    print("\nRESULT:", "PASS" if ok else "CHECK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
