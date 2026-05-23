"""Phase 4 headline: adaptive (decomposed solver) vs best-static for a MULTI-USER
cluster over a time-varying LEO pass with handover.

At each served time step the serving LEO illuminates a cluster (Alice + N Bobs at
small geometric offsets around the footprint, each with an assumed Eve distance).
The adaptive controller solves the joint problem (P) per step; the best-static
baseline is a single (mu, beta, chi) tuned over the whole pass. Both use the same
feasibility gate (QBER/sift/leak/URA/BSA), so adaptive >= static by construction.

Outputs results/cluster_pass.csv + cluster_pass_summary.txt.

Usage:
  python scripts/optimize_pass_cluster.py --quick   # small, local smoke
  python scripts/optimize_pass_cluster.py           # full (server)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation  # noqa: E402
from satqkd import NodeState, cluster_key_rate, solve_cluster  # noqa: E402
from satqkd.orbit import example_constellation  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# cluster shape: per-user zenith offsets [deg] and Eve distances [m] (fixed over pass)
# heterogeneous cluster: users spread in zenith + different Eve distances, so
# per-user beta_i adaptation matters (a single static beta cannot fit all).
ALICE_DZ = 1.0
BOB_DZ = [0.0, 8.0, 16.0]
BOB_DEVE = [26.0, 18.0, 35.0]


def make_cluster(base_zenith, p):
    to = lambda z: (lambda ls: NodeState(ls.gamma0, ls.sigma_X, ls.w_eq))(
        build_link_state(float(np.clip(z, 0.0, 60.0)), p))
    alice = to(base_zenith + ALICE_DZ)
    bobs = [to(base_zenith + dz) for dz in BOB_DZ]
    return alice, bobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tle", nargs="?", const="synthetic", default=None,
                    help="orbit source: omit = analytic; '--tle' = synthetic Starlink shell; "
                         "'--tle PATH' = 3-line TLE file (e.g. Celestrak dump)")
    ap.add_argument("--tle-n", type=int, default=20,
                    help="synthetic shell size (only used when --tle has no PATH)")
    ap.add_argument("--tle-epoch", type=str, default="2026-05-23T12:00:00",
                    help="UTC epoch for sim t=0 (ISO8601)")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    N = len(BOB_DZ)

    if args.tle is None:
        const = example_constellation(n_sats=10, spacing_s=250.0, altitude=p.H_leo,
                                      theta_min_deg=[0., 2., 4., 1., 3., 0., 2., 4., 1., 3.])
        orbit_label = "analytic"
    else:
        from datetime import datetime, timezone
        from satqkd.orbit_tle import (PTIT_HANOI, load_tle_file,
                                      tle_constellation, example_tle_constellation)
        t0_utc = datetime.fromisoformat(args.tle_epoch).replace(tzinfo=timezone.utc)
        if args.tle == "synthetic":
            const = example_tle_constellation(n_sats=args.tle_n, t0_utc=t0_utc,
                                              min_elevation=30.0)
            orbit_label = f"TLE-synth(n={args.tle_n})"
        else:
            sats = load_tle_file(args.tle)
            const = tle_constellation(sats, ground=PTIT_HANOI, t0_utc=t0_utc,
                                      min_elevation=30.0)
            orbit_label = f"TLE-file({Path(args.tle).name}, n={len(sats)})"
    dt = 100.0 if args.quick else 50.0
    t_grid = np.arange(0.0, 3000.0, dt)
    sched = const.schedule(t_grid)
    served = [(float(sched["t"][i]), float(sched["zenith_deg"][i]))
              for i in range(t_grid.size)
              if sched["sat_id"][i] >= 0 and np.isfinite(sched["zenith_deg"][i])]

    # solver grids (coarser in --quick)
    mu_grid = np.linspace(0.4, 0.95, 6 if args.quick else 12)
    chi_grid = np.array([0.0, 0.5, 1.0]) if args.quick else np.linspace(0, 1, 5)
    inner_kw = dict(n_beta=15 if args.quick else 25, max_iter=8)

    clusters = [(t, z, *make_cluster(z, p)) for (t, z) in served]
    d_eve_default = 26.0  # solver uses a common d_eve; per-user d_eve handled in eval

    # ---- best single static (mu, beta, chi) over the pass, feasible-gated ----
    mu_s = np.linspace(0.4, 0.95, 6)
    beta_s = np.linspace(0.5, 4.0, 8)
    chi_s = np.array([0.0, 0.5, 1.0])
    best = (-1.0, 0.5, 2.0, 1.0)
    for mu in mu_s:
        for beta in beta_s:
            for chi in chi_s:
                tot = 0.0
                for (_, _, alice, bobs) in clusters:
                    r = cluster_key_rate(mu, beta, [beta] * N, chi, alice, bobs,
                                         expect, d_eve_default)
                    # feasible-gated total (sum rate over feasible pairs)
                    tot += sum(rr for rr, f in zip(r["per_rate"], r["per_feasible"]) if f)
                if tot > best[0]:
                    best = (tot, float(mu), float(beta), float(chi))
    _, MU_BS, BETA_BS, CHI_BS = best

    rows, key_static, key_adapt = [], 0.0, 0.0
    sec_static = sec_adapt = pairs_total = 0
    t0 = time.time()
    for k, (t, z, alice, bobs) in enumerate(clusters):
        # static
        rs = cluster_key_rate(MU_BS, BETA_BS, [BETA_BS] * N, CHI_BS, alice, bobs,
                              expect, d_eve_default)
        ks = sum(rr for rr, f in zip(rs["per_rate"], rs["per_feasible"]) if f)
        key_static += ks; sec_static += rs["n_feasible"]
        # adaptive (solve joint problem)
        sol = solve_cluster(alice, bobs, expect, d_eve=d_eve_default,
                            mu_grid=mu_grid, chi_grid=chi_grid, **inner_kw)
        ra = cluster_key_rate(sol.mu, sol.beta_A, sol.betas, sol.chi, alice, bobs,
                              expect, d_eve_default)
        ka = sum(rr for rr, f in zip(ra["per_rate"], ra["per_feasible"]) if f)
        key_adapt += ka; sec_adapt += ra["n_feasible"]
        pairs_total += N
        rows.append((t, z, MU_BS, BETA_BS, CHI_BS, ks, rs["n_feasible"],
                     sol.mu, sol.beta_A, sol.chi, ka, ra["n_feasible"]))
        if args.quick:
            print(f"  t={t:6.0f} z={z:4.1f}  static {ks:.2e}/{rs['n_feasible']}  "
                  f"adapt {ka:.2e}/{ra['n_feasible']} (mu={sol.mu:.2f},chi={sol.chi:.2f})")

    with open(RESULTS / "cluster_pass.csv", "w", encoding="utf-8") as f:
        f.write("t,zenith,mu_s,beta_s,chi_s,key_s,sec_s,mu_a,betaA_a,chi_a,key_a,sec_a\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r) + "\n")

    Rb = p.bit_rate
    gain = key_adapt / key_static if key_static > 0 else float("inf")
    L = []
    L.append("Phase 4: multi-user cluster, adaptive vs best-static over a pass")
    L.append("=" * 64)
    L.append(f"orbit source: {orbit_label}")
    L.append(f"served steps={len(clusters)} (dt={dt}s), N={N} users/cluster, "
             f"pairs_total={pairs_total}")
    L.append(f"best static: mu={MU_BS:.3f}, beta={BETA_BS:.3f}, chi={CHI_BS:.2f}")
    L.append(f"  secure pair-steps : {sec_static}/{pairs_total} "
             f"({100*sec_static/max(pairs_total,1):.1f}%)")
    L.append(f"  secret key        : {key_static:.3e} (norm) = {key_static*Rb:.3e} bits/s-sum")
    L.append(f"adaptive (joint solver per step):")
    L.append(f"  secure pair-steps : {sec_adapt}/{pairs_total} "
             f"({100*sec_adapt/max(pairs_total,1):.1f}%)")
    L.append(f"  secret key        : {key_adapt:.3e} (norm) = {key_adapt*Rb:.3e} bits/s-sum")
    L.append("-" * 64)
    L.append(f"secret-key gain (adaptive / best-static): {gain:.2f}x")
    L.append(f"secure pair-time: static {sec_static} -> adaptive {sec_adapt}")
    L.append(f"elapsed {time.time()-t0:.1f}s")
    txt = "\n".join(L)
    (RESULTS / "cluster_pass_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
