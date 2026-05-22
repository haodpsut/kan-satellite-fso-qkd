"""Phase 2 -> Phase 3 bridge: generate the supervised dataset that the KAN
controller learns from.

For a grid of channel states (zenith angle x assumed worst-case Eve distance
d_eve) we solve Eq. (26) for the optimal (mu*, beta*) and record the achieved
normalized key rate. d_eve is included as an input feature because the optimal
modulation depth mu* only becomes non-trivial when Eve is close enough for the
Eve-error constraint to bind (otherwise mu* saturates at its max).

Output: results/dataset.csv with columns
    zenith_deg, gamma0, sigma_X, w_eq, d_eve   (features)
    mu_opt, beta_opt, rnorm_opt, feasible       (labels)

Usage:
    python scripts/generate_dataset.py             # default grid
    python scripts/generate_dataset.py --quick     # small grid (smoke test)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation  # noqa: E402
from satqkd.optimize import optimize_state  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid for smoke test")
    ap.add_argument("--zmax", type=float, default=60.0, help="max zenith [deg]")
    ap.add_argument("--out", type=str, default="dataset.csv")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    p = SystemParams()
    expect = make_expectation(p.gh_order)

    if args.quick:
        zen = np.arange(0.0, args.zmax + 1e-9, 5.0)
        d_eve = np.array([5.0, 26.0])
    else:
        zen = np.arange(0.0, args.zmax + 1e-9, 1.0)
        d_eve = np.array([2.0, 5.0, 8.0, 12.0, 18.0, 26.0, 35.0, 50.0])

    # LinkState depends only on zenith; build once per zenith and reuse over d_eve.
    states = {z: build_link_state(float(z), p) for z in zen}

    rows = []
    n = len(zen) * len(d_eve)
    t0 = time.time()
    done = 0
    for z in zen:
        ls = states[z]
        for de in d_eve:
            opt = optimize_state(ls, expect, d_eve=float(de))
            rows.append((float(z), ls.gamma0, ls.sigma_X, ls.w_eq, float(de),
                         opt.mu, opt.beta, opt.r_norm, int(opt.feasible)))
            done += 1
        if done % max(1, (n // 10)) < len(d_eve):
            print(f"  {done}/{n} states ... ({time.time()-t0:.1f}s)")

    out = RESULTS / args.out
    header = "zenith_deg,gamma0,sigma_X,w_eq,d_eve,mu_opt,beta_opt,rnorm_opt,feasible"
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                             for v in r) + "\n")

    feas = sum(r[-1] for r in rows)
    print(f"\nwrote {len(rows)} rows -> {out}")
    print(f"feasible (secure-capable) states: {feas}/{len(rows)} "
          f"({100*feas/len(rows):.1f}%)")
    print(f"elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
