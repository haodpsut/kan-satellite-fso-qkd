"""Phase 2 headline experiment: adaptive (per-step optimal) vs static baseline
parameters over a time-varying LEO pass.

For each served time step we solve Eq. (26) for the optimal (mu, beta) and
compare the resulting secure time and accumulated secret-key rate against the
static baseline. This quantifies the gain the Phase-3 KAN controller targets.

Outputs (results/):
  optimize_pass.csv      per-step static vs optimal (mu,beta,p_sift,qber,eve,rate)
  optimize_pass_summary.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation  # noqa: E402
from satqkd.optimize import optimize_state, evaluate  # noqa: E402
from satqkd.orbit import example_constellation  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

MU_STATIC = 0.5
BETA_STATIC = 3.0
D_EVE = 26.0
DT = 5.0  # time step [s]


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    const = example_constellation(
        n_sats=10, spacing_s=250.0, altitude=p.H_leo,
        theta_min_deg=[0.0, 2.0, 4.0, 1.0, 3.0, 0.0, 2.0, 4.0, 1.0, 3.0],
    )
    t_grid = np.arange(0.0, 3000.0, DT)
    sched = const.schedule(t_grid)

    # precompute served LinkStates once
    served_idx = [i for i in range(sched["t"].size)
                  if sched["sat_id"][i] >= 0 and np.isfinite(sched["zenith_deg"][i])]
    states = [build_link_state(float(sched["zenith_deg"][i]), p) for i in served_idx]
    served = len(served_idx)

    def feasible(psift, qber, pe):
        return (qber < 1e-3 and psift > 1e-3 and pe > 0.1)

    # ---- best single static (mu,beta) maximizing total FEASIBLE-GATED key ----
    # Key is counted only at steps where the config is operationally feasible
    # (QBER<1e-3 & P_sift>1e-3 & Eve>0.1), the same gate applied to adaptive,
    # so the comparison is fair and per-step adaptive >= any static by construction.
    mu_g = np.linspace(0.05, 0.95, 19)
    beta_g = np.linspace(0.3, 4.5, 22)
    best = (-1.0, MU_STATIC, BETA_STATIC)
    for mu in mu_g:
        for beta in beta_g:
            tot = 0.0
            for ls in states:
                ps, qb, pe, r = evaluate(mu, beta, ls, expect, D_EVE)
                if feasible(ps, qb, pe):
                    tot += r
            if tot > best[0]:
                best = (tot, float(mu), float(beta))
    MU_BS, BETA_BS = best[1], best[2]

    rows = []
    sec_static = sec_opt = sec_bs = 0
    key_static = key_opt = key_bs = 0.0  # accumulated normalized key (sum r_norm * dt)
    Rb = p.bit_rate

    for k, i in enumerate(served_idx):
        t = sched["t"][i]
        z = sched["zenith_deg"][i]
        sid = int(sched["sat_id"][i])
        ls = states[k]

        # naive static baseline (paper-style conservative pick); key gated by feasibility
        ps_s, qb_s, pe_s, r_s = evaluate(MU_STATIC, BETA_STATIC, ls, expect, D_EVE)
        feas_s = feasible(ps_s, qb_s, pe_s)
        sec_static += int(feas_s); key_static += (r_s if feas_s else 0.0) * DT

        # best single static over the whole pass (fair baseline)
        ps_b, qb_b, pe_b, r_b = evaluate(MU_BS, BETA_BS, ls, expect, D_EVE)
        feas_b = feasible(ps_b, qb_b, pe_b)
        sec_bs += int(feas_b); key_bs += (r_b if feas_b else 0.0) * DT

        # adaptive optimum (key counted only when the optimal point is feasible)
        opt = optimize_state(ls, expect, d_eve=D_EVE)
        sec_opt += int(opt.feasible)
        key_opt += (opt.r_norm if opt.feasible else 0.0) * DT

        rows.append((t, sid, z, ls.gamma0,
                     MU_STATIC, BETA_STATIC, ps_s, qb_s, pe_s, r_s, int(feas_s),
                     opt.mu, opt.beta, opt.p_sift, opt.qber, opt.eve_error,
                     opt.r_norm, int(opt.feasible)))

    # CSV
    header = ("t,sat_id,zenith_deg,gamma0,"
              "mu_s,beta_s,psift_s,qber_s,eve_s,rnorm_s,secure_s,"
              "mu_opt,beta_opt,psift_opt,qber_opt,eve_opt,rnorm_opt,secure_opt")
    with open(RESULTS / "optimize_pass.csv", "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v)
                             for v in r) + "\n")

    # key in bits = normalized * Rb
    key_static_bits = key_static * Rb
    key_bs_bits = key_bs * Rb
    key_opt_bits = key_opt * Rb
    gain_naive = key_opt_bits / key_static_bits if key_static_bits > 0 else float("inf")
    gain_fair = key_opt_bits / key_bs_bits if key_bs_bits > 0 else float("inf")

    lines = []
    lines.append("Phase 2: adaptive vs static over a LEO pass")
    lines.append("=" * 56)
    lines.append(f"served steps: {served}  (dt={DT} s)")
    lines.append(f"naive static (mu={MU_STATIC}, beta={BETA_STATIC}):")
    lines.append(f"    secure steps : {sec_static}/{served} "
                 f"({100*sec_static/max(served,1):.1f}%)")
    lines.append(f"    secret key   : {key_static_bits:.3e} bits")
    lines.append(f"best static  (mu={MU_BS:.3f}, beta={BETA_BS:.3f}) [tuned for whole pass]:")
    lines.append(f"    secure steps : {sec_bs}/{served} "
                 f"({100*sec_bs/max(served,1):.1f}%)")
    lines.append(f"    secret key   : {key_bs_bits:.3e} bits")
    lines.append(f"adaptive (per-step optimal mu,beta):")
    lines.append(f"    secure steps : {sec_opt}/{served} "
                 f"({100*sec_opt/max(served,1):.1f}%)")
    lines.append(f"    secret key   : {key_opt_bits:.3e} bits")
    lines.append("-" * 56)
    lines.append(f"secret-key gain vs naive static : {gain_naive:.2f}x")
    lines.append(f"secret-key gain vs BEST static  : {gain_fair:.2f}x   <-- fair headline")
    lines.append(f"secure-time: naive {sec_static} / best-static {sec_bs} / adaptive {sec_opt}")
    # show a few sample optimal points across zenith
    lines.append("\nsample optimal params vs zenith:")
    lines.append("  zenith  gamma0   mu_opt  beta_opt   rnorm_opt  secure")
    seen = set()
    for r in rows:
        zr = int(round(r[2] / 10.0) * 10)
        if zr in seen:
            continue
        seen.add(zr)
        lines.append(f"  {r[2]:5.1f}  {r[3]:6.3f}   {r[11]:5.3f}   {r[12]:6.3f}   "
                     f"{r[16]:.3e}   {r[17]}")
    txt = "\n".join(lines)
    (RESULTS / "optimize_pass_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)
    print(f"\nsaved: {RESULTS/'optimize_pass.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
