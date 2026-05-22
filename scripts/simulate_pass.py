"""Phase 1 simulation: drive the QKD environment over a time-varying LEO pass
with handover, at a fixed baseline (mu, beta).  Produces the time series the
KAN controller will later adapt over.

Outputs (in results/):
  pass_sim.csv       per-timestep: t, sat_id, elevation, zenith, gamma0,
                     sigma_X, w_eq, p_sift, qber, eve_error, secure
  pass_summary.txt   handover events + aggregate stats (committable for review)
  pass_sim.png       elevation / gamma0 / sift / eve vs time (local view)

Run: python scripts/simulate_pass.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation, sift_qber, eve_error  # noqa: E402
from satqkd.orbit import example_constellation  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Baseline operating point (paper-style static choice; Phase 2 will optimize these).
# [P3] uses beta_A=2.5 / beta_Bi=2.25 at a single snapshot; across a whole pass
# beta=2.5 leaves QBER ~1.6e-3 (just above 1e-3) even at overhead, so we take a
# slightly higher static beta=3.0 to expose a clear secure window. This static
# choice is exactly what the Phase-3 KAN controller will improve on.
MU = 0.5
BETA = 3.0
D_EVE = 26.0          # Eve offset [m] ([P3] Sec. V-D)
QBER_MAX = 1e-3
PSIFT_MIN = 1e-3
EVE_MIN = 0.1


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    p = SystemParams()
    expect = make_expectation(p.gh_order)

    # 10 staggered LEOs, closest approach every 250 s. theta_min is the central
    # angle at closest approach: only small values (<~6.5 deg) clear the 30 deg
    # elevation mask, so we use near-overhead passes to span the full zenith range
    # 0..60 deg (cf. [P3] Fig. 8) and produce overlapping windows + handovers.
    const = example_constellation(
        n_sats=10, spacing_s=250.0, altitude=p.H_leo,
        theta_min_deg=[0.0, 2.0, 4.0, 1.0, 3.0, 0.0, 2.0, 4.0, 1.0, 3.0],
    )
    t_grid = np.arange(0.0, 3000.0, 5.0)
    sched = const.schedule(t_grid)

    rows = []
    secure_count = 0
    operational_count = 0
    served_count = 0
    min_qber = np.inf
    for i, t in enumerate(sched["t"]):
        sid = int(sched["sat_id"][i])
        z = sched["zenith_deg"][i]
        e = sched["elevation_deg"][i]
        if sid < 0 or not np.isfinite(z):
            rows.append((t, -1, np.nan, np.nan, np.nan, np.nan, np.nan,
                         np.nan, np.nan, np.nan, 0))
            continue
        served_count += 1
        ls = build_link_state(float(z), p)
        psift, qber = sift_qber(ls.gamma(MU), BETA, ls.sigma_X, expect)
        pe = eve_error(ls.gamma_eve(MU, D_EVE), ls.sigma_X, expect)
        operational = (psift > PSIFT_MIN and pe > EVE_MIN)
        secure = int(operational and qber < QBER_MAX)
        operational_count += int(operational)
        secure_count += secure
        min_qber = min(min_qber, qber)
        rows.append((t, sid, e, z, ls.gamma0, ls.sigma_X, ls.w_eq,
                     psift, qber, pe, secure))

    # --- write CSV ---
    header = ("t,sat_id,elevation_deg,zenith_deg,gamma0,sigma_X,w_eq,"
              "p_sift,qber,eve_error,secure")
    csv_path = RESULTS / "pass_sim.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join("" if (isinstance(v, float) and np.isnan(v)) else
                             (f"{v:.6g}" if isinstance(v, float) else str(v))
                             for v in r) + "\n")

    # --- summary ---
    served = [r for r in rows if r[1] >= 0]
    g0 = np.array([r[4] for r in served])
    summary = []
    summary.append("Phase 1 pass simulation summary")
    summary.append("=" * 50)
    summary.append(f"horizon: {t_grid[0]:.0f}..{t_grid[-1]:.0f} s, step {t_grid[1]-t_grid[0]:.0f} s "
                   f"({t_grid.size} steps)")
    summary.append(f"baseline: mu={MU}, beta={BETA}, d_eve={D_EVE} m")
    summary.append(f"coverage:    {served_count}/{t_grid.size} steps served "
                   f"({100*served_count/t_grid.size:.1f}%)")
    summary.append(f"operational: {operational_count}/{served_count} served steps "
                   f"({100*operational_count/max(served_count,1):.1f}%)  [P_sift>1e-3 & Eve>0.1]")
    summary.append(f"secure:      {secure_count}/{served_count} served steps "
                   f"({100*secure_count/max(served_count,1):.1f}%)  [+ QBER<1e-3]")
    summary.append(f"best QBER over pass: {min_qber:.3e}")
    if g0.size:
        summary.append(f"gamma0 over served steps: min={g0.min():.3f} "
                       f"max={g0.max():.3f} mean={g0.mean():.3f}")
    summary.append(f"handover events: {len(sched['handovers'])}")
    for (ht, a, b) in sched["handovers"]:
        label = f"sat {a} -> sat {b}" if a >= 0 and b >= 0 else \
                (f"acquire sat {b}" if a < 0 else f"lose sat {a} (gap)")
        summary.append(f"   t={ht:7.1f} s : {label}")
    summary_txt = "\n".join(summary)
    (RESULTS / "pass_summary.txt").write_text(summary_txt + "\n", encoding="utf-8")
    print(summary_txt)

    # --- plot (best-effort; skipped if matplotlib missing) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array([r[0] for r in rows])
        elev = np.array([r[2] for r in rows])
        gamma0 = np.array([r[4] for r in rows])
        psift = np.array([r[7] for r in rows])
        pe = np.array([r[9] for r in rows])
        fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        ax[0].plot(t, elev); ax[0].axhline(30, ls="--", c="gray"); ax[0].set_ylabel("elev [deg]")
        ax[1].plot(t, gamma0, c="tab:orange"); ax[1].set_ylabel("gamma0")
        ax[2].semilogy(t, psift, c="tab:green"); ax[2].axhline(PSIFT_MIN, ls="--", c="gray")
        ax[2].set_ylabel("P_sift")
        ax[3].plot(t, pe, c="tab:red"); ax[3].axhline(EVE_MIN, ls="--", c="gray")
        ax[3].set_ylabel("Eve error"); ax[3].set_xlabel("time [s]")
        for (ht, *_), in [(h,) for h in sched["handovers"]]:
            for a in ax:
                a.axvline(ht, c="k", lw=0.5, alpha=0.3)
        fig.suptitle("Phase 1: QKD metrics over a LEO pass (baseline mu=0.5, beta=2.5)")
        fig.tight_layout()
        fig.savefig(RESULTS / "pass_sim.png", dpi=120)
        print(f"\nsaved plot: {RESULTS / 'pass_sim.png'}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[plot skipped: {exc}]")

    print(f"saved CSV:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
