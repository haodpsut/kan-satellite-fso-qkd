"""TLE vs analytic-orbit comparison demo (Phase 1 paper-faithful path check).

Builds an analytic 10-sat staggered constellation (the existing simulator
baseline) and a synthetic Starlink-shell TLE constellation over the same
horizon + ground station, propagates both, and reports / plots:
  * served-time fraction
  * peak elevation distribution
  * zenith-vs-time traces (overlaid)

The purpose is to demonstrate that the TLE path is wired correctly end-to-end
(elevation in the right range, handovers happen, no NaNs) and to expose the
qualitative differences a referee would expect (TLE pass durations / spacings
are no longer free parameters but emerge from real Kepler geometry).

Outputs:
  results/tle_vs_analytic.txt      summary statistics
  results/tle_vs_analytic.png      overlaid zenith / served-mask plot

Run:
  python scripts/tle_pass_demo.py
  python scripts/tle_pass_demo.py --horizon 7200 --tle-n 30
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams  # noqa: E402
from satqkd.orbit import example_constellation  # noqa: E402
from satqkd.orbit_tle import example_tle_constellation, PTIT_HANOI  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=float, default=3600.0,
                    help="simulation horizon in seconds (default 3600 = 1 h)")
    ap.add_argument("--dt", type=float, default=10.0, help="time step in seconds")
    ap.add_argument("--tle-n", type=int, default=20,
                    help="synthetic Starlink-shell size (default 20)")
    ap.add_argument("--tle-epoch", type=str,
                    default="2026-05-23T12:00:00",
                    help="UTC epoch for sim t=0 (ISO8601)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    p = SystemParams()
    t_grid = np.arange(0.0, args.horizon, args.dt)

    # Analytic constellation (same as simulate_pass.py default)
    const_an = example_constellation(
        n_sats=10, spacing_s=250.0, altitude=p.H_leo,
        theta_min_deg=[0.0, 2.0, 4.0, 1.0, 3.0, 0.0, 2.0, 4.0, 1.0, 3.0],
    )
    sched_an = const_an.schedule(t_grid)

    # TLE constellation (synthetic Starlink shell over PTIT/Hanoi)
    t0_utc = datetime.fromisoformat(args.tle_epoch).replace(tzinfo=timezone.utc)
    const_tle = example_tle_constellation(n_sats=args.tle_n, t0_utc=t0_utc,
                                          min_elevation=30.0)
    sched_tle = const_tle.schedule(t_grid)

    # --- Stats helpers ---
    def stats(sched, label):
        served = sched["sat_id"] >= 0
        n_served = int(served.sum())
        n_total = len(sched["sat_id"])
        z = sched["zenith_deg"][served]
        e = sched["elevation_deg"][served]
        peak_per_sat = {}
        for sid in np.unique(sched["sat_id"][served]):
            mask = sched["sat_id"] == sid
            peak_per_sat[int(sid)] = float(np.nanmax(sched["elevation_deg"][mask]))
        return {
            "label": label,
            "served_frac": n_served / n_total,
            "n_served": n_served,
            "n_total": n_total,
            "z_min": float(z.min()) if z.size else float("nan"),
            "z_max": float(z.max()) if z.size else float("nan"),
            "z_mean": float(z.mean()) if z.size else float("nan"),
            "e_peak_mean": float(np.mean(list(peak_per_sat.values()))) if peak_per_sat else float("nan"),
            "n_handovers": len(sched["handovers"]),
            "n_unique_sats": len(peak_per_sat),
        }

    s_an = stats(sched_an, "analytic")
    s_tle = stats(sched_tle, f"TLE-synth(n={args.tle_n}, PTIT@Hanoi)")

    lines = []
    lines.append("TLE vs analytic-orbit comparison (Phase 1 paper-faithful path check)")
    lines.append("=" * 70)
    lines.append(f"horizon: {args.horizon:.0f} s, dt: {args.dt:.0f} s "
                 f"({t_grid.size} steps)")
    lines.append(f"ground station: PTIT Hanoi ({PTIT_HANOI.lat_deg:.4f} N, "
                 f"{PTIT_HANOI.lon_deg:.4f} E)")
    lines.append(f"TLE epoch (sim t=0): {args.tle_epoch} UTC")
    lines.append("")
    fmt = "{:<35s} {:>10s} {:>10s} {:>10s} {:>9s} {:>9s} {:>10s} {:>6s}"
    lines.append(fmt.format("source", "served%", "z_min", "z_max", "z_mean",
                            "peakE_avg", "handovers", "#sats"))
    lines.append("-" * 110)
    for s in (s_an, s_tle):
        lines.append("{:<35s} {:>9.1f}% {:>10.1f} {:>10.1f} {:>9.1f} {:>9.1f} {:>10d} {:>6d}"
                     .format(s["label"], 100 * s["served_frac"], s["z_min"], s["z_max"],
                             s["z_mean"], s["e_peak_mean"], s["n_handovers"],
                             s["n_unique_sats"]))
    lines.append("")
    lines.append("Notes:")
    lines.append("- Analytic constellation hand-picks theta_min for near-overhead passes "
                 "to span the full 0..60 deg zenith range (worst-case Phase-2 oracle).")
    lines.append("- TLE shell is randomly phased (real Kepler geometry); peak elevation "
                 "distribution is empirical and depends on shell size + ground station.")
    lines.append("- A real Celestrak Starlink dump (--tle PATH in simulate_pass.py) "
                 "anchors the experiment to a specific UTC epoch and is the figure "
                 "referees will check.")
    summary = "\n".join(lines)
    (RESULTS / "tle_vs_analytic.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)

    if args.no_plot:
        return 0
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Top: zenith vs time (only when served; gaps are NaN -> matplotlib breaks line)
        ax[0].plot(sched_an["t"], sched_an["zenith_deg"], lw=1.2,
                   color="tab:blue", label=f"analytic ({s_an['n_unique_sats']} sats)")
        ax[0].plot(sched_tle["t"], sched_tle["zenith_deg"], lw=1.2,
                   color="tab:orange", alpha=0.85,
                   label=f"TLE-synth ({s_tle['n_unique_sats']} sats)")
        ax[0].axhline(60.0, ls="--", color="gray", lw=0.7, label="zenith mask (60 deg)")
        ax[0].set_ylabel("zenith [deg]")
        ax[0].set_ylim(0, 90)
        ax[0].legend(loc="upper right", fontsize=9)

        # Bottom: served-mask raster
        ax[1].fill_between(sched_an["t"], 0.6, 1.0,
                           where=(sched_an["sat_id"] >= 0),
                           color="tab:blue", alpha=0.4, step="post",
                           label="analytic served")
        ax[1].fill_between(sched_tle["t"], 0.1, 0.5,
                           where=(sched_tle["sat_id"] >= 0),
                           color="tab:orange", alpha=0.4, step="post",
                           label="TLE served")
        ax[1].set_ylim(0, 1.1)
        ax[1].set_yticks([])
        ax[1].set_xlabel("time since epoch [s]")
        ax[1].legend(loc="upper right", fontsize=9)

        fig.suptitle(f"Analytic vs synthetic-TLE constellation over PTIT Hanoi "
                     f"({args.horizon:.0f} s window)")
        fig.tight_layout()
        out = RESULTS / "tle_vs_analytic.png"
        fig.savefig(out, dpi=120)
        print(f"\nsaved plot: {out}")
    except Exception as exc:  # pragma: no cover
        print(f"\n[plot skipped: {exc}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
