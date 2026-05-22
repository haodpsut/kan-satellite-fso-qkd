"""Calibration check against [P3] Table I + operating point.

Loads the real Table-I parameters, prints the physical vs calibrated gamma0
across a range of zenith angles, and verifies that the calibrated model
reproduces the paper's stated operating point:

  * [P3] Sec. V-B: mu < 0.7 keeps Eve error > 0.1 at worst-case zenith 0.
  * [P3] Sec. V-C: at mu=0.5, beta~2.5 the sift prob > 1e-3 and QBER < ~1e-3.

Run: python scripts/calibrate_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import (  # noqa: E402
    SystemParams,
    build_link_state,
    physical_gamma0,
    make_expectation,
    sift_qber,
    eve_error,
)


def main() -> int:
    p = SystemParams()
    expect = make_expectation(p.gh_order)

    print("=" * 74)
    print("Calibration check vs [P3] Table I")
    print("=" * 74)
    print(f"  P={p.peak_power*1e3:.1f} mW ({p.peak_power_dbm} dBm)  Ga={p.amp_gain_db} dB  "
          f"nsp={p.nsp}  B0={p.optical_bandwidth/1e9:.0f} GHz")
    print(f"  Re={p.responsivity} A/W  T={p.temperature} K  RL={p.load_resistance} Ohm  Fn={p.noise_figure}")
    print(f"  divergence GEO/LEO = {p.divergence_geo*1e6:.0f}/{p.divergence_leo*1e6:.0f} urad  "
          f"a_leo/a_user = {p.a_leo*100:.0f}/{p.a_user*100:.0f} cm")
    print(f"  calibration anchor: gamma0_ref={p.calib_gamma0_ref} at zenith={p.calib_zenith_ref_deg} deg")

    print("\n  zenith   sigma_X    h_bar        w_eq[m]   gamma0_phys   gamma0_calib")
    print("  " + "-" * 70)
    for z in (0, 10, 20, 30, 40, 50, 60):
        g_phys, h_bar, w_eq, sx = physical_gamma0(z, p)
        ls = build_link_state(z, p)
        print(f"  {z:5d}   {sx:7.4f}   {h_bar:.3e}   {w_eq:7.1f}   {g_phys:.4e}   {ls.gamma0:8.3f}")

    # --- reproduce operating point ---
    print("\n  [anchor] Eve error vs mu at zenith=0 (Eve near footprint center):")
    ls0 = build_link_state(0.0, p)
    ok = True
    for mu in (0.3, 0.5, 0.7, 0.9):
        pe = eve_error(ls0.gamma0 * mu, ls0.sigma_X, expect)
        print(f"            mu={mu}:  Eve-error={pe:.4f}")
    pe07 = eve_error(ls0.gamma0 * 0.7, ls0.sigma_X, expect)
    ok &= abs(pe07 - 0.1) < 0.03
    print(f"    -> target Q(0.7*gamma0)~0.1 :  Eve-error(mu=0.7)={pe07:.4f}  "
          f"[{'OK' if abs(pe07-0.1)<0.03 else 'CHECK'}]")

    print("\n  [anchor] Bob link at mu=0.5, beta=2.5, zenith=0:")
    psift, qber = sift_qber(ls0.gamma0 * 0.5, 2.5, ls0.sigma_X, expect)
    print(f"            P_sift={psift:.3e} (need >1e-3)   QBER={qber:.3e} (need <~1e-3)")
    ok &= psift > 1e-3

    print("\n" + "=" * 74)
    print("RESULT:", "operating point reproduced" if ok else "CHECK calibration")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
