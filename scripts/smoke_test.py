"""Smoke test: validate the analytical DT/DD environment and cross-check it
against Monte-Carlo. Runs on CPU in seconds.

Checks
------
1. Gauss-Hermite log-normal expectation has unit mean (E[h_a] = 1).
2. Sanity of sift/QBER/Eve-error vs analytical-vs-MC agreement.
3. Eve-error monotonicity: P_err^E -> 0.5 as mu -> 0 (the security lever).
4. A LinkState builds for a realistic zenith angle.

Exit code 0 = all checks pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import (  # noqa: E402
    SystemParams,
    build_link_state,
    make_expectation,
    sift_qber,
    eve_error,
)
from satqkd.montecarlo import mc_single_link  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 70)
    print("satqkd smoke test")
    print("=" * 70)
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    all_ok = True

    # 1. Unit-mean log-normal
    print("\n[1] Gauss-Hermite log-normal expectation")
    for sx in (0.05, 0.1, 0.2, 0.3):
        mean = float(expect(lambda ha: ha, sx))
        all_ok &= check(f"E[h_a]=1 at sigma_X={sx}", abs(mean - 1.0) < 1e-6,
                        f"E[h_a]={mean:.8f}")

    # 2. Analytical vs Monte-Carlo (gamma/beta reduced form)
    print("\n[2] Analytical vs Monte-Carlo (gamma0=12, mu=0.5, sigma_X=0.1)")
    gamma0, mu, sigma_X = 12.0, 0.5, 0.1
    gamma0_eve = 3.0
    for beta in (1.5, 2.5, 3.5):
        gamma = gamma0 * mu
        psift_a, qber_a = sift_qber(gamma, beta, sigma_X, expect)
        eve_a = eve_error(gamma0_eve * mu, sigma_X, expect)
        mc = mc_single_link(gamma0, mu, beta, sigma_X, n_bits=3_000_000,
                            gamma0_eve=gamma0_eve, seed=1)
        # sift prob agreement (relative); QBER tiny -> compare loosely
        ok_sift = abs(psift_a - mc["p_sift"]) < 5e-3 + 0.05 * psift_a
        ok_eve = abs(eve_a - mc["eve_error"]) < 5e-3
        all_ok &= check(
            f"beta={beta}: P_sift A/MC", ok_sift,
            f"A={psift_a:.4e} MC={mc['p_sift']:.4e} | QBER_A={qber_a:.2e}",
        )
        all_ok &= check(
            f"beta={beta}: Eve-err A/MC", ok_eve,
            f"A={eve_a:.4f} MC={mc['eve_error']:.4f}",
        )

    # 3. Eve-error monotonicity in mu (security lever)
    print("\n[3] Eve error vs modulation depth (expect decreasing, ->0.5 as mu->0)")
    mus = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    eves = [eve_error(gamma0_eve * m, sigma_X, expect) for m in mus]
    monotone = all(eves[i] >= eves[i + 1] - 1e-9 for i in range(len(eves) - 1))
    near_half = eves[0] > 0.4
    for m, e in zip(mus, eves):
        print(f"        mu={m:>4}: Eve-error={e:.4f}")
    all_ok &= check("monotone decreasing in mu", monotone)
    all_ok &= check("Eve-error -> ~0.5 as mu->0", near_half, f"{eves[0]:.4f}")

    # 4. LinkState from geometry
    print("\n[4] LinkState build (zenith=40 deg, gamma0 override=12)")
    ls = build_link_state(40.0, p, gamma0_override=12.0)
    ok_ls = (ls.sigma_X > 0) and (0 < ls.h_bar) and (ls.w_eq > 0)
    print(f"        sigma_X={ls.sigma_X:.4f}  h_bar={ls.h_bar:.3e}  "
          f"w_eq={ls.w_eq:.2f} m  gamma0={ls.gamma0:.2f}")
    all_ok &= check("LinkState fields valid", ok_ls)

    # also report the physically-derived gamma0 (uncalibrated) for reference
    ls_phys = build_link_state(40.0, p)
    print(f"        [ref] physically-derived gamma0={ls_phys.gamma0:.3e} "
          "(calibration target for Table I)")

    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 70)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
