# kan-satellite-fso-qkd

**KAN-based adaptive parameter control for time-varying multi-user satellite FSO/QKD.**

Research code for a paper extending the DT/DD CV-QKD lineage (Vu, Pham, Dang, Le,
Pham — IEEE Access 2020/2022, IEEE Photonics J. 2023) toward an **adaptive**
system: as a LEO satellite traverses its pass, the channel geometry changes, so
the optimal QKD parameters (modulation depth `mu`, DT scale coefficients
`beta`, exclusion ratio `chi`) drift with time. We learn a Kolmogorov–Arnold
Network (KAN) controller that selects these parameters online to maximize the
total secret-key rate under QBER / sift / Eve-error / BSA-detectability
constraints, and extract closed-form design rules from the trained splines.

Target venue: high IEEE transaction (TCOM / JLT / TWC).

## Status

| Phase | Content | State |
|------|---------|-------|
| 0 | Analytical DT/DD environment + Monte-Carlo validator | **done (this commit)** |
| 1 | Time-varying LEO pass from TLE + handover | todo |
| 2 | Constrained optimization + dataset generation | todo |
| 3 | KAN controller + surrogate (GPU) + design-rule extraction | todo |

## Layout

```
docs/formulation.md     equation-by-equation derivation (transaction style)
src/satqkd/
  params.py             system parameters (Table I)
  geometry.py           Gaussian-beam spreading, A0, attenuation (Eq. 4-10)
  turbulence.py         Hufnagel-Valley Cn2, sigma_X^2, Gauss-Hermite (Eq. 12-13,19)
  detection.py          DT/DD sift/QBER/Eve-error in reduced (gamma,beta) form (Eq. 16-22)
  link.py               noise budget + LinkState assembly (Eq. 14)
  montecarlo.py         Monte-Carlo validator
scripts/smoke_test.py   CPU sanity + analytical-vs-MC cross-check
```

## Run

```bash
pip install -r requirements.txt
python scripts/smoke_test.py        # exit 0 = all checks pass
```

The smoke test confirms (1) the log-normal Gauss-Hermite expectation has unit
mean, (2) analytical sift/QBER/Eve-error match Monte-Carlo, (3) Eve error rises
to 0.5 as `mu -> 0` (the security lever), and (4) a `LinkState` builds from
geometry.

> **Calibration note.** The reduced model exposes a single peak-SNR ratio
> `gamma0` (at `mu=1`, beam center). The physically-derived value depends on
> [P3] Table I quantities (B0, Fn, sun irradiance, apertures) still being
> transcribed; `build_link_state(..., gamma0_override=...)` lets the environment
> run with a calibrated SNR while keeping the geometric/turbulence structure
> intact. See `# TODO calibrate vs [P3] Table I` markers in `params.py`.

## Collaboration workflow

PTIT: physical channel model + QBER/key-rate + Monte-Carlo. This repo / Hao:
optimization layer, KAN controller, time-varying & handover framework.
Code is smoke-tested locally (CPU), pushed here; KAN training runs on the RTX
4090 server.