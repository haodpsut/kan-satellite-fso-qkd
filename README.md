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
| 0 | Analytical DT/DD environment + Monte-Carlo validator | **done** |
| 1 | Time-varying LEO pass + handover (analytic orbit; TLE optional later) | **done** |
| 2 | Constrained optimization + dataset generation | todo |
| 3 | KAN controller + surrogate (GPU) + design-rule extraction | todo |

Phase 1 result (`scripts/simulate_pass.py`): over a 3000 s, 10-LEO scenario the
static baseline (mu=0.5, beta=3.0) is **operational 100%** of served time but
strictly **secure (QBER<1e-3) only ~1.4%** — quantifying why a static parameter
choice wastes most of a pass and motivating the adaptive controller.

## Layout

```
docs/formulation.md     equation-by-equation derivation (transaction style)
src/satqkd/
  params.py             system parameters (Table I)
  geometry.py           Gaussian-beam spreading, A0, attenuation (Eq. 4-10)
  turbulence.py         Hufnagel-Valley Cn2, sigma_X^2, Gauss-Hermite (Eq. 12-13,19)
  detection.py          DT/DD sift/QBER/Eve-error in reduced (gamma,beta) form (Eq. 16-22)
  link.py               noise budget + LinkState assembly (Eq. 14)
  orbit.py              LEO pass elevation/zenith vs time + handover
  montecarlo.py         Monte-Carlo validator
scripts/
  smoke_test.py         CPU sanity + analytical-vs-MC cross-check
  calibrate_check.py    Table-I gamma0 calibration + operating-point check
  simulate_pass.py      Phase 1: QKD metrics over a LEO pass with handover
```

## Run

On the RTX 4090 server (conda-only): see [docs/SETUP_CONDA.md](docs/SETUP_CONDA.md).

```bash
conda env create -f environment.yml && conda activate satqkd
python scripts/smoke_test.py        # exit 0 = all checks pass
```

CPU-only / local with pip:

```bash
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test confirms (1) the log-normal Gauss-Hermite expectation has unit
mean, (2) analytical sift/QBER/Eve-error match Monte-Carlo, (3) Eve error rises
to 0.5 as `mu -> 0` (the security lever), and (4) a `LinkState` builds from
geometry.

> **Calibration note.** `params.py` now carries the real [P3] Table I values.
> The physically-derived peak SNR is link-budget-limited (`gamma0 ~ 0.01` at
> zenith 0: a 10 urad GEO beam spreads to ~350 m over 35,000 km), so `gamma0` is
> *anchored* to the paper's operating point (`gamma0_ref ~ 1.8` at zenith 0,
> back-solved from [P3] Sec. V-B) and then scaled by the physical `s/N` ratio,
> keeping the zenith/turbulence dependence paper-faithful. Verify with
> `python scripts/calibrate_check.py` (reproduces Eve-error and sift/QBER
> operating points). Set `calib_gamma0_ref=None` for the raw physical value.

## Collaboration workflow

PTIT: physical channel model + QBER/key-rate + Monte-Carlo. This repo / Hao:
optimization layer, KAN controller, time-varying & handover framework.
Code is smoke-tested locally (CPU), pushed here; KAN training runs on the RTX
4090 server.