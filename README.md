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
| 2 | Constrained optimization (Eq. 26) + dataset generation | **done** |
| 3 | KAN controller vs baselines (multi-axis) + design-rule extraction | **in progress** (scaffold + MLP/KNN/Linear baselines local; KAN runs on the 4090) |

Phase 1 result (`scripts/simulate_pass.py`): the static baseline (mu=0.5,
beta=3.0) is operational 100% of served time but strictly secure (QBER<1e-3)
only ~1.4% over a pass.

Phase 2 result (`scripts/optimize_pass.py`): per-step optimal (mu,beta) vs the
**best single static** (tuned for the whole pass), with the same operational
feasibility gate on both. Adaptive achieves **2.07x secret key** and **3x
secure time** (86 vs 28 served steps) over the best static — the gain the KAN
controller targets. `scripts/generate_dataset.py` produces the supervised set
`(zenith, gamma0, sigma_X, w_eq, d_eve) -> (mu*, beta*)` for Phase 3; d_eve is a
feature so mu* is non-trivial (it drops when Eve is close enough to bind the
Eve-error constraint).

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
  keyrate.py            mutual information + secret-key rate (Eq. 25-29)
  optimize.py           per-state constrained optimization of (mu,beta) (Eq. 26)
  montecarlo.py         Monte-Carlo validator
scripts/
  smoke_test.py         CPU sanity + analytical-vs-MC cross-check
  calibrate_check.py    Table-I gamma0 calibration + operating-point check
  simulate_pass.py      Phase 1: QKD metrics over a LEO pass with handover
  optimize_pass.py      Phase 2: adaptive vs best-static over a pass
  generate_dataset.py   Phase 2: supervised dataset for the KAN controller
  train_kan.py          Phase 3: KAN vs MLP/KNN/Linear, multi-axis trade-off
```

### Phase 3 evaluation philosophy

The KAN controller is **not** expected to dominate every axis. We report a
multi-axis trade-off (regression accuracy, closed-loop secret-key retention,
parameter count, inference latency, interpretability) and analyse it honestly.

Findings so far:
- **Fair baselines matter.** An unregularized MLP overfits the small dataset
  (key retention 0.31); with weight decay + early stopping it jumps to ~0.74 and
  is competitive with / better than KAN on closed-loop key. We therefore report
  the *regularized* MLP as the baseline.
- **MAE != what matters.** KNN wins MAE but not key retention; a 10-parameter
  Linear model loses on MAE yet retains more key than raw MAE would suggest.
- **Feasibility-boundary sensitivity.** Because QBER<1e-3 is a hard gate, small
  beta errors flip a state infeasible and forfeit its key, so key retention sits
  well below what the high R^2 implies. A beta safety margin is a natural fix.
- **KAN's value is interpretability + parameter efficiency**, not necessarily
  raw accuracy. Closed-form rules require the full pykan recipe (prune ->
  auto_symbolic -> **refit**); skipping the refit collapses the formula to a
  constant.
- **Report mean +/- std over seeds.** MLP closed-loop key retention has high
  run-to-run variance (~0.66 +/- 0.10 over seeds; GPU LBFGS / init noise), so a
  single KAN run is statistically indistinguishable from MLP on key retention.
  Use `--seeds N`. The honest read is: KAN ~ MLP on key retention, while KAN
  wins MAE, feasibility count, and parameter count, and yields a clean closed-form
  rule for beta (R^2~0.99) but **not** for mu (mu* is near-saturated with a sharp
  Eve-driven transition; symbolic R^2 < 0), so KAN is only *partially*
  interpretable here.

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