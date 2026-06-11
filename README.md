# kan-satellite-fso-qkd

**KAN-based adaptive parameter control for multi-user satellite FSO/QKD systems**, with a Walker-Starlink TLE/SGP4 propagator for realistic-orbit validation.

This repository is the companion code to the paper

> P. H. Do, M. Q. Vu, and N. T. Dang, *"KAN-Based Adaptive Parameter Control for Multi-User Satellite FSO/QKD Systems,"* in preparation for IEEE Transactions on Communications, 2026.

It implements, from scratch, every numerical result in the paper: the analytical SIM-BPSK/DT-DD environment, the decomposed solver for the joint multi-user problem **(P)**, the two-stage KAN/MLP/Linear controller (with residual learning and the $\beta$ safety-margin recipe), the synthetic Walker-shell + real-TLE propagator path, and the per-step adaptive-vs-static evaluator.

---

## Reproducibility map

Every figure and table in the paper is produced by a script in this repository. Figures 1-3 are TikZ flow diagrams authored in the paper source; Figures 4-8 are plotted by `paper/figures/make_figs.py` from the committed result files.

| Paper artefact | Script / source | Output file(s) |
| --- | --- | --- |
| Fig. 1 (system architecture) | `paper/figures/tikz_arch.tex` (TikZ) | rendered inline |
| Fig. 2 (decomposed solver) | `paper/figures/tikz_decomp.tex` (TikZ) | rendered inline |
| Fig. 3 (two-stage controller) | `paper/figures/tikz_controller.tex` (TikZ) | rendered inline |
| Fig. 4 (adaptive vs static, cluster) | `scripts/optimize_pass_cluster.py` → `make_figs.py` | `results/cluster_pass_analytic.csv`, `cluster_pass_tle_walker_long.csv` |
| Fig. 5 (TLE-Walker coverage) | `scripts/tle_pass_demo.py` (swept over `--tle-n`) → `make_figs.py` | `results/tle_vs_analytic.txt`; sweep values embedded in `make_figs.py` |
| Fig. 6 (controller five-axis) | `scripts/train_kan.py --seeds 5` | `results/kan_comparison.txt` |
| Fig. 7 ($\beta$ margin sweep) | `scripts/train_kan.py --margin-sweep` | `results/margin_sweep.csv` |
| Fig. 8 (single-link trace) | `scripts/optimize_pass.py` | `results/optimize_pass.csv` |
| Table I (system parameters) | `src/satqkd/params.py` | static (paper-cited) |
| Table II (headline adaptive vs static) | `scripts/optimize_pass.py` + `optimize_pass_cluster.py` (4 variants) | `*_summary.txt` in `results/` |
| Table III (single-link controller) | `scripts/train_kan.py --seeds 5` | `results/kan_comparison.txt` |
| Table IV (cluster controller) | `scripts/train_cluster.py --models kan,mlp,linear --seeds 5` | `results/cluster_controller.txt` |
| Eq. (21) ($\beta^\star$ closed form) | `scripts/train_kan.py --symbolic` | symbolic formula printed in `results/kan_comparison.txt` |

> **Note on Fig. 5.** `tle_vs_analytic.txt` holds the analytic baseline (46.8 %) and the $n{=}200$ Walker point (49.7 %) that anchors the "approaches the baseline" claim. The full coverage curve ($n\in\{50,100,200,500\}$) aggregates separate `tle_pass_demo.py` runs at each shell size; those served-time / count values are currently embedded at the top of `fig_tle_walker_coverage()` in `make_figs.py`. Re-running `tle_pass_demo.py --tle-n N` for each `N` reproduces them.

The paper TeX source is in `paper/`; running `make` there rebuilds the PDF from these result files.

---

## Quick start

### 1. Server-side (GPU, conda)

```bash
git clone https://github.com/haodpsut/kan-satellite-fso-qkd.git
cd kan-satellite-fso-qkd
conda env create -f environment.yml
conda activate satqkd
pip install pykan --no-deps
python -m pytest -q
```

See `docs/SETUP_CONDA.md` for the GPU PyTorch wheel selection (cu121 vs cu118). The GPU is needed only for the KAN training phases (Phase 3 and Phase 4); everything else runs on a single CPU core.

### 2. Local CPU-only smoke test

```bash
pip install -r requirements.txt
python scripts/smoke_test.py        # Phase 0: analytical vs Monte-Carlo
python -m pytest -q
```

### 3. Reproduce the headlines

```bash
# Phase 2 single-link (analytic, ~5 min)
python scripts/optimize_pass.py

# Phase 4 cluster (analytic, ~15 min)
python scripts/optimize_pass_cluster.py

# Phase 4 cluster (TLE Walker n=500, 90-min window; ~20 min)
python scripts/optimize_pass_cluster.py --tle --tle-n 500 --horizon 5400 \
                                        --tle-epoch 2026-05-23T12:00:00

# Phase 1 TLE-vs-analytic side-by-side
python scripts/tle_pass_demo.py --horizon 7200 --tle-n 200

# Phase 3 KAN training + five-axis trade-off + symbolic extraction (~10 min)
python scripts/train_kan.py --seeds 5 --margin-sweep
```

### 4. Rebuild the paper

```bash
cd paper
make            # pdflatex + bibtex + 2x pdflatex
```

---

## Repository layout

```
src/satqkd/
  params.py        system parameters (paper Table I)
  geometry.py      Gaussian-beam spreading, A0, attenuation (paper Eq. 4-10)
  turbulence.py    Hufnagel-Valley Cn2, sigma_X^2, Gauss-Hermite (paper Eq. 12-13, 19)
  detection.py     DT/DD sift/QBER/Eve-error in reduced (gamma,beta) form (paper Eq. 16-22)
  link.py          noise budget + LinkState assembly (paper Eq. 14)
  orbit.py         analytic circular-orbit LEO pass + handover
  orbit_tle.py     SGP4 propagator + Walker constellation + PTIT/Hanoi GS
  keyrate.py       mutual information + secret-key rate (paper Eq. 25-29)
  optimize.py      per-state constrained optimization of (mu, beta)
  mu_solver.py     decomposed solver for cluster problem (P) (paper Algorithm 1)
  multiuser.py     cluster key rate + exclusion field + BSA detection
  mldata.py        dataset assembly for the supervised controller
  montecarlo.py    Monte-Carlo validator for the analytical reduction

scripts/
  smoke_test.py                CPU sanity check (analytical vs Monte-Carlo)
  calibrate_check.py           Table-I gamma0 calibration vs paper operating point
  simulate_pass.py             Phase 1: QKD metrics over a pass (--tle [PATH])
  tle_pass_demo.py             Phase 1: TLE vs analytic side-by-side
  make_synthetic_tle.py        regenerate data/tle/starlink_synth.tle
  optimize_pass.py             Phase 2: single-link adaptive vs best-static
  generate_dataset.py          Phase 2: supervised dataset (single-link controller)
  train_kan.py                 Phase 3: KAN vs MLP/KNN/Linear, five-axis + symbolic
  optimize_pass_cluster.py     Phase 4: cluster adaptive vs best-static (--tle [PATH])
  generate_dataset_cluster.py  Phase 4: cluster supervised dataset
  train_cluster.py             Phase 4: two-stage KAN/MLP/Linear cluster controller
  smoke_mu_solver.py           decomposed-solver consistency check (mean-field vs brute force)

data/tle/starlink_synth.tle    bundled synthetic Walker shell (reproducible offline)
docs/formulation.md            equation-by-equation derivation (transaction style)
docs/decomposition.md          formal derivation of the joint (P) and its decomposition
docs/SETUP_CONDA.md            server-side conda env setup notes
tests/                         pytest suite covering every module
results/                       committed CSVs and .txt summaries (paper-cited)
```

---

## Collaboration workflow

This is a joint effort between Da Nang Architecture University (DAU) and the Posts and Telecommunications Institute of Technology (PTIT), Hanoi. The PTIT line of work provides the SIM-BPSK/DT-DD physical channel model and the BBM92-style QKD security analysis on which we build; DAU contributes the optimization layer (problem **(P)**, the two-level decomposition, the oracle, and the learned controller) and the time-varying / handover framework with the TLE/SGP4 path.

Development workflow: code is smoke-tested locally on CPU and pushed here; the GPU server pulls and runs the training phases (Phase 3 KAN, Phase 4 cluster controllers) and pushes the resulting CSVs / summaries back. The companion paper in `paper/` then renders every data figure from those committed result files.

---

## Status

| Phase | Content | State |
| --- | --- | --- |
| 0 | Analytical DT/DD environment + Monte-Carlo validator | done |
| 1 | Time-varying LEO pass + handover (analytic + SGP4/TLE Walker) | done |
| 2 | Constrained optimization of $(\mu,\beta)$ + dataset generation | done |
| 3 | Single-link KAN controller vs baselines + closed-form $\beta$ rule + margin | done |
| 4 | Multi-user joint problem (P) + decomposition + two-stage controller | done |

**Headline adaptive-vs-static results (Table II):**

* Single-link analytic (row 1): **2.07× key**, 3.07× secure time.
* Cluster analytic (row 2): **1.64× key**, 2.5× secure pair-time.
* Cluster TLE-Walker $n{=}500$, 90 min (row 3): **2.10× key**, 3.6× secure pair-time (5 → 18 secure pair-steps).
* Cluster TLE-Walker $n{=}200$ sparse (row 4): static collapses to 0 secure pair-steps; adaptive recovers 2/87.

**Decomposed solver as oracle.** On a two-user cluster the decomposed solver value matches a brute-force joint grid over $(\beta_A,\beta_1,\beta_2)$ to four decimals (ratio 1.000; see `smoke_mu_solver.py` → `results/run_mu_solver.txt`). The closed-form exclusion field is separately checked against a direct subset-enumeration inclusion–exclusion sum for $N\in\{1,2,4,6\}$ to $10^{-12}$ (`tests/test_multiuser.py`). A brute-force optimality check for asymmetric $N>2$ clusters is left to future work.

**Single-link controller (Table III).** KAN reaches $0.716\pm0.123$ closed-loop key retention and $0.011\pm0.002$ MAE on $\beta^\star$ with 392 parameters; the regularized MLP is statistically tied on key retention ($0.683\pm0.086$) while using $3.3\times$ more parameters. The KAN predicts $\beta^\star$ with full-network regression $R^2=1.00\pm0.00$, and the symbolic extraction (`--symbolic`) yields a closed-form rule

```
beta* ~= 2.42 - 0.10*gamma0 - 0.03*sigma_X + 0.23*w_eq
         + 1.24*sin(0.50*sigma_X + 5.27)
         - 0.30*sin(0.48*w_eq + 2.04)
         - 0.17*tanh(9.2*d_E + 8.62)
```

with symbolic-fit $R^2\approx0.97$ (above the $0.95$ interpretability bar). The exact coefficients are seed-dependent; the formula above is the representative seed-0 extraction printed in `results/kan_comparison.txt`. The symbolic rule for $\mu^\star$ is reported but not deployed (it saturates near its upper bound except for a sharp Eve-driven transition the candidate library cannot fit reliably).

---

## Citing

If you use this code, please cite the paper above. A BibTeX entry will be added once the manuscript is on arXiv.

## Licence

The code is released under the MIT licence (see `LICENSE`). The bundled TLE data is synthetic and is offered under CC0.
