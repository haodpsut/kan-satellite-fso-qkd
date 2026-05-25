# kan-satellite-fso-qkd

**KAN-based adaptive parameter control for multi-user satellite FSO/QKD systems**, with a Walker-Starlink TLE/SGP4 propagator for realistic-orbit validation.

This repository is the companion code to the paper

> H.~Do-Phuc *et~al.*, *"KAN-Based Adaptive Parameter Control for Multi-User Satellite FSO/QKD Systems"*, in preparation for IEEE TCOM, 2026.

It implements, from scratch, every numerical result in the paper: the analytical SIM-BPSK/DT-DD environment, the decomposed solver for the joint multi-user problem (P), the two-stage KAN/MLP/Linear controller (with residual learning and the $\beta$ safety-margin recipe), the synthetic Walker-shell + real-TLE Phase-1 path, and the per-step adaptive-vs-static evaluator.

---

## Reproducibility map

Every figure and table in the paper is produced by a single script in this repository:

| Paper artefact | Script / source                                                          | Output file(s)                                                          |
| -------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| Fig. 1 (arch)  | `paper/figures/tikz_arch.tex` (TikZ)                                     | rendered inline                                                         |
| Fig. 2 (decomp)| `paper/figures/tikz_decomp.tex` (TikZ)                                   | rendered inline                                                         |
| Fig. 3 (ctrl)  | `paper/figures/tikz_controller.tex` (TikZ)                               | rendered inline                                                         |
| Fig. 4 (cluster compare) | `scripts/optimize_pass_cluster.py` + `paper/figures/make_figs.py` | `results/cluster_pass_analytic.csv` + `cluster_pass_tle_walker_long.csv`|
| Fig. 5 (TLE Walker coverage) | `scripts/tle_pass_demo.py`                                   | `results/tle_vs_analytic.txt`                                           |
| Fig. 6 (controller 5-axis) | `scripts/train_kan.py --seeds 5`                               | `results/kan_comparison.txt`                                            |
| Fig. 7 ($\beta$ margin sweep) | `scripts/train_kan.py --margin-sweep`                       | `results/margin_sweep.csv`                                              |
| Fig. 8 (single-link trace) | `scripts/optimize_pass.py`                                     | `results/optimize_pass.csv`                                             |
| Table I (system params) | `src/satqkd/params.py`                                            | static (paper-cited)                                                    |
| Table II (headline) | `scripts/optimize_pass.py` + `optimize_pass_cluster.py` (3 variants) | summary `.txt` files in `results/`                              |
| Table III (single-link controller) | `scripts/train_kan.py --seeds 5`                       | `results/kan_comparison.txt`                                            |
| Table IV (cluster controller) | `scripts/train_cluster.py --models kan,mlp,linear --seeds 5` | `results/cluster_controller.txt`                                        |
| Eq. (21) ($\beta^\star$ closed form) | `scripts/train_kan.py --symbolic`                        | KAN symbolic extraction, refit, formula printed                         |

The accompanying paper TeX source is in `../paper/`; running `make` there rebuilds the PDF from these CSVs.

---

## Quick start

### 1. Server-side (RTX 4090, conda-only)

```bash
git clone https://github.com/haodpsut/kan-satellite-fso-qkd.git
cd kan-satellite-fso-qkd
conda env create -f environment.yml
conda activate satqkd
pip install pykan --no-deps
python -m pytest -q                          # 37/37 should pass
```

See `docs/SETUP_CONDA.md` for the GPU PyTorch wheel selection (cu121 vs cu118).

### 2. Local CPU-only smoke test

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                 # Phase 0 analytical vs Monte-Carlo
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

# Phase 3 KAN training + 5-axis trade-off + symbolic extraction (~10 min)
python scripts/train_kan.py --seeds 5 --margin-sweep
```

### 4. Rebuild the paper

```bash
cd ../paper
make            # full pdflatex + bibtex + 2x pdflatex
```

---

## Repository layout

```
src/satqkd/
  params.py             system parameters (paper Table I)
  geometry.py           Gaussian-beam spreading, A0, attenuation (paper Eq. 4-10)
  turbulence.py         Hufnagel-Valley Cn2, sigma_X^2, Gauss-Hermite (paper Eq. 12-13, 19)
  detection.py          DT/DD sift/QBER/Eve-error in reduced (gamma,beta) form (paper Eq. 16-22)
  link.py               noise budget + LinkState assembly (paper Eq. 14)
  orbit.py              analytic circular-orbit LEO pass + handover
  orbit_tle.py          SGP4 propagator + Walker constellation + PTIT/Hanoi GS
  keyrate.py            mutual information + secret-key rate (paper Eq. 25-29)
  optimize.py           per-state constrained optimization of (mu, beta)
  mu_solver.py          decomposed solver for cluster problem (P) (paper Algorithm 1)
  multiuser.py          cluster key rate + exclusion field + BSA detection
  mldata.py             dataset assembly for the supervised controller
  montecarlo.py         Monte-Carlo validator for the analytical reduction

scripts/
  smoke_test.py             CPU sanity check (analytical vs Monte-Carlo)
  calibrate_check.py        Table-I gamma0 calibration vs paper operating point
  simulate_pass.py          Phase 1: QKD metrics over a pass (--tle [PATH])
  tle_pass_demo.py          Phase 1: TLE vs analytic side-by-side + plot
  make_synthetic_tle.py     regenerate data/tle/starlink_synth.tle
  optimize_pass.py          Phase 2: single-link adaptive vs best-static
  generate_dataset.py       Phase 2: supervised dataset for the single-link controller
  train_kan.py              Phase 3: KAN vs MLP/KNN/Linear, 5-axis + symbolic
  optimize_pass_cluster.py  Phase 4: cluster adaptive vs best-static (--tle [PATH])
  generate_dataset_cluster.py  Phase 4: cluster supervised dataset
  train_cluster.py          Phase 4: two-stage KAN/MLP/Linear cluster controller

data/tle/starlink_synth.tle  bundled 50-sat synthetic Walker shell (reproducible offline)
docs/formulation.md          equation-by-equation derivation (transaction style)
docs/decomposition.md        formal derivation of the joint (P) and its decomposition
docs/SETUP_CONDA.md          server-side conda env setup notes
tests/                       37 pytest tests covering every module
results/                     committed CSVs and .txt summaries (paper-cited)
```

---

## Collaboration workflow

The work is a joint effort between Da Nang Architecture University (DAU) and the Posts and Telecommunications Institute of Technology (PTIT), Hanoi. PTIT provides the physical channel model and the BBM92-style QKD security analysis on which we build; DAU contributes the optimization layer (problem (P), decomposition, oracle, controller) and the time-varying / handover framework with the TLE/SGP4 path.

The development workflow is: code is smoke-tested locally on CPU and pushed here; the RTX 4090 server pulls and runs the GPU training phases (Phase 3 KAN, Phase 4 cluster controllers) and pushes the resulting CSVs / summaries back. The companion paper in `../paper/` then renders every figure from those committed result files.

---

## Status

| Phase | Content                                                                       | State   |
|-------|-------------------------------------------------------------------------------|---------|
| 0     | Analytical DT/DD environment + Monte-Carlo validator                          | done    |
| 1     | Time-varying LEO pass + handover (analytic + SGP4/TLE Walker)                 | done    |
| 2     | Constrained optimization of $(\mu,\beta)$ + dataset generation                | done    |
| 3     | Single-link KAN controller vs baselines + closed-form $\beta$ rule + margin   | done    |
| 4     | Multi-user joint problem (P) + decomposition + two-stage controller           | done    |

Headline numbers from the latest server runs (commit `b6dd8b9`):

* Single-link analytic (Table II row 1): **2.07$\times$ key**, 3.07$\times$ secure time.
* Cluster analytic (Table II row 2): **1.64$\times$ key**, 2.5$\times$ secure pair-time.
* Cluster TLE Walker $n{=}500$, 90 min (Table II row 3): **2.10$\times$ key**, 3.6$\times$ secure pair-time (5 $\to$ 18 secure pair-steps).
* Cluster TLE Walker $n{=}200$ sparse (Table II row 4): static collapses to 0 secure pair-steps; adaptive recovers 2/87.

Single-link controller comparison (Table III): KAN $0.716 \pm 0.123$ key retention, $0.011 \pm 0.002$ MAE on $\beta^\star$, 392 parameters, closed-form rule $\beta^\star \approx 1.13 + 0.42\,\gamma_0 - 2.81\,\sigma_X$ with $R^2 = 0.99$. MLP is statistically tied on key retention while using $3.3\times$ more parameters.

---

## Citing

If you use this code, please cite the paper above. A BibTeX entry will be added to the paper repository once the manuscript is on arXiv.

## Licence

The code is released under the MIT licence (see `LICENSE`). The bundled TLE data is synthetic and is offered under CC0.
