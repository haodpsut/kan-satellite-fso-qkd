# Setup on the RTX 4090 server (conda-only)

You only have conda access — that's enough. We create one conda env and install
the GPU PyTorch + KAN inside it with pip (pip runs *inside* the env, no system
permissions needed).

## 1. Clone

```bash
git clone https://github.com/haodpsut/kan-satellite-fso-qkd.git
cd kan-satellite-fso-qkd
```

## 2. Create the environment

```bash
conda env create -f environment.yml      # conda deps + cu121 torch
conda activate satqkd
pip install pykan --no-deps              # pykan deps already came from conda
```

`pykan` is installed with `--no-deps` on purpose: its declared deps would drag
in a PyPI torch wheel built for CUDA 12.6+, which breaks on drivers capped at
12.5. All of pykan's runtime deps (scikit-learn, sympy, tqdm, seaborn, pyyaml,
numpy, matplotlib, pandas) are already provided by `environment.yml`.

If `conda env create` is slow resolving, enable libmamba first:

```bash
conda config --set solver libmamba       # one-time, optional
```

### Already created the env before this fix?

If you hit `No module named 'sklearn'` or "NVIDIA driver too old", repair in
place without recreating:

```bash
conda activate satqkd
conda install -c conda-forge scikit-learn sympy tqdm seaborn pyyaml -y
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pykan --no-deps
```

## 3. Verify

```bash
# Phase 0 — analytical + Monte-Carlo (CPU, ~2 s). Expect "RESULT: ALL PASS".
PYTHONIOENCODING=utf-8 python scripts/smoke_test.py

# pytest (9 tests)
python -m pytest -q

# GPU check (Phase 3) — expect: cuda True, device NVIDIA GeForce RTX 4090
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"

# KAN import check (Phase 3)
python -c "import kan; print('pykan OK')"
```

## Manual fallback (if `environment.yml` fails)

```bash
conda create -n satqkd python=3.11 -y
conda activate satqkd
conda install -c conda-forge numpy scipy matplotlib pandas pytest \
    scikit-learn sympy tqdm seaborn pyyaml skyfield sgp4 -y
# GPU PyTorch for RTX 4090. Use --index-url (not --extra) so pip stays on the
# cu121 channel; cu121 wheels only need driver CUDA >= 12.1.
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pykan --no-deps
```

### Picking the CUDA wheel

Check the driver's max CUDA version:

```bash
nvidia-smi          # top-right "CUDA Version: 12.x"
```

- driver CUDA >= 12.1  -> use `cu121` (default; works on the 12.5 server driver)
- driver CUDA 11.8     -> replace `cu121` with `cu118` in environment.yml, recreate
- CPU-only smoke test  -> `pip install torch` (no index-url); Phase 0 needs no GPU

> Why `--index-url`, not `--extra-index-url`: with `--extra-`, pip also sees the
> default PyPI index and prefers the newest torch there (built for CUDA 12.6+),
> which fails with "NVIDIA driver too old" on a driver capped at 12.5.

## Notes

- The repo runs straight from source (scripts add `src/` to `sys.path`); no
  `pip install -e .` required.
- Phase 0/1/2 (analytical, TLE, optimization) are CPU. Only Phase 3 (KAN
  training) uses the 4090.
- To remove the env: `conda env remove -n satqkd`.
