"""pytest for dataset loading + a trivial controller fit (no torch/pykan needed)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satqkd.mldata import load_dataset, mae, r2_score, FEATURES, LABELS

DATA = ROOT / "results" / "dataset.csv"


def test_dataset_loads_and_splits():
    if not DATA.exists():
        import pytest
        pytest.skip("dataset.csv not generated")
    ds = load_dataset(str(DATA), feasible_only=True, test_frac=0.2, seed=0)
    assert ds.Xtr.shape[1] == len(FEATURES)
    assert ds.ytr.shape[1] == len(LABELS)
    assert ds.Xte.shape[0] == ds.yte.shape[0] == ds.yte_rnorm.shape[0]
    # standardized train features ~ zero mean / unit std
    assert np.allclose(ds.Xtr.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(ds.Xtr.std(axis=0), 1.0, atol=1e-6)


def test_linear_fit_predicts_in_range():
    if not DATA.exists():
        import pytest
        pytest.skip("dataset.csv not generated")
    ds = load_dataset(str(DATA), feasible_only=True, test_frac=0.2, seed=0)
    A = np.hstack([ds.Xtr, np.ones((ds.Xtr.shape[0], 1))])
    W, *_ = np.linalg.lstsq(A, ds.ytr, rcond=None)
    pred = np.hstack([ds.Xte, np.ones((ds.Xte.shape[0], 1))]) @ W
    assert pred.shape == ds.yte.shape
    r2 = r2_score(np.clip(pred, [0.05, 0.3], [0.95, 4.5]), ds.yte)
    assert r2[1] > 0.5  # beta reasonably learnable even by a linear model
