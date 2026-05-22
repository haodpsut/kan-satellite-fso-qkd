"""Dataset loading / feature scaling / split for the controller learning.

Pure numpy (no sklearn) so it runs in any environment. The supervised set is
produced by scripts/generate_dataset.py:
    features = [gamma0, sigma_X, w_eq, d_eve]   (channel state + Eve assumption)
    labels   = [mu_opt, beta_opt]
Only feasible states carry meaningful labels, so by default we train on those.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ["gamma0", "sigma_X", "w_eq", "d_eve"]
LABELS = ["mu_opt", "beta_opt"]


def _read_csv(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    return cols


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse(self, Xn):
        return Xn * self.std + self.mean

    @classmethod
    def fit(cls, X):
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean, std)


@dataclass
class MLDataset:
    Xtr: np.ndarray
    ytr: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray
    xscale: Standardizer
    feature_names: list
    label_names: list
    # raw (unscaled) test features kept for closed-loop evaluation
    Xte_raw: np.ndarray
    # oracle normalized key on the test states (for closed-loop key retention)
    yte_rnorm: np.ndarray


def load_dataset(path, feasible_only: bool = True, test_frac: float = 0.2,
                 seed: int = 0) -> MLDataset:
    cols = _read_csv(path)
    mask = cols["feasible"] > 0.5 if feasible_only else np.ones_like(cols["feasible"], bool)
    X = np.stack([cols[f][mask] for f in FEATURES], axis=1)
    y = np.stack([cols[l][mask] for l in LABELS], axis=1)
    rnorm = cols["rnorm_opt"][mask]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_te = max(1, int(round(test_frac * X.shape[0])))
    te, tr = idx[:n_te], idx[n_te:]

    xscale = Standardizer.fit(X[tr])
    return MLDataset(
        Xtr=xscale.transform(X[tr]), ytr=y[tr],
        Xte=xscale.transform(X[te]), yte=y[te],
        xscale=xscale, feature_names=FEATURES, label_names=LABELS,
        Xte_raw=X[te], yte_rnorm=rnorm[te],
    )


def mae(pred, true):
    return np.abs(pred - true).mean(axis=0)


def r2_score(pred, true):
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    ss_tot[ss_tot == 0] = 1.0
    return 1.0 - ss_res / ss_tot
