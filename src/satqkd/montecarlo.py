"""Monte-Carlo validator for the analytical DT/DD formulas.

Mirrors the M-C procedure of [P1] Fig. 7 / [P3] Sec. V-A: generate bits, apply
log-normal turbulence + AWGN, detect with the DT rule, and count
sift/error/Eve-error events.  Used to confirm Eq. (17-22).
"""
from __future__ import annotations

import numpy as np


def mc_single_link(gamma0: float, mu: float, beta: float, sigma_X: float,
                   n_bits: int = 2_000_000, gamma0_eve: float | None = None,
                   seed: int | None = 0):
    """Monte-Carlo sift prob, QBER, and (optionally) Eve error for one link.

    Signal currents are normalised by the noise std N (so N=1, s = gamma = gamma0*mu).
    Turbulence h_a ~ log-normal, unit mean. Decision currents: i = +/- gamma*h_a + n,
    n ~ N(0,1). Thresholds d0 = -gamma_bar - beta, d1 = +gamma_bar + beta where
    gamma_bar = gamma (mean, since E[h_a]=1).
    """
    rng = np.random.default_rng(seed)
    gamma = gamma0 * mu

    # transmitted bits (equiprobable)
    bits = rng.integers(0, 2, size=n_bits)
    # log-normal turbulence, unit mean: h_a = exp(2X), X~N(mu_X, sigma_X^2), mu_X=-sigma_X^2
    X = rng.normal(loc=-sigma_X ** 2, scale=sigma_X, size=n_bits)
    h_a = np.exp(2.0 * X)
    noise = rng.normal(0.0, 1.0, size=n_bits)

    signal = np.where(bits == 1, gamma * h_a, -gamma * h_a)
    received = signal + noise

    gamma_bar = gamma  # mean signal level used to place thresholds
    d1 = gamma_bar + beta
    d0 = -gamma_bar - beta

    decided_one = received >= d1
    decided_zero = received <= d0
    sifted = decided_one | decided_zero          # 'X' otherwise

    p_sift = sifted.mean()
    # error among sifted bits
    decoded = np.where(decided_one, 1, np.where(decided_zero, 0, -1))
    sift_mask = sifted
    errors = (decoded[sift_mask] != bits[sift_mask]).sum()
    qber = errors / max(sift_mask.sum(), 1)

    result = {"p_sift": float(p_sift), "qber": float(qber)}

    if gamma0_eve is not None:
        gamma_E = gamma0_eve * mu
        signal_E = np.where(bits == 1, gamma_E * h_a, -gamma_E * h_a)
        recv_E = signal_E + rng.normal(0.0, 1.0, size=n_bits)
        eve_dec = (recv_E >= 0.0).astype(int)     # optimal threshold d_E = 0
        result["eve_error"] = float((eve_dec != bits).mean())

    return result
