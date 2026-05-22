"""DT/DD decision statistics in the reduced (gamma, beta) form.

This is the pure-math core that the optimizer / KAN controller acts on. Every
function maps directly to docs/formulation.md Section 4-6 (Eq. 16-22).

    gamma   = s/N  = (1/4) mu Re P Ga h_bar / N      (normalized mean SNR amplitude, prop. to mu)
    beta    = DT scale coefficient
    sigma_X = log-amplitude std (sqrt of Eq. 12)
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc

from .turbulence import gauss_hermite_lognormal


def qfunc(x: np.ndarray) -> np.ndarray:
    """Gaussian Q-function Q(x) = 0.5 * erfc(x / sqrt(2))."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def p_correct(gamma: float, beta: float, sigma_X: float, expect) -> float:
    """P_corr = E_ha[ Q( beta + gamma (1 - h_a) ) ]  (Eq. 17)."""
    return float(expect(lambda ha: qfunc(beta + gamma * (1.0 - ha)), sigma_X))


def p_error(gamma: float, beta: float, sigma_X: float, expect) -> float:
    """P_err = E_ha[ Q( beta + gamma (1 + h_a) ) ]  (Eq. 18)."""
    return float(expect(lambda ha: qfunc(beta + gamma * (1.0 + ha)), sigma_X))


def sift_qber(gamma: float, beta: float, sigma_X: float, expect):
    """Return (P_sift, QBER) for a single Charlie->user link (Eq. 20-21)."""
    pc = p_correct(gamma, beta, sigma_X, expect)
    pe = p_error(gamma, beta, sigma_X, expect)
    psift = pc + pe
    qber = pe / psift if psift > 0 else np.nan
    return psift, qber


def eve_error(gamma_E: float, sigma_X: float, expect) -> float:
    """Eve's URA error probability with optimal threshold d_E=0 (Eq. 22).

    P_err^E = E_ha[ Q( gamma_E * h_a ) ] -> 0.5 as gamma_E -> 0 (mu -> 0).
    """
    return float(expect(lambda ha: qfunc(gamma_E * ha), sigma_X))


def make_expectation(gh_order: int = 20):
    """Convenience: build the Gauss-Hermite log-normal expectation operator."""
    return gauss_hermite_lognormal(gh_order)
