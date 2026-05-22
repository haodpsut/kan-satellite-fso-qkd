"""Atmospheric turbulence: Hufnagel-Valley Cn^2 profile, log-amplitude variance
sigma_X^2 (Eq. 12-13), and the log-normal Gauss-Hermite expectation (Eq. 19).

See docs/formulation.md Section 2.3 and 4.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .params import SystemParams


def cn2_hufnagel_valley(h: np.ndarray, wind_rms: float, cn2_ground: float) -> np.ndarray:
    """Refractive-index structure parameter Cn^2(h) [m^-2/3], H-V model (Eq. 13).

    h : altitude above ground [m] (array or scalar).
    """
    h = np.asarray(h, dtype=float)
    term1 = 0.00594 * (wind_rms / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
    term2 = 2.7e-16 * np.exp(-h / 1500.0)
    term3 = cn2_ground * np.exp(-h / 100.0)
    return term1 + term2 + term3


def sigma_X2(zenith_rad: float, p: SystemParams, n_steps: int = 2000) -> float:
    """Log-amplitude variance sigma_X^2 (Eq. 12), numerically integrated.

    sigma_X^2 = 0.56 k^{7/6} sec^{11/6}(theta) * int_{H_user}^{H_geo} Cn2(h)(h-H_user)^{5/6} dh

    The integrand decays fast with altitude, so the integral is dominated by the
    lower atmosphere; we integrate up to the top of the turbulent layer.
    """
    sec = 1.0 / np.cos(zenith_rad)
    k = p.wave_number
    h_top = min(p.H_geo, 30e3)  # turbulence negligible above ~30 km
    h = np.linspace(p.H_user + 1.0, h_top, n_steps)
    integrand = cn2_hufnagel_valley(h, p.wind_rms, p.Cn2_ground) * (h - p.H_user) ** (5.0 / 6.0)
    integral = np.trapezoid(integrand, h)
    return 0.56 * k ** (7.0 / 6.0) * sec ** (11.0 / 6.0) * integral


def gauss_hermite_lognormal(order: int):
    """Return (h_a_nodes_fn, weights) helper for E_{h_a}[g(h_a)] with E[h_a]=1.

    Returns a callable ``expect(g, sigma_X)`` computing the unit-mean log-normal
    expectation via Eq. (19):
        E[g(h_a)] ~ (1/sqrt(pi)) sum_j w_j g( exp(2*sqrt(2)*sigma_X*x_j - 2*sigma_X^2) ).
    ``g`` must be vectorised over an array of h_a values.
    """
    x, w = hermgauss(order)              # nodes/weights for weight e^{-x^2}
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)

    def expect(g, sigma_X: float) -> np.ndarray:
        sigma_X = max(sigma_X, 1e-9)
        h_a = np.exp(2.0 * np.sqrt(2.0) * sigma_X * x - 2.0 * sigma_X ** 2)
        vals = g(h_a)                    # shape (..., order) via broadcasting
        return inv_sqrt_pi * np.tensordot(vals, w, axes=([-1], [0]))

    return expect
