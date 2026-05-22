"""Link geometry: Gaussian-beam radius, collected-power fraction A0, equivalent
beam radius, atmospheric attenuation, and the deterministic channel gain.

See docs/formulation.md Section 2.1-2.2 (Eq. 4-10).
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from .params import SystemParams


def beam_radius(distance: float, divergence: float, wavelength: float) -> float:
    """Gaussian beam radius w_L at link distance (Eq. 5).

    ``divergence`` is the (full) beam divergence angle theta [rad], given directly
    in [P3] Table I. Waist w0 = lambda / (pi theta); in the far field
    w_L -> theta * distance.
    """
    w0 = wavelength / (np.pi * divergence)
    return w0 * np.sqrt(1.0 + (wavelength * distance / (np.pi * w0 ** 2)) ** 2)


def collected_fraction(distance: float, divergence: float, aperture_radius: float,
                       wavelength: float):
    """Return (A0, w_eq) for a link (Eq. 6-7).

    A0   : fraction of power collected at the footprint center (r=0).
    w_eq : equivalent beam radius governing the off-center decay exp(-2 r^2 / w_eq^2).
    """
    w_L = beam_radius(distance, divergence, wavelength)
    upsilon = np.sqrt(np.pi) * aperture_radius / (np.sqrt(2.0) * w_L)
    A0 = erf(upsilon) ** 2
    w_eq2 = w_L ** 2 * (np.sqrt(np.pi) * erf(upsilon)) / (2.0 * upsilon * np.exp(-upsilon ** 2))
    return A0, np.sqrt(w_eq2)


def geometric_offset_factor(r: float, w_eq: float) -> float:
    """Off-center geometric leakage exp(-2 r^2 / w_eq^2) (Eq. 8). r=0 -> 1."""
    return float(np.exp(-2.0 * r ** 2 / w_eq ** 2))


def attenuation_coeff(p: SystemParams) -> float:
    """Beer-Lambert attenuation coefficient sigma [1/m] (Eq. 9-10)."""
    V = p.visibility_km
    lam_nm = p.wavelength * 1e9
    if V > 50.0:
        q = 1.6
    elif V > 6.0:
        q = 1.3
    else:
        q = 0.585 * V ** (1.0 / 3.0)
    sigma_per_km = 3.912 / V * (lam_nm / 550.0) ** (-q)
    return sigma_per_km / 1000.0  # per metre


def deterministic_channel(zenith_rad: float, p: SystemParams):
    """Deterministic end-to-end gain h_g * h_l at the user (beam center) and the
    equivalent beam radius of the LEO->user link (for Eve offset).

    Returns (h_bar, w_eq_leo_user).
    """
    # GEO -> LEO geometric (user/LEO at center): A0 of the GEO link.
    L_geo_leo = (p.H_geo - p.H_leo) / np.cos(zenith_rad)
    A0_geo, _ = collected_fraction(L_geo_leo, p.divergence_geo, p.a_leo, p.wavelength)

    # LEO -> user geometric.
    L_leo_user = (p.H_leo - p.H_user) / np.cos(zenith_rad)
    A0_leo, w_eq = collected_fraction(L_leo_user, p.divergence_leo, p.a_user, p.wavelength)

    # Atmospheric attenuation (LEO -> user, through the H_atm layer).
    L_atm = (p.H_atm - p.H_user) / np.cos(zenith_rad)
    h_l = np.exp(-attenuation_coeff(p) * L_atm)

    h_bar = A0_geo * A0_leo * h_l
    return float(h_bar), float(w_eq)
