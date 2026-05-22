"""Link budget: assemble the noise variance and the per-link SNR ratio gamma
from physical parameters + geometry + turbulence.

Bridges geometry.py / turbulence.py / params.py into the (gamma, beta, sigma_X)
interface consumed by detection.py.  See docs/formulation.md Section 3 (Eq. 14).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SystemParams, Q_E, K_B, H_PLANCK, C_LIGHT
from .geometry import deterministic_channel, geometric_offset_factor
from .turbulence import sigma_X2


def noise_std(p: SystemParams, h_bar: float) -> float:
    """Total receiver noise std N^U (sqrt of Eq. 14).

    Components: shot, background (LEO+user), ASE, thermal. ``h_bar`` is the
    deterministic end-to-end gain h_g*h_l at the user.
    """
    df = p.delta_f
    Ga = p.amp_gain
    # zeta relating optical bandwidth to background collection [m^2-less factor]
    zeta = p.optical_bandwidth * p.wavelength ** 2 / (2.0 * C_LIGHT)
    # background optical power at user / leo apertures
    Pb_user = p.sun_irradiance * p.a_user ** 2 * zeta
    Pb_leo = p.sun_irradiance * p.a_leo ** 2 * zeta
    # mean received signal optical power scale (1/4 P Ga h_bar)
    P_sig = 0.25 * p.peak_power * Ga * h_bar
    # ASE noise power
    Pa_leo = (H_PLANCK * C_LIGHT / p.wavelength) * (p.nsp - 1.0) * Ga * p.optical_bandwidth

    var_shot = 2.0 * Q_E * p.responsivity * P_sig * df
    var_bg_user = 2.0 * Q_E * p.responsivity * Pb_user * df
    var_bg_leo = 2.0 * Q_E * p.responsivity * Pb_leo * Ga * h_bar * df
    var_ase = 2.0 * Q_E * Pa_leo * h_bar * df
    var_th = 4.0 * K_B * p.temperature * p.noise_figure / p.load_resistance * df

    var_total = var_shot + var_bg_user + var_bg_leo + var_ase + var_th
    return float(np.sqrt(var_total))


@dataclass
class LinkState:
    """Channel state at one time instant, summarising everything the optimizer
    needs.  gamma at a given modulation depth mu is gamma0 * mu."""
    zenith_rad: float
    sigma_X: float
    h_bar: float
    w_eq: float
    gamma0: float          # peak SNR ratio at mu=1, beam center (Bob/Alice)

    def gamma(self, mu: float) -> float:
        return self.gamma0 * mu

    def gamma_eve(self, mu: float, d_eve: float) -> float:
        """Eve at radial offset d_eve from footprint center (Eq. 8 + 16)."""
        leak = geometric_offset_factor(d_eve, self.w_eq)
        return self.gamma0 * mu * leak


def build_link_state(zenith_deg: float, p: SystemParams,
                     gamma0_override: float | None = None) -> LinkState:
    """Construct the LinkState for a given zenith angle [deg].

    ``gamma0_override`` lets the smoke test / calibration pin the peak SNR
    directly (decoupling the detection math from uncertain Table I values such
    as B0, Fn, sun irradiance) while keeping the geometric/turbulence structure.
    """
    zr = np.deg2rad(zenith_deg)
    sx2 = sigma_X2(zr, p)
    sigma_X = float(np.sqrt(max(sx2, 0.0)))
    h_bar, w_eq = deterministic_channel(zr, p)
    if gamma0_override is not None:
        gamma0 = float(gamma0_override)
    else:
        N = noise_std(p, h_bar)
        gamma0 = 0.25 * p.responsivity * p.peak_power * p.amp_gain * h_bar / N
    return LinkState(zenith_rad=zr, sigma_X=sigma_X, h_bar=h_bar, w_eq=w_eq, gamma0=gamma0)
