"""Link budget: assemble the noise variance and the per-link SNR ratio gamma
from physical parameters + geometry + turbulence.

Bridges geometry.py / turbulence.py / params.py into the (gamma, beta, sigma_X)
interface consumed by detection.py.  See docs/formulation.md Section 3 (Eq. 14).

gamma0 calibration
------------------
With [P3] Table I, the physically-derived peak SNR is link-budget-limited
(gamma0 ~ 0.01 at zenith 0: a 10 urad GEO beam spreads to ~350 m over 35,000 km,
so a 10 cm LEO aperture collects ~1e-7, plus daytime background). The paper,
however, operates at an effective gamma0 ~ 1.8 (back-solved from [P3] Sec. V-B:
mu < 0.7 keeps Eve error > 0.1 at worst-case zenith 0 => Q(0.7*gamma0)=0.1).
We therefore *anchor* gamma0 to that value while letting it scale with geometry
through the physical ratio s/N, so the zenith / turbulence dependence stays
paper-faithful and only the absolute scale is calibrated once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import SystemParams, Q_E, K_B, H_PLANCK, C_LIGHT
from .geometry import deterministic_channel, geometric_offset_factor
from .turbulence import sigma_X2


def noise_std(p: SystemParams, h_bar: float) -> float:
    """Total receiver noise std N^U (sqrt of Eq. 14).

    Components: shot, background (LEO relay + user), ASE, thermal. ``h_bar`` is
    the deterministic end-to-end gain h_g*h_l at the user.
    """
    df = p.delta_f
    Ga = p.amp_gain
    Re = p.responsivity
    # wavelength bandwidth (half) in metres: Delta_lambda/2 = B0 lambda^2 / (2 c)
    zeta = p.optical_bandwidth * p.wavelength ** 2 / (2.0 * C_LIGHT)
    # background optical powers: Pb = irradiance[W/(m^2 m)] * a^2[m^2] * zeta[m]
    Pb_user = p.sun_irr_earth * p.a_user ** 2 * zeta      # collected at user
    Pb_leo = p.sun_irr_atm * p.a_leo ** 2 * zeta          # collected at LEO relay
    # mean received signal optical power scale at the user (1/4 P Ga h_bar)
    P_sig = 0.25 * p.peak_power * Ga * h_bar
    # ASE noise power generated at the LEO EDFA
    Pa_leo = (H_PLANCK * C_LIGHT / p.wavelength) * (p.nsp - 1.0) * Ga * p.optical_bandwidth

    var_shot = 2.0 * Q_E * Re * P_sig * df
    var_bg_user = 2.0 * Q_E * Re * Pb_user * df
    var_bg_leo = 2.0 * Q_E * Re * Pb_leo * Ga * h_bar * df   # relayed + 2nd-hop loss
    var_ase = 2.0 * Q_E * Pa_leo * h_bar * df               # relayed + 2nd-hop loss
    var_th = 4.0 * K_B * p.temperature * p.noise_figure / p.load_resistance * df

    var_total = var_shot + var_bg_user + var_bg_leo + var_ase + var_th
    return float(np.sqrt(var_total))


def physical_gamma0(zenith_deg: float, p: SystemParams):
    """Raw, uncalibrated peak SNR ratio gamma0 = (1/4 Re P Ga h_bar)/N at mu=1,
    beam center.  Returns (gamma0, h_bar, w_eq, sigma_X)."""
    zr = np.deg2rad(zenith_deg)
    sx2 = sigma_X2(zr, p)
    sigma_X = float(np.sqrt(max(sx2, 0.0)))
    h_bar, w_eq = deterministic_channel(zr, p)
    N = noise_std(p, h_bar)
    gamma0 = 0.25 * p.responsivity * p.peak_power * p.amp_gain * h_bar / N
    return gamma0, h_bar, w_eq, sigma_X


def calibration_scale(p: SystemParams) -> float:
    """Multiplicative factor k so that k * physical_gamma0(zenith_ref) = gamma0_ref.
    Returns 1.0 if no calibration anchor is set."""
    if p.calib_gamma0_ref is None:
        return 1.0
    g_ref_phys, *_ = physical_gamma0(p.calib_zenith_ref_deg, p)
    if g_ref_phys <= 0:
        return 1.0
    return p.calib_gamma0_ref / g_ref_phys


@dataclass
class LinkState:
    """Channel state at one time instant, summarising everything the optimizer
    needs.  gamma at a given modulation depth mu is gamma0 * mu."""
    zenith_rad: float
    sigma_X: float
    h_bar: float
    w_eq: float
    gamma0: float          # peak SNR ratio at mu=1, beam center (Bob/Alice)

    @property
    def zenith_deg(self) -> float:
        return float(np.rad2deg(self.zenith_rad))

    def gamma(self, mu: float) -> float:
        return self.gamma0 * mu

    def gamma_eve(self, mu: float, d_eve: float) -> float:
        """Eve at radial offset d_eve from footprint center (Eq. 8 + 16)."""
        leak = geometric_offset_factor(d_eve, self.w_eq)
        return self.gamma0 * mu * leak


def build_link_state(zenith_deg: float, p: SystemParams,
                     gamma0_override: float | None = None,
                     calibrate: bool = True) -> LinkState:
    """Construct the LinkState for a given zenith angle [deg].

    Parameters
    ----------
    gamma0_override : pin gamma0 directly (bypasses physics+calibration).
    calibrate : if True (default), scale the physical gamma0 by the Table-I
        anchor (see module docstring); if False, use the raw physical value.
    """
    zr = np.deg2rad(zenith_deg)
    g_phys, h_bar, w_eq, sigma_X = physical_gamma0(zenith_deg, p)
    if gamma0_override is not None:
        gamma0 = float(gamma0_override)
    elif calibrate:
        gamma0 = float(calibration_scale(p) * g_phys)
    else:
        gamma0 = float(g_phys)
    return LinkState(zenith_rad=zr, sigma_X=sigma_X, h_bar=h_bar, w_eq=w_eq, gamma0=gamma0)
