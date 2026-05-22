"""System parameters for the GEO/LEO multi-user satellite FSO/QKD model.

Values are transcribed from [P3] IEEE Photonics J. 2023, "TABLE I — SYSTEM
PARAMETERS" (the GEO/LEO multi-user architecture, our target system). See
docs/formulation.md for the equations that consume them.
"""
from __future__ import annotations

from dataclasses import dataclass

# Physical constants (SI)
Q_E = 1.602176634e-19          # electron charge [C]
K_B = 1.380649e-23             # Boltzmann constant [J/K]
H_PLANCK = 6.62607015e-34      # Planck constant [J*s]
C_LIGHT = 2.99792458e8         # speed of light [m/s]

# Sun spectral irradiance unit conversion: W/(cm^2 * um) -> W/(m^2 * m)
#   1 W/(cm^2 um) = 1 / (1e-4 m^2 * 1e-6 m) = 1e10 W/(m^2 m)
WCM2UM_TO_SI = 1e10


@dataclass
class SystemParams:
    """All static system parameters ([P3] Table I). Lengths in metres, time in s."""

    # --- Optical / link ([P3] Table I) ---
    wavelength: float = 1550e-9        # lambda [m]
    bit_rate: float = 1e9              # Rb = 1 Gbps
    peak_power_dbm: float = 32.0       # P at GEO = 32 dBm  (= 1.585 W)
    responsivity: float = 0.9          # Re [A/W]
    amp_gain_db: float = 40.0          # Ga EDFA at LEO relay = 40 dB
    nsp: float = 5.0                   # ASE spontaneous-emission factor
    optical_bandwidth: float = 250e9   # B0 [Hz] = 250 GHz
    noise_bandwidth: float = 0.5e9     # Delta f [Hz] = 0.5 GHz (= Rb/2)
    noise_figure: float = 2.0          # Fn (linear factor, NOT dB)

    # --- Geometry / altitudes [m] ([P3] Table I) ---
    H_geo: float = 35_793e3            # Charlie (GEO) altitude
    H_leo: float = 550e3               # LEO altitude (Starlink shell)
    H_user: float = 2.0                # Alice/Bob/Eve altitude
    H_atm: float = 20e3                # top of attenuating layer (Beer-Lambert)

    # --- Beam divergence (full angle) [rad] ([P3] Table I) ---
    divergence_geo: float = 10e-6      # theta_C = 10 urad
    divergence_leo: float = 50e-6      # theta_L = 50 urad

    # --- Receiver apertures (radius) [m] ([P3] Table I) ---
    a_leo: float = 0.10                # LEO receiving aperture radius = 10 cm
    a_user: float = 0.05               # user (A/B/E) receiving aperture radius = 5 cm

    # --- Receiver electronics ([P3] Table I) ---
    temperature: float = 298.0         # T [K]
    load_resistance: float = 1000.0    # RL = 1 kOhm

    # --- Atmosphere ([P3] Table I) ---
    visibility_km: float = 30.0        # V clear weather [km]
    wind_rms: float = 21.0             # w [m/s], Hufnagel-Valley
    Cn2_ground: float = 1e-15          # Cn^2(0) [m^-2/3]
    # Sun spectral irradiance [W/(cm^2 um)] -> stored SI in properties below
    sun_irr_atm_wcm2um: float = 0.1    # above atmosphere @1550nm (LEO background)
    sun_irr_earth_wcm2um: float = 0.005  # above Earth @1550nm (user background)

    # --- gamma0 calibration anchor (see docs/formulation.md / README) ---
    # Physically-derived gamma0 from this Table is link-budget-limited (~0.01);
    # we anchor it to the paper's operating point: [P3] Sec. V-B states mu<0.7
    # keeps Eve error >0.1 at worst-case zenith=0, i.e. Q(gamma0_ref*0.7)=0.1
    # => gamma0_ref ~= 1.8. Set calib_gamma0_ref=None to use the raw physical value.
    calib_gamma0_ref: float | None = 1.8
    calib_zenith_ref_deg: float = 0.0

    # --- Numerics ---
    gh_order: int = 20                 # Gauss-Hermite nodes ([P2]: n=20 converges)

    # ---------- derived ----------
    @property
    def peak_power(self) -> float:
        """P [W] from dBm."""
        return 10 ** (self.peak_power_dbm / 10.0) * 1e-3

    @property
    def amp_gain(self) -> float:
        return 10 ** (self.amp_gain_db / 10.0)

    @property
    def delta_f(self) -> float:
        """Effective electrical bandwidth Delta f [Hz] (Table I gives 0.5 GHz = Rb/2)."""
        return self.noise_bandwidth

    @property
    def wave_number(self) -> float:
        return 2.0 * 3.141592653589793 / self.wavelength

    @property
    def sun_irr_atm(self) -> float:
        """Above-atmosphere Sun spectral irradiance in SI W/(m^2 m)."""
        return self.sun_irr_atm_wcm2um * WCM2UM_TO_SI

    @property
    def sun_irr_earth(self) -> float:
        """Above-Earth Sun spectral irradiance in SI W/(m^2 m)."""
        return self.sun_irr_earth_wcm2um * WCM2UM_TO_SI
