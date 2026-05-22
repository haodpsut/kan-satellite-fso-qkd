"""System parameters for the GEO/LEO multi-user satellite FSO/QKD model.

Defaults follow the lineage papers [P1] IEEE Access 2020, [P2] IEEE Access 2022,
[P3] IEEE Photonics J. 2023. Values marked ``# TODO calibrate vs [P3] Table I``
are physically reasonable but should be checked against the printed table once
the PDF table is transcribed; they affect absolute SNR, not the model structure.
"""
from __future__ import annotations

from dataclasses import dataclass

# Physical constants (SI)
Q_E = 1.602176634e-19          # electron charge [C]
K_B = 1.380649e-23             # Boltzmann constant [J/K]
H_PLANCK = 6.62607015e-34      # Planck constant [J*s]
C_LIGHT = 2.99792458e8         # speed of light [m/s]


@dataclass
class SystemParams:
    """All static system parameters. Lengths in metres, time in seconds."""

    # --- Optical / link ---
    wavelength: float = 1550e-9        # lambda [m]
    peak_power: float = 0.68           # P [W]  (680 mW, [P1])
    responsivity: float = 0.8          # Re [A/W]
    amp_gain_db: float = 30.0          # Ga [dB] EDFA at LEO relay
    bit_rate: float = 1e9              # Rb [b/s]
    optical_bandwidth: float = 125e9   # B0 [Hz]            # TODO calibrate vs [P3] Table I
    nsp: float = 2.0                   # spontaneous-emission factor
    noise_figure_db: float = 4.0       # Fn [dB]            # TODO calibrate

    # --- Geometry / altitudes [m] ---
    H_geo: float = 35_786e3            # GEO altitude
    H_leo: float = 550e3               # LEO altitude (Starlink shell)
    H_user: float = 0.0                # ground user altitude
    H_atm: float = 20e3                # top of attenuating layer (Beer-Lambert)

    # --- Apertures / divergence ---
    D_geo_tx: float = 0.3              # GEO transmit telescope diameter [m]
    D_leo_tx: float = 0.3              # LEO transmit telescope diameter [m]
    a_user: float = 0.1               # user receiver aperture radius [m]
    a_leo: float = 0.1                # LEO receiver aperture radius [m]

    # --- Receiver electronics ---
    temperature: float = 300.0         # T [K]
    load_resistance: float = 50.0      # RL [ohm]

    # --- Atmosphere ---
    visibility_km: float = 23.0        # V clear weather [km]
    wind_rms: float = 21.0             # w [m/s], Hufnagel-Valley
    Cn2_ground: float = 1.7e-14        # Cn^2(0) [m^-2/3]
    sun_irradiance: float = 1e-3       # Sun spectral irradiance [W/(m^2 nm)] # TODO calibrate

    # --- Numerics ---
    gh_order: int = 20                 # Gauss-Hermite nodes ([P2]: n=20 converges)

    # ---------- derived ----------
    @property
    def amp_gain(self) -> float:
        return 10 ** (self.amp_gain_db / 10.0)

    @property
    def noise_figure(self) -> float:
        return 10 ** (self.noise_figure_db / 10.0)

    @property
    def delta_f(self) -> float:
        """Effective electrical bandwidth Delta f = Rb / 2."""
        return self.bit_rate / 2.0

    @property
    def wave_number(self) -> float:
        return 2.0 * 3.141592653589793 / self.wavelength