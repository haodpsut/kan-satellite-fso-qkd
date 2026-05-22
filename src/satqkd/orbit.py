"""Time-varying LEO geometry: satellite passes, elevation/zenith vs time, and
handover across a constellation serving one ground region.

This drives the *time-varying environment*: as a LEO traverses its pass the
zenith angle (hence gamma0, sigma_X, w_eq) changes, and the serving satellite
switches at handover. We use a dependency-free analytic circular-orbit pass
model so the environment runs offline; a TLE/skyfield path can be layered on
later for paper-faithful reproduction of [P3] Fig. 6-8.

Spherical-geometry relations (ground station U, satellite S, Earth center O):
    r  = R_e + h                                  (orbital radius)
    d  = sqrt(R_e^2 + r^2 - 2 R_e r cos(theta))    (slant range; theta = central angle)
    sin(E) = (r cos(theta) - R_e) / d              (elevation above horizon)
    zenith = 90 - E
At theta=0 (overhead): E=90 deg, zenith=0.  Visibility requires E >= E_min (30 deg
=> zenith <= 60 deg), matching [P3]'s minimum acceptable elevation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EARTH_RADIUS = 6_371e3        # mean Earth radius [m]
MU_EARTH = 3.986004418e14     # Earth gravitational parameter [m^3/s^2]


def orbital_rate(altitude: float) -> float:
    """Mean orbital angular rate n = sqrt(mu / r^3) [rad/s] for a circular orbit."""
    r = EARTH_RADIUS + altitude
    return float(np.sqrt(MU_EARTH / r ** 3))


def elevation_deg(central_angle: float, altitude: float):
    """Return (elevation_deg, slant_range_m) for a geocentric central angle [rad]."""
    r = EARTH_RADIUS + altitude
    theta = np.asarray(central_angle, dtype=float)
    d = np.sqrt(EARTH_RADIUS ** 2 + r ** 2 - 2.0 * EARTH_RADIUS * r * np.cos(theta))
    sin_E = (r * np.cos(theta) - EARTH_RADIUS) / d
    sin_E = np.clip(sin_E, -1.0, 1.0)
    return np.rad2deg(np.arcsin(sin_E)), d


@dataclass
class SatellitePass:
    """One LEO pass over the ground region.

    t0         : time of closest approach [s].
    theta_min  : minimum central angle at closest approach [rad] (0 = overhead).
    altitude   : orbit altitude [m].
    sat_id     : label.
    """
    t0: float
    theta_min: float
    altitude: float = 550e3
    sat_id: int = 0

    def central_angle(self, t):
        """theta(t) ~ hypot(theta_min, n*(t-t0)) for a near-straight ground track."""
        n = orbital_rate(self.altitude)
        return np.hypot(self.theta_min, n * (np.asarray(t, dtype=float) - self.t0))

    def elevation(self, t):
        return elevation_deg(self.central_angle(t), self.altitude)[0]

    def zenith(self, t):
        return 90.0 - self.elevation(t)

    def visible(self, t, min_elevation: float = 30.0):
        return self.elevation(t) >= min_elevation


@dataclass
class Constellation:
    """A set of LEO passes serving one region; provides the serving satellite
    (highest elevation among visible) at each time, i.e. the handover policy."""
    passes: list[SatellitePass] = field(default_factory=list)
    min_elevation: float = 30.0

    def serving(self, t: float):
        """Return (sat_id, zenith_deg, elevation_deg) of the best visible satellite
        at time t, or (None, nan, nan) during a coverage gap."""
        best = (None, np.nan, np.nan)
        best_elev = self.min_elevation
        for sp in self.passes:
            e = float(sp.elevation(t))
            if e >= best_elev:
                best_elev = e
                best = (sp.sat_id, 90.0 - e, e)
        return best

    def schedule(self, t_grid):
        """Vectorised serving schedule over a time grid.

        Returns dict with arrays: ``t``, ``sat_id`` (-1 in gaps), ``zenith_deg``,
        ``elevation_deg``, and a list of ``handovers`` as (time, from_id, to_id).
        """
        t_grid = np.asarray(t_grid, dtype=float)
        sat_id = np.full(t_grid.shape, -1, dtype=int)
        zenith = np.full(t_grid.shape, np.nan)
        elev = np.full(t_grid.shape, np.nan)
        # elevation of every satellite over the grid
        all_elev = np.stack([sp.elevation(t_grid) for sp in self.passes], axis=0) \
            if self.passes else np.empty((0, t_grid.size))
        for i in range(t_grid.size):
            col = all_elev[:, i]
            j = int(np.argmax(col)) if col.size else -1
            if col.size and col[j] >= self.min_elevation:
                sat_id[i] = self.passes[j].sat_id
                elev[i] = col[j]
                zenith[i] = 90.0 - col[j]
        handovers = []
        for i in range(1, t_grid.size):
            if sat_id[i] != sat_id[i - 1]:
                handovers.append((float(t_grid[i]), int(sat_id[i - 1]), int(sat_id[i])))
        return {"t": t_grid, "sat_id": sat_id, "zenith_deg": zenith,
                "elevation_deg": elev, "handovers": handovers}


def example_constellation(n_sats: int = 5, spacing_s: float = 500.0,
                          altitude: float = 550e3, theta_min_deg=0.0) -> Constellation:
    """A staggered constellation: passes whose closest approach times are spaced
    ``spacing_s`` apart, giving overlapping visibility windows + handovers over a
    multi-hundred-second horizon (cf. [P3] Fig. 8).

    ``theta_min_deg`` may be a scalar or a per-satellite iterable of off-zenith
    closest-approach angles (deg) to make passes non-overhead/realistic.
    """
    if np.isscalar(theta_min_deg):
        theta_min_deg = [theta_min_deg] * n_sats
    passes = []
    for k in range(n_sats):
        passes.append(SatellitePass(
            t0=spacing_s * (k + 1),
            theta_min=np.deg2rad(float(theta_min_deg[k])),
            altitude=altitude,
            sat_id=k,
        ))
    return Constellation(passes=passes)
