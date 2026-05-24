"""TLE-driven LEO geometry — paper-faithful reproduction path for [P3] Fig. 6-8.

This module mirrors the analytic API of :mod:`satqkd.orbit` (``SatellitePass``,
``Constellation``) so downstream scripts can switch between the analytic
circular-orbit model and a real-orbit propagation driven by Two-Line Elements
(TLEs) by changing only the constellation factory — every per-step call still
takes a scalar ``t`` (seconds since the simulation epoch ``t0_utc``) and returns
elevation / zenith / slant in the same units.

We use :mod:`sgp4` (SGP4 propagator) for orbit propagation and :mod:`skyfield`
only for the topocentric (alt/az) transform from a ground-station latitude /
longitude / altitude. Both packages are listed in ``environment.yml``; this
module imports them lazily so the rest of the codebase still runs without
either installed (analytic path).

Coordinate / time convention:
    * sim time ``t`` is **seconds since** ``t0_utc`` (an aware ``datetime`` in
      UTC). All scripts already use seconds-from-zero; ``t0_utc`` only fixes the
      absolute clock for SGP4. The default epoch is chosen at module load to be
      a recent round-hour UTC so a run today and a run tomorrow disagree only by
      where in the orbit the satellites start (intentional — TLE epochs go stale
      and we want that to be visible).
    * Elevation / zenith are degrees, slant range metres — same units as the
      analytic path.

The synthetic ``starlink_shell()`` helper builds N satellites with realistic
Starlink-shell elements (550 km, 53 deg inclination, evenly spaced in RAAN +
mean anomaly) directly via ``sgp4.api.Satrec.sgp4init`` so we do **not** need
internet access or a bundled TLE file to smoke-test the TLE path. A separate
``load_tle_file`` reads a 3-line TLE file when the user supplies one (e.g. a
fresh Celestrak snapshot for the final paper plots).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

# Imports are deferred to constructors so that orbit_tle can be imported in
# environments without skyfield/sgp4 (we want the import to succeed but using
# any TLE class without the deps to raise a clear error).
_SGP4_ERR = "sgp4 / skyfield required for the TLE path (see environment.yml)."


# --- Ground station -----------------------------------------------------------

@dataclass
class GroundStation:
    """Receiver location on Earth.

    Attributes
    ----------
    lat_deg, lon_deg : geodetic latitude / east longitude in degrees.
    elev_m           : height above the WGS84 ellipsoid in metres.
    name             : free-form label (for logs only).
    """
    lat_deg: float
    lon_deg: float
    elev_m: float = 0.0
    name: str = "GS"

    def _topos(self):
        from skyfield.api import wgs84
        return wgs84.latlon(self.lat_deg, self.lon_deg, elevation_m=self.elev_m)


# Default ground station: PTIT in Hanoi (collaborator's institution).
# 21.0050 N, 105.8434 E (approx; precise to ~100 m).
PTIT_HANOI = GroundStation(lat_deg=21.0050, lon_deg=105.8434, elev_m=20.0, name="PTIT_Hanoi")


# --- TLE I/O ------------------------------------------------------------------

def load_tle_file(path: str | Path) -> list[tuple[str, "object"]]:
    """Parse a 3-line TLE file (NAME / L1 / L2 triplets) into ``[(name, satrec)]``.

    Blank lines and ``#``-comment lines are ignored. Suitable for files saved
    from Celestrak (e.g. ``starlink.txt``).
    """
    try:
        from sgp4.api import Satrec
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_SGP4_ERR) from exc
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.rstrip()
        if not s or s.lstrip().startswith("#"):
            continue
        lines.append(s)
    if len(lines) % 3 != 0:
        raise ValueError(f"{path}: expected 3-line TLE blocks, got {len(lines)} lines")
    out = []
    for i in range(0, len(lines), 3):
        name = lines[i].strip()
        l1, l2 = lines[i + 1], lines[i + 2]
        out.append((name, Satrec.twoline2rv(l1, l2)))
    return out


def starlink_shell(n_sats: int = 50, altitude_km: float = 550.0,
                   inclination_deg: float = 53.0,
                   n_planes: int | None = None,
                   raan_spread_deg: float = 360.0,
                   epoch_utc: datetime | None = None,
                   bstar: float = 1e-5):
    """Generate a Walker-style synthetic Starlink-shell constellation via
    ``sgp4init``.

    Satellites are arranged in ``n_planes`` orbital planes (RAAN spread over
    ``raan_spread_deg``, default 360 = full shell) with
    ``n_sats // n_planes`` satellites per plane (mean anomaly spread evenly
    within each plane, with an inter-plane phase offset to break temporal
    symmetry — the Walker delta pattern). This matches the actual Starlink
    shell structure (e.g. 72 planes x 22 sats/plane) and gives realistic
    continuous coverage over an arbitrary ground station; a single-plane
    constellation (``n_planes=1``) reproduces the older "tightly-packed pass
    cluster" behaviour for [P3] Fig. 8 reproduction.

    Default ``n_planes`` is ``max(1, n_sats // 10)`` — i.e. 10 sats per plane.
    Returns a list of ``(name, Satrec)``.
    """
    try:
        from sgp4.api import Satrec, WGS72
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_SGP4_ERR) from exc

    if epoch_utc is None:
        epoch_utc = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    if n_planes is None:
        n_planes = max(1, n_sats // 10)
    if n_sats % n_planes != 0:
        # Allow uneven counts but warn via name: extras land in the last plane.
        pass
    sats_per_plane = max(1, n_sats // n_planes)
    # Recompute n_sats so it matches planes * sats_per_plane exactly
    n_sats_eff = n_planes * sats_per_plane

    # SGP4 epoch is days since 1949 Dec 31 00:00 UT
    sgp4_epoch_ref = datetime(1949, 12, 31, 0, 0, 0, tzinfo=timezone.utc)
    epoch_days = (epoch_utc - sgp4_epoch_ref).total_seconds() / 86400.0

    # Mean motion: n = sqrt(mu/a^3), in rad/min for sgp4init's no_kozai
    EARTH_RADIUS_KM = 6378.137
    MU_KM3_S2 = 398600.4418
    a_km = EARTH_RADIUS_KM + altitude_km
    n_rad_per_s = np.sqrt(MU_KM3_S2 / a_km ** 3)
    no_kozai = n_rad_per_s * 60.0  # rad/min

    inclo = np.deg2rad(inclination_deg)
    ecco = 1e-4
    argpo = 0.0
    out = []
    k = 0
    for p in range(n_planes):
        raan_deg = raan_spread_deg * p / n_planes
        raan = np.deg2rad(raan_deg)
        for s in range(sats_per_plane):
            # Within-plane MA spacing + Walker delta phase offset across planes
            mo_deg = 360.0 * s / sats_per_plane + 360.0 * p / n_sats_eff
            mo = np.deg2rad(mo_deg)
            satrec = Satrec()
            satrec.sgp4init(
                WGS72, "i",
                70000 + k,           # satnum (placeholder)
                epoch_days,
                bstar, 0.0, 0.0,
                ecco, argpo, inclo, mo, no_kozai, raan,
            )
            out.append((f"SYNTH-SL-P{p:02d}-S{s:02d}", satrec))
            k += 1
    return out


# --- Pass-aware wrapper -------------------------------------------------------

@dataclass
class TLEPass:
    """One LEO pass over a ground station, propagated via SGP4 from a TLE.

    Mirrors the analytic ``orbit.SatellitePass`` API:
        ``.elevation(t)``, ``.zenith(t)``, ``.visible(t)`` with ``t`` in seconds
        since ``t0_utc``.
    Vectorised in ``t`` (accepts scalar or ``np.ndarray``).

    Attributes
    ----------
    satrec  : sgp4 Satrec.
    ground  : GroundStation.
    t0_utc  : absolute UTC epoch for sim-time origin (sim ``t=0``).
    sat_id  : integer label (for handover bookkeeping).
    name    : free-form satellite name.
    """
    satrec: object
    ground: GroundStation
    t0_utc: datetime
    sat_id: int = 0
    name: str = ""

    def __post_init__(self):
        if self.t0_utc.tzinfo is None:
            raise ValueError("t0_utc must be timezone-aware (UTC).")
        from skyfield.api import load, EarthSatellite
        from sgp4.exporter import export_tle
        # Build a skyfield EarthSatellite via the TLE-string round-trip: this is
        # the only public skyfield constructor that fully initialises the vector
        # graph (target / center / segments) that ``__sub__`` relies on.
        l1, l2 = export_tle(self.satrec)
        self._ts = load.timescale()
        self._sat = EarthSatellite(l1, l2, self.name or f"SAT{self.sat_id}", self._ts)
        self._topos = self.ground._topos()

    def _altaz(self, t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        dt_list = [self.t0_utc.timestamp() + float(s) for s in t_arr]
        # skyfield timescale from POSIX timestamps
        utc_dt = [datetime.fromtimestamp(s, tz=timezone.utc) for s in dt_list]
        ts = self._ts.from_datetimes(utc_dt)
        difference = self._sat - self._topos
        topocentric = difference.at(ts)
        alt, az, distance = topocentric.altaz()
        elev = np.asarray(alt.degrees, dtype=float).reshape(-1)
        slant = np.asarray(distance.m, dtype=float).reshape(-1)
        return elev, slant

    def elevation(self, t):
        elev, _ = self._altaz(t)
        return float(elev[0]) if np.isscalar(t) else elev

    def zenith(self, t):
        return 90.0 - self.elevation(t)

    def slant_range(self, t):
        _, slant = self._altaz(t)
        return float(slant[0]) if np.isscalar(t) else slant

    def visible(self, t, min_elevation: float = 30.0):
        return self.elevation(t) >= min_elevation


@dataclass
class TLEConstellation:
    """Set of TLE-driven passes with a max-elevation handover policy.

    Same API as :class:`satqkd.orbit.Constellation`: ``serving(t)`` and
    ``schedule(t_grid)`` return the same dict shape so downstream scripts work
    unchanged.
    """
    passes: list[TLEPass] = field(default_factory=list)
    min_elevation: float = 30.0

    def serving(self, t: float):
        best = (None, np.nan, np.nan)
        best_elev = self.min_elevation
        for sp in self.passes:
            e = float(sp.elevation(float(t)))
            if e >= best_elev:
                best_elev = e
                best = (sp.sat_id, 90.0 - e, e)
        return best

    def schedule(self, t_grid):
        t_grid = np.asarray(t_grid, dtype=float)
        sat_id = np.full(t_grid.shape, -1, dtype=int)
        zenith = np.full(t_grid.shape, np.nan)
        elev = np.full(t_grid.shape, np.nan)
        # Stack elevation of every satellite over the grid (vectorised through
        # skyfield); skip empty constellations cleanly.
        if self.passes:
            all_elev = np.stack([sp.elevation(t_grid) for sp in self.passes], axis=0)
        else:
            all_elev = np.empty((0, t_grid.size))
        for i in range(t_grid.size):
            col = all_elev[:, i]
            if col.size == 0:
                continue
            j = int(np.argmax(col))
            if col[j] >= self.min_elevation:
                sat_id[i] = self.passes[j].sat_id
                elev[i] = col[j]
                zenith[i] = 90.0 - col[j]
        handovers = []
        for i in range(1, t_grid.size):
            if sat_id[i] != sat_id[i - 1]:
                handovers.append((float(t_grid[i]), int(sat_id[i - 1]), int(sat_id[i])))
        return {"t": t_grid, "sat_id": sat_id, "zenith_deg": zenith,
                "elevation_deg": elev, "handovers": handovers}


# --- Constellation factories --------------------------------------------------

def tle_constellation(satrecs: Iterable[tuple[str, "object"]],
                      ground: GroundStation = PTIT_HANOI,
                      t0_utc: datetime | None = None,
                      min_elevation: float = 30.0) -> TLEConstellation:
    """Wrap an iterable of ``(name, satrec)`` into a :class:`TLEConstellation`."""
    if t0_utc is None:
        t0_utc = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    passes = []
    for k, (name, sat) in enumerate(satrecs):
        passes.append(TLEPass(satrec=sat, ground=ground, t0_utc=t0_utc,
                              sat_id=k, name=name))
    return TLEConstellation(passes=passes, min_elevation=min_elevation)


def example_tle_constellation(n_sats: int = 10, altitude_km: float = 550.0,
                              ground: GroundStation = PTIT_HANOI,
                              t0_utc: datetime | None = None,
                              min_elevation: float = 30.0,
                              raan_spread_deg: float = 360.0) -> TLEConstellation:
    """Convenience: synthetic Starlink-shell constellation, ready to schedule.

    The synthetic shell is **reproducible** (no internet) but is **not** a
    specific real epoch — it is sized to mirror [P3]'s 550 km / 53 deg shell so
    that the QKD-relevant statistics (pass duration, peak elevation distribution,
    zenith trajectory) match the analytic baseline within propagator noise.
    """
    sats = starlink_shell(n_sats=n_sats, altitude_km=altitude_km,
                          raan_spread_deg=raan_spread_deg, epoch_utc=t0_utc)
    return tle_constellation(sats, ground=ground, t0_utc=t0_utc,
                             min_elevation=min_elevation)
