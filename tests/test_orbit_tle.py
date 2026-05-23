"""pytest for the TLE-driven orbit / handover model (Phase 1 paper-faithful path).

Skipped automatically when ``skyfield``/``sgp4`` are not installed, so the rest
of the test suite still runs on the analytic-only environment.
"""
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("skyfield")
pytest.importorskip("sgp4")

from satqkd.orbit_tle import (  # noqa: E402
    GroundStation, PTIT_HANOI, TLEPass, TLEConstellation,
    starlink_shell, example_tle_constellation, tle_constellation,
)


T0 = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)


def test_starlink_shell_returns_satrecs():
    sats = starlink_shell(n_sats=5)
    assert len(sats) == 5
    for name, sat in sats:
        assert isinstance(name, str)
        # Satrec basic sanity: positive mean motion (rad/min)
        assert sat.no_kozai > 0.0


def test_tle_pass_elevation_bounded():
    sats = starlink_shell(n_sats=1)
    sp = TLEPass(satrec=sats[0][1], ground=PTIT_HANOI, t0_utc=T0, sat_id=0,
                 name=sats[0][0])
    t = np.arange(0.0, 6000.0, 30.0)
    elev = sp.elevation(t)
    assert elev.shape == t.shape
    # Elevation is always in [-90, +90] for an Earth-bound observer/satellite.
    assert (elev >= -90.0).all() and (elev <= 90.0).all()


def test_tle_pass_zenith_and_visibility():
    sats = starlink_shell(n_sats=10)
    const = tle_constellation(sats, ground=PTIT_HANOI, t0_utc=T0,
                              min_elevation=30.0)
    t = np.arange(0.0, 3600.0, 30.0)
    sched = const.schedule(t)
    # Some satellites should be visible during a 1-hour window over Hanoi.
    served = (sched["sat_id"] >= 0).sum()
    assert served > 0, "expected at least one served step over a 1-hour window"
    # Where served, zenith = 90 - elevation in [0, 60] (visibility floor at 30 deg)
    mask = np.isfinite(sched["zenith_deg"])
    assert (sched["zenith_deg"][mask] <= 60.0 + 1e-6).all()
    assert (sched["zenith_deg"][mask] >= 0.0 - 1e-6).all()
    # elev + zenith ~ 90
    np.testing.assert_allclose(
        sched["elevation_deg"][mask] + sched["zenith_deg"][mask], 90.0, atol=1e-6)


def test_tle_constellation_produces_handovers():
    """A multi-satellite shell should hand over from one sat to another at least
    once during a 1-hour window over a low-latitude site."""
    const = example_tle_constellation(n_sats=10, min_elevation=30.0)
    sched = const.schedule(np.arange(0.0, 3600.0, 30.0))
    # Either inter-satellite handovers or acquire/drop transitions.
    assert len(sched["handovers"]) >= 1


def test_ground_station_default_is_ptit():
    assert PTIT_HANOI.lat_deg == pytest.approx(21.0050, abs=1e-4)
    assert PTIT_HANOI.lon_deg == pytest.approx(105.8434, abs=1e-4)


def test_naive_t0_raises():
    sats = starlink_shell(n_sats=1)
    naive = datetime(2026, 5, 23, 12, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        TLEPass(satrec=sats[0][1], ground=PTIT_HANOI, t0_utc=naive)
