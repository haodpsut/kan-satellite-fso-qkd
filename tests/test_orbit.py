"""pytest for the time-varying LEO orbit / handover model."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd.orbit import (
    SatellitePass, Constellation, example_constellation, orbital_rate, elevation_deg,
)


def test_orbital_period_leo():
    # 550 km LEO period ~ 95 min
    n = orbital_rate(550e3)
    period_min = 2 * np.pi / n / 60.0
    assert 90 < period_min < 100


def test_overhead_pass_peaks_at_90():
    sp = SatellitePass(t0=1000.0, theta_min=0.0, altitude=550e3, sat_id=0)
    assert sp.elevation(1000.0) == pytest.approx(90.0, abs=1e-6)
    assert sp.zenith(1000.0) == pytest.approx(0.0, abs=1e-6)


def test_elevation_symmetric_and_decreasing():
    sp = SatellitePass(t0=1000.0, theta_min=np.deg2rad(10.0), altitude=550e3)
    e_before = sp.elevation(900.0)
    e_after = sp.elevation(1100.0)
    e_peak = sp.elevation(1000.0)
    assert e_peak > e_before
    assert e_before == pytest.approx(e_after, abs=1e-6)  # symmetric about t0


def test_visibility_window_finite():
    sp = SatellitePass(t0=1000.0, theta_min=0.0, altitude=550e3)
    t = np.arange(0.0, 2000.0, 5.0)
    vis = sp.visible(t, min_elevation=30.0)
    assert vis.any() and not vis.all()  # rises and sets


def test_handover_picks_max_elevation():
    const = example_constellation(n_sats=3, spacing_s=400.0, altitude=550e3,
                                  theta_min_deg=0.0)
    sched = const.schedule(np.arange(0.0, 2000.0, 5.0))
    # at least one handover, serving id changes over time
    assert len(sched["handovers"]) >= 1
    served = sched["sat_id"][sched["sat_id"] >= 0]
    assert served.size > 0


def test_elevation_zero_at_horizon():
    # central angle where r cos(theta) = R_e -> elevation 0
    from satqkd.orbit import EARTH_RADIUS
    r = EARTH_RADIUS + 550e3
    theta_h = np.arccos(EARTH_RADIUS / r)
    e, _ = elevation_deg(theta_h, 550e3)
    assert float(e) == pytest.approx(0.0, abs=1e-6)
