"""pytest mirror of the core smoke-test invariants."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation, sift_qber, eve_error
from satqkd.montecarlo import mc_single_link


@pytest.fixture(scope="module")
def expect():
    return make_expectation(20)


@pytest.mark.parametrize("sigma_X", [0.05, 0.1, 0.2, 0.3])
def test_lognormal_unit_mean(expect, sigma_X):
    mean = float(expect(lambda ha: ha, sigma_X))
    assert abs(mean - 1.0) < 1e-6


@pytest.mark.parametrize("beta", [1.5, 2.5, 3.5])
def test_analytical_matches_montecarlo(expect, beta):
    gamma0, mu, sigma_X, gamma0_eve = 12.0, 0.5, 0.1, 3.0
    psift_a, _ = sift_qber(gamma0 * mu, beta, sigma_X, expect)
    eve_a = eve_error(gamma0_eve * mu, sigma_X, expect)
    mc = mc_single_link(gamma0, mu, beta, sigma_X, n_bits=3_000_000,
                        gamma0_eve=gamma0_eve, seed=1)
    assert abs(psift_a - mc["p_sift"]) < 5e-3 + 0.05 * psift_a
    assert abs(eve_a - mc["eve_error"]) < 5e-3


def test_eve_error_monotone_in_mu(expect):
    gamma0_eve, sigma_X = 3.0, 0.1
    mus = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    eves = [eve_error(gamma0_eve * m, sigma_X, expect) for m in mus]
    assert all(eves[i] >= eves[i + 1] - 1e-9 for i in range(len(eves) - 1))
    assert eves[0] > 0.4  # -> 0.5 as mu -> 0


def test_linkstate_build():
    p = SystemParams()
    ls = build_link_state(40.0, p, gamma0_override=12.0)
    assert ls.sigma_X > 0 and ls.h_bar > 0 and ls.w_eq > 0