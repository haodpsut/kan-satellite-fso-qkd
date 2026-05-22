"""pytest for the key-rate model and the per-state optimizer."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import (
    SystemParams, build_link_state, make_expectation,
    binary_entropy, secret_fraction, normalized_key_rate,
)
from satqkd.optimize import optimize_state, evaluate


def test_binary_entropy_bounds():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)


def test_secret_fraction_positive_when_eve_worse():
    # Bob good (QBER tiny), Eve bad (Pe high) -> positive secret fraction
    assert secret_fraction(1e-4, 0.4) > 0
    # Eve as good as Bob -> no secret
    assert secret_fraction(1e-4, 1e-4) == 0.0


def test_normalized_key_rate_zero_if_insecure():
    assert normalized_key_rate(0.1, 0.4, 0.4) == 0.0  # QBER==Pe -> 0


@pytest.fixture(scope="module")
def setup():
    p = SystemParams()
    return p, make_expectation(p.gh_order)


def test_optimizer_secure_near_overhead(setup):
    p, expect = setup
    ls = build_link_state(0.0, p)        # overhead, gamma0 ~ 1.8
    opt = optimize_state(ls, expect, d_eve=26.0)
    assert opt.feasible
    assert opt.r_norm > 0
    # achieved metrics satisfy the constraints
    assert opt.qber < 1e-3 and opt.p_sift > 1e-3 and opt.eve_error > 0.1


def test_optimizer_beats_or_matches_arbitrary_static(setup):
    p, expect = setup
    ls = build_link_state(10.0, p)
    opt = optimize_state(ls, expect, d_eve=26.0)
    _, _, _, r_static = evaluate(0.5, 3.0, ls, expect, 26.0)
    assert opt.r_norm >= r_static - 1e-12


def test_closer_eve_lowers_optimal_mu(setup):
    # When Eve is very close the Eve-error constraint should bind, pushing mu*
    # no higher than when Eve is far.
    p, expect = setup
    ls = build_link_state(0.0, p)
    opt_near = optimize_state(ls, expect, d_eve=2.0)
    opt_far = optimize_state(ls, expect, d_eve=50.0)
    assert opt_near.mu <= opt_far.mu + 1e-9
