"""pytest for the multi-user cluster model (exclusion closed form + reductions)."""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from satqkd import SystemParams, build_link_state, make_expectation
from satqkd.multiuser import (
    NodeState, exclusion_term, cluster_key_rate, link_stats,
)


def brute_exclusion(pc_A, pc_i, pc_others):
    """Direct inclusion-exclusion sum over non-empty subsets (ground truth)."""
    total = 0.0
    n = len(pc_others)
    for r in range(1, n + 1):
        for S in combinations(range(n), r):
            prod = pc_A * pc_i
            for j in S:
                prod *= pc_others[j]
            total += (-1) ** (r + 1) * prod
    return total


@pytest.mark.parametrize("N", [1, 2, 4, 6])
def test_exclusion_closed_form_matches_bruteforce(N):
    rng = np.random.default_rng(N)
    pc_A = float(rng.uniform(0.1, 0.6))
    pc = rng.uniform(0.05, 0.5, size=N)
    for i in range(N):
        others = [float(pc[j]) for j in range(N) if j != i]
        cf = exclusion_term(pc_A, float(pc[i]), others)
        bf = brute_exclusion(pc_A, float(pc[i]), others)
        assert abs(cf - bf) < 1e-12


def test_single_user_no_exclusion():
    assert exclusion_term(0.4, 0.3, []) == 0.0


@pytest.fixture(scope="module")
def setup():
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    alice = build_link_state(10.0, p)
    bobs = [build_link_state(z, p) for z in (5.0, 12.0, 18.0, 25.0)]
    to_node = lambda ls: NodeState(ls.gamma0, ls.sigma_X, ls.w_eq)
    return expect, to_node(alice), [to_node(b) for b in bobs]


def test_chi_zero_disables_exclusion(setup):
    expect, alice, bobs = setup
    betas = [1.0] * len(bobs)
    r0 = cluster_key_rate(0.5, 1.0, betas, 0.0, alice, bobs, expect)
    r1 = cluster_key_rate(0.5, 1.0, betas, 1.0, alice, bobs, expect)
    # with chi=0 every P_chi equals the raw pairwise sift (no exclusion)
    # exclusion can only reduce P_chi, so chi=1 total <= chi=0 total
    assert r1["total_rate"] <= r0["total_rate"] + 1e-9


def test_cluster_rate_positive(setup):
    expect, alice, bobs = setup
    betas = [1.5] * len(bobs)
    res = cluster_key_rate(0.5, 1.5, betas, 0.5, alice, bobs, expect)
    assert res["total_rate"] > 0
    assert 0 <= res["n_feasible"] <= len(bobs)


def test_cluster_feasible_region_exists():
    # The multi-user feasible region (pair QBER<1e-3 AND P_chi>1e-3 AND leak<=tau
    # AND BSA detectable) is tight because pair sift is a product of two link
    # sifts and BSA imposes a lower beta bound; verify the solver still finds a
    # non-empty feasible operating point for a near-overhead cluster.
    from satqkd import solve_cluster
    p = SystemParams()
    expect = make_expectation(p.gh_order)
    to = lambda ls: NodeState(ls.gamma0, ls.sigma_X, ls.w_eq)
    alice = to(build_link_state(2.0, p))
    bobs = [to(build_link_state(z, p)) for z in (0.0, 3.0)]
    sol = solve_cluster(alice, bobs, expect, d_eve=26.0)
    assert sol.n_feasible >= 1
    assert sol.total_rate > 0
    # BSA pushed beta off its floor (two-sided window), not at beta_lo=0.3
    assert sol.beta_A > 0.5
