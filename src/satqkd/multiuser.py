"""Multi-user cluster key rate with the fading-randomness multiple access (chi
exclusion), heterogeneous users, and URA threat.

Implements docs/decomposition.md Section 2 (Eq. 1-6). The exclusion term uses the
closed form (Eq. 3), generalizing [P3] Eq. 24 to an asymmetric cluster. BSA is
handled as a separate detectability constraint (see bsa.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detection import p_correct, p_error, eve_error
from .keyrate import binary_entropy


@dataclass
class NodeState:
    """Geometry-driven state of one node (Alice or a Bob)."""
    gamma0: float
    sigma_X: float
    w_eq: float


def link_stats(node: NodeState, mu: float, beta: float, expect):
    """Return (p_c, p_e, p_sift) for a node's Charlie->node link."""
    g = node.gamma0 * mu
    pc = p_correct(g, beta, node.sigma_X, expect)
    pe = p_error(g, beta, node.sigma_X, expect)
    return pc, pe, pc + pe


def node_eve_error(node: NodeState, mu: float, d_eve: float, expect) -> float:
    """URA Eve error at a node, Eve at radial offset d_eve (Eq. 8 + 22)."""
    leak = np.exp(-2.0 * d_eve ** 2 / node.w_eq ** 2)
    return eve_error(node.gamma0 * mu * leak, node.sigma_X, expect)


def bsa_deviation(node: NodeState, mu: float, beta: float, expect,
                  split: float = 0.015) -> float:
    """BSA detectability margin for a Charlie->node link (docs/decomposition.md).

    A beam-splitting attacker at the relay diverts a fraction ``split`` of the
    optical power, lowering the legitimate signal current by (1-split), i.e.
    gamma -> gamma*(1-split). The attack is detected by the relative drop in the
    monitored sift probability:
        m_BSA = 1 - P_sift(gamma*(1-split), beta) / P_sift(gamma, beta).
    m_BSA increases with beta (steeper Q tail), so detectability puts a LOWER
    bound on beta, opposing the sift upper bound -> a two-sided beta window.
    """
    g = node.gamma0 * mu
    ps = p_correct(g, beta, node.sigma_X, expect) + p_error(g, beta, node.sigma_X, expect)
    gB = g * (1.0 - split)
    psB = p_correct(gB, beta, node.sigma_X, expect) + p_error(gB, beta, node.sigma_X, expect)
    return (1.0 - psB / ps) if ps > 0 else 0.0


def exclusion_term(pc_A: float, pc_i: float, pc_others: list[float]) -> float:
    """Closed-form exclusion P^{AB_i}_excl = pc_A*pc_i*[1 - prod_{j!=i}(1-pc_j)]
    (docs/decomposition.md Eq. 3)."""
    prod = 1.0
    for pcj in pc_others:
        prod *= (1.0 - pcj)
    return pc_A * pc_i * (1.0 - prod)


def cluster_key_rate(mu, beta_A, betas, chi, alice: NodeState,
                     bobs: list[NodeState], expect, d_eve: float = 26.0,
                     Rb: float = 1e9, qber_max=1e-3, psift_min=1e-3, eve_min=0.1,
                     leak_max=0.05, bsa_split=0.015, bsa_min=0.005):
    """Total cluster secret-key rate and per-pair detail (Eq. 1-6).

    betas : per-user DT coefficients (len N). alice/bobs : NodeState.
    leak_max : max inter-user leakage L_i = (1-chi)*P_excl_i/P_chi_i allowed
        (the residual fraction of pair-i bits still shared with other users).
        This constraint is what makes chi non-trivial: chi=0 maximizes rate but
        leaks; raising chi cuts leakage at a rate cost -> interior optimum.
    Returns dict with total_rate [bits/s], per-user lists, and feasibility.
    """
    N = len(bobs)
    pc_A, pe_A, ps_A = link_stats(alice, mu, beta_A, expect)
    peE_A = node_eve_error(alice, mu, d_eve, expect)
    bsa_A = bsa_deviation(alice, mu, beta_A, expect, bsa_split)

    pc_i, pe_i, ps_i = [], [], []
    peE_i = []
    for u, b in zip(bobs, betas):
        c, e, s = link_stats(u, mu, b, expect)
        pc_i.append(c); pe_i.append(e); ps_i.append(s)
        peE_i.append(node_eve_error(u, mu, d_eve, expect))

    H2 = binary_entropy
    per_rate, per_qber, per_pchi, per_feasible, per_leak = [], [], [], [], []
    total = 0.0
    for i in range(N):
        P_AB = ps_A * ps_i[i]                                  # Eq. 1
        qber = (pc_A * pe_i[i] + pe_A * pc_i[i]) / P_AB if P_AB > 0 else 1.0  # Eq. 2
        others = [pc_i[j] for j in range(N) if j != i]
        P_excl = exclusion_term(pc_A, pc_i[i], others)         # Eq. 3
        P_chi = max(P_AB - chi * P_excl, 0.0)                  # Eq. 4
        leak = (1.0 - chi) * P_excl / P_chi if P_chi > 0 else float("inf")
        I_AB = 1.0 - H2(qber)
        I_AE = max(1.0 - H2(peE_A), 1.0 - H2(peE_i[i]))
        sigma = max(0.0, I_AB - I_AE)                          # Eq. 5
        R_i = P_chi * Rb * sigma                               # Eq. 6
        bsa_i = bsa_deviation(bobs[i], mu, betas[i], expect, bsa_split)
        feasible = (qber < qber_max and P_chi > psift_min and leak <= leak_max
                    and peE_A > eve_min and peE_i[i] > eve_min
                    and bsa_A >= bsa_min and bsa_i >= bsa_min)
        per_rate.append(R_i); per_qber.append(qber)
        per_pchi.append(P_chi); per_feasible.append(feasible); per_leak.append(leak)
        total += R_i
    return {
        "total_rate": total,
        "per_rate": per_rate, "per_qber": per_qber, "per_pchi": per_pchi,
        "per_feasible": per_feasible, "per_leak": per_leak,
        "pc_A": pc_A, "pc_i": pc_i, "peE_A": peE_A, "peE_i": peE_i,
        "n_feasible": int(sum(per_feasible)),
    }
