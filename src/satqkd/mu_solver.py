"""Decomposed solver for the joint multi-user problem (P) of docs/decomposition.md.

Structure:
  Outer (O)  : 2-D search over (mu, chi).
  Inner (I)  : block-coordinate over (beta_A, {beta_i}).
  Mean field : the exclusion field phi_i = prod_{j!=i}(1-pc_j) decouples the
               per-user subproblems (U_i); iterate to a fixed point.

This is the oracle that produces labels state -> (mu, beta_A, {beta_i}, chi) for
the learned controller, and the adaptive upper bound for the headline comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multiuser import (NodeState, link_stats, node_eve_error, cluster_key_rate,
                        bsa_deviation)
from .keyrate import binary_entropy


@dataclass
class ClusterSolution:
    mu: float
    beta_A: float
    betas: list
    chi: float
    total_rate: float
    n_feasible: int


def _user_rate(beta_i, node_i, mu, pcA, peA, psA, phi_i, chi, peE_A, bsa_A, expect,
               d_eve, Rb, qmax, psmin, pemin, leak_max, bsa_split, bsa_min):
    """R_i and feasibility for user i given the field and Alice stats (subproblem U_i)."""
    pc, pe, ps = link_stats(node_i, mu, beta_i, expect)
    P_AB = psA * ps
    qber = (pcA * pe + peA * pc) / P_AB if P_AB > 0 else 1.0
    P_excl = pcA * pc * (1.0 - phi_i)
    P_chi = max(0.0, P_AB - chi * P_excl)
    leak = (1.0 - chi) * P_excl / P_chi if P_chi > 0 else float("inf")
    peE_i = node_eve_error(node_i, mu, d_eve, expect)
    bsa_i = bsa_deviation(node_i, mu, beta_i, expect, bsa_split)
    I_AB = 1.0 - binary_entropy(qber)
    I_AE = max(1.0 - binary_entropy(peE_A), 1.0 - binary_entropy(peE_i))
    sigma = max(0.0, I_AB - I_AE)
    R_i = P_chi * Rb * sigma
    feasible = (qber < qmax and P_chi > psmin and leak <= leak_max
                and peE_A > pemin and peE_i > pemin
                and bsa_A >= bsa_min and bsa_i >= bsa_min)
    return R_i, feasible, pc


def _argmax_beta(fn, beta_grid):
    """Return beta maximizing fn(beta)->(rate, feasible, ...), preferring feasible."""
    best_feas = best_any = None
    for b in beta_grid:
        out = fn(b)
        r, feas = out[0], out[1]
        if feas and (best_feas is None or r > best_feas[0]):
            best_feas = (r, b)
        if best_any is None or r > best_any[0]:
            best_any = (r, b)
    return (best_feas or best_any)[1]


def inner_meanfield(mu, chi, alice, bobs, expect, d_eve=26.0, Rb=1e9,
                    beta_range=(0.3, 4.5), n_beta=29, max_iter=12, tol=1e-3,
                    qmax=1e-3, psmin=1e-3, pemin=0.1, leak_max=0.05,
                    bsa_split=0.015, bsa_min=0.005):
    """Solve inner (I) for fixed (mu, chi) by mean-field block coordinate."""
    N = len(bobs)
    bg = np.linspace(*beta_range, n_beta)
    betas = np.full(N, float(np.mean(beta_range)))
    beta_A = float(np.mean(beta_range))
    peE_A = node_eve_error(alice, mu, d_eve, expect)

    for _ in range(max_iter):
        prev = (beta_A, betas.copy())
        # current Alice stats + per-user pc for the field
        pcA, peA, psA = link_stats(alice, mu, beta_A, expect)
        bsa_A = bsa_deviation(alice, mu, beta_A, expect, bsa_split)
        pc_now = [link_stats(b, mu, betas[i], expect)[0] for i, b in enumerate(bobs)]
        # (U_i): each user solves its beta_i given field phi_i
        for i, node in enumerate(bobs):
            phi_i = 1.0
            for j in range(N):
                if j != i:
                    phi_i *= (1.0 - pc_now[j])
            betas[i] = _argmax_beta(
                lambda b: _user_rate(b, node, mu, pcA, peA, psA, phi_i, chi,
                                     peE_A, bsa_A, expect, d_eve, Rb, qmax, psmin,
                                     pemin, leak_max, bsa_split, bsa_min),
                bg)
            pc_now[i] = link_stats(node, mu, betas[i], expect)[0]
        # block-coordinate update of shared beta_A (maximize cluster total)
        def total_for_betaA(bA):
            res = cluster_key_rate(mu, bA, betas.tolist(), chi, alice, bobs,
                                   expect, d_eve, Rb, qmax, psmin, pemin, leak_max,
                                   bsa_split, bsa_min)
            return res["total_rate"], res["n_feasible"] > 0
        beta_A = _argmax_beta(total_for_betaA, bg)
        # convergence
        if abs(beta_A - prev[0]) < tol and np.all(np.abs(betas - prev[1]) < tol):
            break

    res = cluster_key_rate(mu, beta_A, betas.tolist(), chi, alice, bobs,
                           expect, d_eve, Rb, qmax, psmin, pemin, leak_max,
                           bsa_split, bsa_min)
    return beta_A, betas.tolist(), res


def solve_cluster(alice, bobs, expect, d_eve=26.0, Rb=1e9,
                  mu_grid=None, chi_grid=None, **inner_kw) -> ClusterSolution:
    """Outer (O): 2-D search over (mu, chi); inner solved by mean field."""
    if mu_grid is None:
        mu_grid = np.linspace(0.2, 0.95, 16)
    if chi_grid is None:
        chi_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    best = None
    for mu in mu_grid:
        for chi in chi_grid:
            beta_A, betas, res = inner_meanfield(
                float(mu), float(chi), alice, bobs, expect, d_eve, Rb, **inner_kw)
            # rank by feasible count then total rate (prefer operationally usable)
            key = (res["n_feasible"], res["total_rate"])
            if best is None or key > best[0]:
                best = (key, ClusterSolution(float(mu), beta_A, betas, float(chi),
                                             res["total_rate"], res["n_feasible"]))
    return best[1]
