"""Per-state constrained optimization of (mu, beta) for one channel state.

Solves docs/formulation.md Eq. (26) for a single Alice-Bob pair:
    max_{mu,beta}  r_norm(mu, beta)   s.t.  QBER<1e-3, P_sift>1e-3, Eve>0.1.

Grid search + local refinement (robust, derivative-free; the objective is cheap
- a few Gauss-Hermite expectations per evaluation). Produces the optimal
parameters that the Phase-3 KAN controller will learn to predict.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .detection import sift_qber, eve_error
from .keyrate import normalized_key_rate
from .link import LinkState


# default constraint thresholds ([P3] Sec. V)
QBER_MAX = 1e-3
PSIFT_MIN = 1e-3
EVE_MIN = 0.1


@dataclass
class OptResult:
    mu: float
    beta: float
    p_sift: float
    qber: float
    eve_error: float
    r_norm: float          # normalized secret-key rate (per R_b)
    feasible: bool


def evaluate(mu: float, beta: float, ls: LinkState, expect, d_eve: float):
    """Return (p_sift, qber, eve_error, r_norm) for given (mu, beta) at a state."""
    psift, qber = sift_qber(ls.gamma(mu), beta, ls.sigma_X, expect)
    pe = eve_error(ls.gamma_eve(mu, d_eve), ls.sigma_X, expect)
    r = normalized_key_rate(psift, qber, pe)
    return psift, qber, pe, r


def _search(ls, expect, d_eve, mu_vals, beta_vals,
            qber_max, psift_min, eve_min):
    """Grid search; return (best_feasible OptResult or None, best_any OptResult)."""
    best_feas = None
    best_any = None
    for mu in mu_vals:
        for beta in beta_vals:
            psift, qber, pe, r = evaluate(mu, beta, ls, expect, d_eve)
            feasible = (qber < qber_max and psift > psift_min and pe > eve_min)
            res = OptResult(float(mu), float(beta), psift, qber, pe, r, feasible)
            if feasible and (best_feas is None or r > best_feas.r_norm):
                best_feas = res
            if best_any is None or r > best_any.r_norm:
                best_any = res
    return best_feas, best_any


def optimize_state(ls: LinkState, expect, d_eve: float = 26.0,
                   qber_max: float = QBER_MAX, psift_min: float = PSIFT_MIN,
                   eve_min: float = EVE_MIN,
                   mu_range=(0.05, 0.95), beta_range=(0.3, 4.5),
                   coarse: int = 31, refine: int = 21) -> OptResult:
    """Optimize (mu, beta) for one LinkState.

    Two-stage: a coarse grid then a refined grid around the coarse optimum.
    Returns the best feasible result; if none is feasible, returns the best-rate
    result with feasible=False.
    """
    mu0 = np.linspace(*mu_range, coarse)
    b0 = np.linspace(*beta_range, coarse)
    best_feas, best_any = _search(ls, expect, d_eve, mu0, b0,
                                  qber_max, psift_min, eve_min)
    anchor = best_feas if best_feas is not None else best_any

    # refine around the anchor
    dmu = (mu_range[1] - mu_range[0]) / (coarse - 1)
    db = (beta_range[1] - beta_range[0]) / (coarse - 1)
    mu1 = np.linspace(max(mu_range[0], anchor.mu - dmu),
                      min(mu_range[1], anchor.mu + dmu), refine)
    b1 = np.linspace(max(beta_range[0], anchor.beta - db),
                     min(beta_range[1], anchor.beta + db), refine)
    feas2, any2 = _search(ls, expect, d_eve, mu1, b1,
                          qber_max, psift_min, eve_min)

    # combine coarse + refined
    candidates_feas = [r for r in (best_feas, feas2) if r is not None]
    if candidates_feas:
        return max(candidates_feas, key=lambda r: r.r_norm)
    return max([best_any, any2], key=lambda r: r.r_norm)


def result_to_row(res: OptResult) -> dict:
    return asdict(res)
