"""Secret-key rate for a single Alice-Bob pair (wiretap / Csiszar-Korner).

After DT/DD detection and sifting, the legitimate channel on the *sifted* bits is
a BSC with crossover = QBER, while Eve (who learns the sift positions from the
public channel) detects the same sifted bits with error Pe. Per [P3] Eq. 25-29:

    I(A;B) = 1 - H2(QBER)            (per sifted bit)
    I(A;E) = 1 - H2(Pe)             (per sifted bit, [P1] Eq. 42)
    secret fraction = max(0, I(A;B) - I(A;E)) = max(0, H2(Pe) - H2(QBER))
    R_f = R_s * secret fraction,  R_s = P_sift * R_b      (sifted-key rate)

so the *normalized* key rate (per R_b) is

    r_norm = P_sift * max(0, H2(Pe) - H2(QBER)).

Maximizing it trades off: large P_sift wants small beta, but QBER<1e-3 wants
large beta/gamma; large Pe wants small mu, but P_sift/QBER want large mu. See
docs/formulation.md Section 7-8.
"""
from __future__ import annotations

import numpy as np


def binary_entropy(x: float) -> float:
    """H2(x) = -x log2 x - (1-x) log2(1-x), with H2(0)=H2(1)=0."""
    x = float(np.clip(x, 0.0, 1.0))
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)


def info_AB(qber: float) -> float:
    """Per-sifted-bit mutual information I(A;B) = 1 - H2(QBER)."""
    return 1.0 - binary_entropy(qber)


def info_AE(pe: float) -> float:
    """Per-sifted-bit mutual information I(A;E) = 1 - H2(Pe)  ([P1] Eq. 42)."""
    return 1.0 - binary_entropy(pe)


def secret_fraction(qber: float, pe: float) -> float:
    """Secret bits per sifted bit = max(0, I(A;B) - I(A;E))."""
    return max(0.0, info_AB(qber) - info_AE(pe))


def normalized_key_rate(p_sift: float, qber: float, pe: float) -> float:
    """Normalized secret-key rate r_norm = P_sift * secret_fraction (per R_b).

    Multiply by the system bit rate R_b to get bits/s.
    """
    return p_sift * secret_fraction(qber, pe)
