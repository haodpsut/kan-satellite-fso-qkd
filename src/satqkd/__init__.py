"""satqkd — analytical ground-truth + Monte-Carlo environment for adaptive
multi-user satellite FSO/QKD with DT/DD receivers.

See docs/formulation.md for the equation-by-equation derivation.
"""
from .params import SystemParams
from .link import (
    LinkState,
    build_link_state,
    noise_std,
    physical_gamma0,
    calibration_scale,
)
from .detection import (
    qfunc,
    p_correct,
    p_error,
    sift_qber,
    eve_error,
    make_expectation,
)
from .turbulence import sigma_X2, cn2_hufnagel_valley, gauss_hermite_lognormal
from .geometry import deterministic_channel, collected_fraction, beam_radius

__all__ = [
    "SystemParams",
    "LinkState",
    "build_link_state",
    "noise_std",
    "physical_gamma0",
    "calibration_scale",
    "qfunc",
    "p_correct",
    "p_error",
    "sift_qber",
    "eve_error",
    "make_expectation",
    "sigma_X2",
    "cn2_hufnagel_valley",
    "gauss_hermite_lognormal",
    "deterministic_channel",
    "collected_fraction",
    "beam_radius",
]
