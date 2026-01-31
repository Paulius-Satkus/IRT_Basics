"""
Item parameter prior distributions and derivatives.

Provides log-density, gradient, and Hessian for common priors used
in item-parameter estimation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def prior_domain(kind: str) -> tuple[float | None, float | None, bool, bool]:
    """
    Return domain bounds for a prior distribution.

    Returns
    -------
    (lo, hi, lo_open, hi_open)
    """
    kind = kind.lower()
    if kind == "normal":
        return None, None, False, False
    if kind in ("lognormal", "gamma"):
        return 0.0, None, True, False
    if kind == "beta":
        return 0.0, 1.0, True, True
    raise ValueError(f"Unknown prior kind '{kind}'.")


def validate_prior_params(kind: str, params: dict[str, Any]) -> dict[str, float]:
    """
    Validate and normalize prior parameters.

    Returns
    -------
    dict with float values.
    """
    kind = kind.lower()
    if kind == "normal":
        mean = _require_param(params, "mean")
        sd = _require_param(params, "sd")
        _require_positive(sd, "sd")
        return {"mean": float(mean), "sd": float(sd)}
    if kind == "lognormal":
        mean = _require_param(params, "mean")
        sd = _require_param(params, "sd")
        _require_positive(sd, "sd")
        return {"mean": float(mean), "sd": float(sd)}
    if kind == "gamma":
        shape = _require_param(params, "shape")
        scale = _require_param(params, "scale")
        _require_positive(shape, "shape")
        _require_positive(scale, "scale")
        return {"shape": float(shape), "scale": float(scale)}
    if kind == "beta":
        alpha = _require_param(params, "alpha")
        beta = _require_param(params, "beta")
        _require_positive(alpha, "alpha")
        _require_positive(beta, "beta")
        return {"alpha": float(alpha), "beta": float(beta)}
    raise ValueError(f"Unknown prior kind '{kind}'.")


def prior_logpdf_grad_hess(
    kind: str,
    x: float,
    params: dict[str, float],
) -> tuple[float, float, float]:
    """
    Compute log-density, gradient, and Hessian at x.
    """
    kind = kind.lower()

    if kind == "normal":
        mean = params["mean"]
        sd = params["sd"]
        var = sd * sd
        logpdf = -0.5 * math.log(2 * math.pi * var) - (x - mean) ** 2 / (2 * var)
        grad = -(x - mean) / var
        hess = -1.0 / var
        return logpdf, grad, hess

    if kind == "lognormal":
        if x <= 0:
            return -np.inf, 0.0, 0.0
        mean = params["mean"]
        sd = params["sd"]
        logx = math.log(x)
        logpdf = -math.log(x * sd * math.sqrt(2 * math.pi)) - (
            (logx - mean) ** 2
        ) / (2 * sd * sd)
        u = 1.0 + (logx - mean) / (sd * sd)
        grad = -u / x
        hess = (u - 1.0 / (sd * sd)) / (x * x)
        return logpdf, grad, hess

    if kind == "gamma":
        if x <= 0:
            return -np.inf, 0.0, 0.0
        shape = params["shape"]
        scale = params["scale"]
        logpdf = (
            (shape - 1.0) * math.log(x)
            - x / scale
            - shape * math.log(scale)
            - math.lgamma(shape)
        )
        grad = (shape - 1.0) / x - 1.0 / scale
        hess = -(shape - 1.0) / (x * x)
        return logpdf, grad, hess

    if kind == "beta":
        if x <= 0 or x >= 1:
            return -np.inf, 0.0, 0.0
        alpha = params["alpha"]
        beta = params["beta"]
        logpdf = (
            (alpha - 1.0) * math.log(x)
            + (beta - 1.0) * math.log(1.0 - x)
            - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
        )
        grad = (alpha - 1.0) / x - (beta - 1.0) / (1.0 - x)
        hess = -(alpha - 1.0) / (x * x) - (beta - 1.0) / ((1.0 - x) ** 2)
        return logpdf, grad, hess

    raise ValueError(f"Unknown prior kind '{kind}'.")


def _require_param(params: dict[str, Any], key: str) -> float:
    if key not in params:
        raise ValueError(f"Missing prior parameter '{key}'.")
    return float(params[key])


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"Prior parameter '{name}' must be positive.")
