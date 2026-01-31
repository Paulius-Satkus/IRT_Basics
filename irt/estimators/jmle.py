"""
Joint Maximum Likelihood Estimation for Rasch model.

This module implements JMLE for the Rasch model, where both person
abilities (theta) and item difficulties (b) are estimated simultaneously
as fixed effects.

Model Parameterization
----------------------
P(X=1|theta, b) = 1 / (1 + exp(-(theta - b)))

Identification
--------------
JMLE requires an identification constraint since theta and b are only
identified up to a constant. We use mean(b) = 0 (centered difficulties).

Extreme Scores
--------------
Persons with perfect (all 1s) or zero (all 0s) scores have MLE at
+/- infinity. These are handled via:
1. Nudging: Modify the score slightly before estimation
2. Bayesian prior: Add a weak prior to regularize

Note: 2PL JMLE is not implemented (computationally more complex and
less commonly used in practice).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.special import expit

from ..types import FitResult, ScoreResult
from ..core import center_difficulties
from ..data import (
    compute_item_pvalues,
    compute_person_scores,
    identify_extreme_scores,
    validate_start_params,
    validate_fixed_params,
    parse_item_priors,
)
from ..item_priors import prior_logpdf_grad_hess


def initialize_params_jmle(
    X: np.ndarray,
    mask_obs: np.ndarray,
    start: dict | None = None,
    method: str = "prox",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize parameters for JMLE.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    start : dict or None
        User-provided starting values.
    method : str, default="prox"
        Initialization method:
        - "prox": Use p-values and raw scores (proportional)
        - "zeros": Start with all zeros

    Returns
    -------
    theta : np.ndarray
        Initial person abilities, shape (N,).
    b : np.ndarray
        Initial item difficulties, shape (J,).

    Notes
    -----
    Default initialization:
    - b: logit(1 - p_value) where p_value is proportion correct
    - theta: logit(score / max_score) adjusted for extreme scores
    """
    N, J = X.shape

    # Initialize b from starting values or p-values
    if start is not None and "b" in start:
        b = start["b"].copy()
    else:
        if method == "zeros":
            b = np.zeros(J, dtype=np.float64)
        else:  # prox
            p_values = compute_item_pvalues(X, mask_obs)
            b = np.log((1 - p_values) / p_values)
            b = np.clip(b, -5.0, 5.0)

    # Initialize theta from starting values or raw scores
    if start is not None and "theta" in start:
        theta = start["theta"].copy()
    else:
        if method == "zeros":
            theta = np.zeros(N, dtype=np.float64)
        else:  # prox
            sum_score, max_score = compute_person_scores(X, mask_obs)
            # Proportion correct
            prop = sum_score / np.maximum(max_score, 1)
            # Adjust extreme scores
            prop = np.clip(prop, 0.01, 0.99)
            theta = np.log(prop / (1 - prop))
            theta = np.clip(theta, -5.0, 5.0)

    # Center difficulties for identification
    b = center_difficulties(b)

    return theta, b


def _compute_loglik_jmle(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
) -> float:
    """Compute complete-data log-likelihood for JMLE."""
    N, J = X.shape
    loglik = 0.0

    for j in range(J):
        obs_j = mask_obs[:, j]
        if not np.any(obs_j):
            continue
        theta_obs = theta[obs_j]
        x_j = X[obs_j, j]
        z = theta_obs - b[j]
        p = expit(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        loglik += np.sum(x_j * np.log(p) + (1 - x_j) * np.log(1 - p))

    return loglik


def _update_person_theta(
    X_row: np.ndarray,
    mask_row: np.ndarray,
    b: np.ndarray,
    theta0: float,
    bounds: tuple[float, float],
    max_iter: int = 15,
    prior_sd: float | None = None,
) -> float:
    """
    Update theta for one person using Newton-Raphson.

    Parameters
    ----------
    X_row : np.ndarray
        Responses for one person, shape (J,).
    mask_row : np.ndarray
        Observation mask, shape (J,).
    b : np.ndarray
        Item difficulties, shape (J,).
    theta0 : float
        Starting value.
    bounds : tuple[float, float]
        Bounds for theta.
    max_iter : int
        Maximum iterations.
    prior_sd : float or None
        If provided, adds a N(0, prior_sd^2) prior for regularization.

    Returns
    -------
    float
        Updated theta value.
    """
    theta_lo, theta_hi = bounds
    theta = np.clip(theta0, theta_lo, theta_hi)

    # Get observed items
    obs = mask_row
    if not np.any(obs):
        return theta

    x_obs = X_row[obs]
    b_obs = b[obs]

    for _ in range(max_iter):
        z = theta - b_obs
        p = expit(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)

        # Gradient: dL/dtheta = sum(x - p)
        grad = np.sum(x_obs - p)

        # Hessian: d2L/dtheta2 = -sum(p * (1-p))
        hess = -np.sum(p * (1 - p))

        # Add prior contribution if specified
        if prior_sd is not None:
            # Prior: log p(theta) = -theta^2 / (2 * sd^2) + const
            grad -= theta / prior_sd**2
            hess -= 1.0 / prior_sd**2

        if abs(grad) < 1e-8:
            break

        if abs(hess) < 1e-12:
            step = 0.1 * np.sign(grad)
        else:
            step = -grad / hess

        step = np.clip(step, -2.0, 2.0)
        theta = np.clip(theta + step, theta_lo, theta_hi)

    return float(theta)


def _update_item_b(
    X_col: np.ndarray,
    mask_col: np.ndarray,
    theta: np.ndarray,
    b0: float,
    bounds: tuple[float, float],
    max_iter: int = 15,
    prior: dict | None = None,
) -> float:
    """
    Update b for one item using Newton-Raphson.

    Parameters
    ----------
    X_col : np.ndarray
        Responses for one item, shape (N,).
    mask_col : np.ndarray
        Observation mask, shape (N,).
    theta : np.ndarray
        Person abilities, shape (N,).
    b0 : float
        Starting value.
    bounds : tuple[float, float]
        Bounds for b.
    max_iter : int
        Maximum iterations.
    prior : dict or None
        Optional prior specification for b.

    Returns
    -------
    float
        Updated b value.
    """
    b_lo, b_hi = bounds
    b = np.clip(b0, b_lo, b_hi)

    # Get observed persons
    obs = mask_col
    if not np.any(obs):
        return b

    x_obs = X_col[obs]
    theta_obs = theta[obs]

    for _ in range(max_iter):
        z = theta_obs - b
        p = expit(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)

        # Gradient: dL/db = sum(p - x) (note sign from -b in z)
        grad = np.sum(p - x_obs)

        # Hessian: d2L/db2 = -sum(p * (1-p))
        hess = -np.sum(p * (1 - p))

        if prior is not None:
            _, grad_p, hess_p = prior_logpdf_grad_hess(
                prior["kind"], b, prior["params"]
            )
            grad += grad_p
            hess += hess_p

        if abs(grad) < 1e-8:
            break

        if abs(hess) < 1e-12:
            step = -0.1 * np.sign(grad)
        else:
            step = -grad / hess

        step = np.clip(step, -2.0, 2.0)
        b = np.clip(b + step, b_lo, b_hi)

    return float(b)


def _compute_se_theta(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute standard errors for theta estimates.

    SE = 1 / sqrt(sum_j I_j(theta)) where I_j = p_j * (1 - p_j)
    """
    N, J = X.shape
    se = np.zeros(N, dtype=np.float64)

    for i in range(N):
        obs = mask_obs[i]
        if not np.any(obs):
            se[i] = np.inf
            continue

        b_obs = b[obs]
        z = theta[i] - b_obs
        p = expit(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        info = np.sum(p * (1 - p))
        se[i] = 1.0 / np.sqrt(max(info, 1e-12))

    return se


def _compute_se_b(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Compute standard errors for b estimates.

    SE = 1 / sqrt(sum_n I_n(b_j)) where I_n = p_nj * (1 - p_nj)
    """
    N, J = X.shape
    se = np.zeros(J, dtype=np.float64)

    for j in range(J):
        obs = mask_obs[:, j]
        if not np.any(obs):
            se[j] = np.inf
            continue

        theta_obs = theta[obs]
        z = theta_obs - b[j]
        p = expit(z)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        info = np.sum(p * (1 - p))
        se[j] = 1.0 / np.sqrt(max(info, 1e-12))

    return se


def jmle_iterate_rasch(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
    technical: dict[str, Any],
    priors: dict[str, list[dict[str, Any] | None]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run JMLE iterations for Rasch model.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Initial person abilities, shape (N,).
    b : np.ndarray
        Initial item difficulties, shape (J,).
    technical : dict
        Technical parameters.
    priors : dict or None
        Parsed item priors for b.

    Returns
    -------
    theta_hat : np.ndarray
        Estimated abilities.
    b_hat : np.ndarray
        Estimated difficulties.
    se_theta : np.ndarray
        Standard errors for theta.
    se_b : np.ndarray
        Standard errors for b.
    history : dict
        Convergence history.
    """
    N, J = X.shape

    max_iter = technical["max_iter"]
    tol = technical["tol"]
    nudge = technical["nudge"]
    theta_bounds = technical["theta_bounds"]
    b_bounds = technical["b_bounds"]

    # Handle extreme scores
    is_perfect, is_zero = identify_extreme_scores(X, mask_obs)
    has_extreme = is_perfect | is_zero

    # Nudge extreme thetas
    if np.any(has_extreme):
        warnings.warn(
            f"{has_extreme.sum()} persons have extreme scores. "
            "Using Bayesian prior for regularization.",
            RuntimeWarning,
            stacklevel=3,
        )
    prior_sd_extreme = 3.0  # Weak prior for extreme scorers

    history = {"loglik": [], "max_param_change": []}

    theta = theta.copy()
    b = b.copy()

    for iteration in range(max_iter):
        theta_old = theta.copy()
        b_old = b.copy()

        # Update theta for each person
        for i in range(N):
            prior_sd = prior_sd_extreme if has_extreme[i] else None
            theta[i] = _update_person_theta(
                X[i], mask_obs[i], b, theta[i], theta_bounds,
                prior_sd=prior_sd
            )

        # Update b for each item
        for j in range(J):
            prior_b = None if priors is None else priors["b"][j]
            b[j] = _update_item_b(
                X[:, j],
                mask_obs[:, j],
                theta,
                b[j],
                b_bounds,
                prior=prior_b,
            )

        # Center difficulties
        b = center_difficulties(b)

        # Compute log-likelihood
        loglik = _compute_loglik_jmle(X, mask_obs, theta, b)
        history["loglik"].append(loglik)

        # Compute parameter changes
        max_change_theta = np.max(np.abs(theta - theta_old))
        max_change_b = np.max(np.abs(b - b_old))
        max_change = max(max_change_theta, max_change_b)
        history["max_param_change"].append(max_change)

        # Check convergence
        if max_change < tol:
            break

    # Compute standard errors
    se_theta = _compute_se_theta(X, mask_obs, theta, b)
    se_b = _compute_se_b(X, mask_obs, theta, b)

    # Set SE to inf for extreme scorers
    se_theta[has_extreme] = np.inf

    return theta, b, se_theta, se_b, history


def fit_jmle(
    X: np.ndarray,
    mask_obs: np.ndarray,
    technical: dict[str, Any],
    start: dict | None,
    fixed: dict | None,
    priors: Any | None,
    constraints: dict[str, Any],
    item_names: list[str],
    person_names: list[str],
) -> FitResult:
    """
    Fit Rasch model using JMLE.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    technical : dict
        Technical parameters.
    start : dict or None
        Starting values.
    fixed : dict or None
        Fixed parameter masks.
    priors : dict or DataFrame or None
        Item parameter priors.
    constraints : dict
        Constraints.
    item_names : list[str]
        Item names.
    person_names : list[str]
        Person names.

    Returns
    -------
    FitResult
        Fitted model result.
    """
    N, J = X.shape

    # Validate starting values
    validated_start = validate_start_params(start, J, N, "rasch", "jmle")
    validated_fixed = validate_fixed_params(fixed, J, validated_start)
    if validated_fixed is not None and (
        np.any(validated_fixed.get("a", False)) or np.any(validated_fixed.get("b", False))
    ):
        raise ValueError(
            "Fixed parameters are not supported for JMLE estimation. "
            "Use estimator='mml_em' or omit fixed parameters."
        )

    # Initialize parameters
    theta, b = initialize_params_jmle(X, mask_obs, validated_start)

    parsed_priors = parse_item_priors(
        priors=priors,
        item_names=item_names,
        model="rasch",
        b_bounds=technical["b_bounds"],
    )

    # Run JMLE iterations
    theta_hat, b_hat, se_theta, se_b, history = jmle_iterate_rasch(
        X, mask_obs, theta, b, technical, parsed_priors
    )

    # Check convergence
    converged = len(history["max_param_change"]) > 0 and \
                history["max_param_change"][-1] < technical["tol"]

    if not converged:
        warnings.warn(
            f"JMLE did not converge in {technical['max_iter']} iterations. "
            f"Final max parameter change: {history['max_param_change'][-1]:.2e}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Store theta estimates for scoring
    theta_estimates = theta_hat.copy()
    se_estimates = se_theta.copy()

    # Create scoring function
    def score_fn(
        X_score: np.ndarray | None = None,
        method: str = "map",
        **kwargs: Any,
    ) -> ScoreResult:
        return _score_jmle(
            X_score=X_score,
            method=method,
            X_train=X,
            mask_obs_train=mask_obs,
            b=b_hat,
            theta_train=theta_estimates,
            se_train=se_estimates,
            person_names_train=person_names,
            technical=technical,
            **kwargs,
        )

    return FitResult(
        model="rasch",
        estimator="jmle",
        params={"a": np.ones(J), "b": b_hat.copy()},
        converged=converged,
        n_iter=len(history["loglik"]),
        loglik=history["loglik"][-1] if history["loglik"] else None,
        history=history,
        theta_grid=None,  # JMLE doesn't use quadrature
        log_prior=None,
        posterior_weights=None,
        item_names=item_names,
        person_names=person_names,
        mask_obs=mask_obs,
        X=X,
        _score_fn=score_fn,
    )


def _score_jmle(
    X_score: np.ndarray | None,
    method: str,
    X_train: np.ndarray,
    mask_obs_train: np.ndarray,
    b: np.ndarray,
    theta_train: np.ndarray,
    se_train: np.ndarray,
    person_names_train: list[str],
    technical: dict[str, Any],
    **kwargs: Any,
) -> ScoreResult:
    """
    Score persons using JMLE estimates.

    Parameters
    ----------
    X_score : np.ndarray or None
        Data to score. If None, return training estimates.
    method : str
        Scoring method: "map" or "mle".
    X_train, mask_obs_train : np.ndarray
        Training data.
    b : np.ndarray
        Estimated item difficulties.
    theta_train, se_train : np.ndarray
        Training theta estimates and SEs.
    person_names_train : list[str]
        Training person names.
    technical : dict
        Technical parameters.
    **kwargs
        Additional arguments.

    Returns
    -------
    ScoreResult
        Scoring results.
    """
    method = method.lower()

    # EAP not available for JMLE
    if method == "eap":
        raise ValueError(
            "EAP scoring is not available for JMLE estimation. "
            "Use method='map' or method='mle' instead."
        )

    valid_methods = ("map", "mle")
    if method not in valid_methods:
        raise ValueError(
            f"Invalid scoring method '{method}'. Must be one of {valid_methods}."
        )

    if X_score is None:
        # Return training estimates
        return ScoreResult(
            theta=theta_train.copy(),
            se=se_train.copy(),
            method=method,
            person_names=person_names_train,
        )

    # Score new data
    from ..data import as_binary_matrix
    X_new, mask_new, _, person_names = as_binary_matrix(X_score)
    N_new = X_new.shape[0]

    # Check dimensions match
    if X_new.shape[1] != len(b):
        raise ValueError(
            f"New data has {X_new.shape[1]} items but model has {len(b)} items."
        )

    # Initialize from raw scores
    theta_init, _ = initialize_params_jmle(X_new, mask_new)

    # Handle extreme scores
    is_perfect, is_zero = identify_extreme_scores(X_new, mask_new)
    has_extreme = is_perfect | is_zero
    prior_sd_extreme = 3.0 if method == "map" else None

    theta_hat = np.zeros(N_new, dtype=np.float64)
    for i in range(N_new):
        prior_sd = prior_sd_extreme if has_extreme[i] else (3.0 if method == "map" else None)
        theta_hat[i] = _update_person_theta(
            X_new[i], mask_new[i], b, theta_init[i],
            technical["theta_bounds"],
            prior_sd=prior_sd,
        )

    # Compute SEs
    se = _compute_se_theta(X_new, mask_new, theta_hat, b)
    se[has_extreme] = np.inf

    return ScoreResult(
        theta=theta_hat,
        se=se,
        method=method,
        person_names=person_names,
    )
