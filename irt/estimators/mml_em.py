"""
Marginal Maximum Likelihood estimation via EM algorithm.

This module implements MML-EM for Rasch and 2PL models:
- E-step: Compute posterior weights over ability distribution
- M-step: Update item parameters given posterior weights

The algorithm iterates between these steps until convergence.

Model Parameterization
----------------------
P(X=1|theta) = 1 / (1 + exp(-a * (theta - b)))

Identification
--------------
The ability scale is identified through the prior distribution,
typically N(0, 1). This means theta has a population mean of 0
and variance of 1.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.special import logsumexp

from ..types import FitResult, ScoreResult
from ..core import prob_2pl, log_prob_2pl, log_prob_3pl, center_difficulties
from ..item_priors import validate_prior_params
from ..data import (
    compute_item_pvalues,
    validate_start_params,
    validate_fixed_params,
    parse_item_priors,
)
from ..quadrature import make_theta_grid, prior_logpdf


# =============================================================================
# E-Step Functions
# =============================================================================


def compute_loglik_nk(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute log-likelihood for each person at each quadrature point.

    Computes log P(X_n | theta_k) for all persons n and quadrature points k.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Boolean observation mask, shape (N, J).
    theta : np.ndarray
        Quadrature points, shape (K,).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).

    Returns
    -------
    np.ndarray
        Log-likelihood matrix, shape (N, K).

    Notes
    -----
    Implementation is memory-efficient: loops over items rather than
    allocating a full (N, J, K) array.

    The log-likelihood for person n at quadrature point k is:
        log P(X_n | theta_k) = sum_j [x_nj * log(p_jk) + (1-x_nj) * log(1-p_jk)]
    where the sum is only over observed items (where mask_obs[n,j] = True).
    """
    N, J = X.shape
    K = len(theta)

    # Initialize log-likelihood to 0
    logL_nk = np.zeros((N, K), dtype=np.float64)

    # Loop over items (memory efficient)
    for j in range(J):
        # Get observed persons for this item
        obs_j = mask_obs[:, j]
        if not np.any(obs_j):
            continue

        # Compute log P and log(1-P) at all quadrature points
        if c is None:
            log_p_k, log_q_k = log_prob_2pl(theta, a[j], b[j])
        else:
            log_p_k, log_q_k = log_prob_3pl(theta, a[j], b[j], c[j])

        # Get responses for observed persons
        x_j = X[obs_j, j]  # shape (N_obs,)

        # Add contribution to log-likelihood
        # x_j[:, None] broadcasts to (N_obs, K)
        # log_p_k[None, :] broadcasts to (N_obs, K)
        contribution = x_j[:, np.newaxis] * log_p_k[np.newaxis, :] + \
                      (1 - x_j[:, np.newaxis]) * log_q_k[np.newaxis, :]
        logL_nk[obs_j, :] += contribution

    return logL_nk


def posterior_weights(
    logL_nk: np.ndarray,
    log_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute posterior weights and marginal log-likelihood.

    Parameters
    ----------
    logL_nk : np.ndarray
        Log-likelihood matrix from compute_loglik_nk, shape (N, K).
    log_prior : np.ndarray
        Log-prior values at quadrature points, shape (K,).

    Returns
    -------
    w_nk : np.ndarray
        Posterior weights, shape (N, K). Rows sum to 1.
    log_evidence_n : np.ndarray
        Log marginal likelihood for each person, shape (N,).
        These are the denominators in the posterior computation.

    Notes
    -----
    The posterior is computed as:
        w_nk = P(theta_k | X_n) = P(X_n | theta_k) * P(theta_k) / P(X_n)

    In log space:
        log(w_nk) = logL_nk + log_prior - logsumexp(logL_nk + log_prior)

    The marginal likelihood contribution from person n is:
        P(X_n) = sum_k P(X_n | theta_k) * P(theta_k) * dx
               ≈ sum_k exp(logL_nk[n, k] + log_prior[k])

    We use logsumexp for numerical stability.
    """
    # Unnormalized log-posterior
    log_post_unnorm = logL_nk + log_prior[np.newaxis, :]

    # Log marginal likelihood (normalizing constant)
    log_evidence_n = logsumexp(log_post_unnorm, axis=1)

    # Normalized log-posterior
    log_w_nk = log_post_unnorm - log_evidence_n[:, np.newaxis]

    # Exponentiate to get weights
    w_nk = np.exp(log_w_nk)

    # Ensure rows sum to 1 (handle numerical issues)
    w_nk = w_nk / w_nk.sum(axis=1, keepdims=True)

    return w_nk, log_evidence_n


# =============================================================================
# Sufficient Statistics
# =============================================================================


def sufficient_stats_item(
    X_col: np.ndarray,
    mask_col: np.ndarray,
    w_nk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sufficient statistics for a single item.

    Parameters
    ----------
    X_col : np.ndarray
        Responses for one item, shape (N,).
    mask_col : np.ndarray
        Observation mask for this item, shape (N,).
    w_nk : np.ndarray
        Posterior weights, shape (N, K).

    Returns
    -------
    N_k : np.ndarray
        Expected number of observations at each quadrature point, shape (K,).
    R_k : np.ndarray
        Expected number of correct responses at each quadrature point, shape (K,).

    Notes
    -----
    N_k[k] = sum_n w_nk[n, k] * mask_col[n]
    R_k[k] = sum_n w_nk[n, k] * mask_col[n] * X_col[n]
    """
    # Only include observed responses
    w_obs = w_nk[mask_col]  # shape (N_obs, K)
    x_obs = X_col[mask_col]  # shape (N_obs,)

    # N_k: expected count at each quadrature point
    N_k = w_obs.sum(axis=0)  # shape (K,)

    # R_k: expected correct at each quadrature point
    R_k = (w_obs * x_obs[:, np.newaxis]).sum(axis=0)  # shape (K,)

    return N_k, R_k


def sufficient_stats_all_items(
    X: np.ndarray,
    mask_obs: np.ndarray,
    w_nk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sufficient statistics for all items.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    w_nk : np.ndarray
        Posterior weights, shape (N, K).

    Returns
    -------
    N_jk : np.ndarray
        Expected counts, shape (J, K).
    R_jk : np.ndarray
        Expected correct counts, shape (J, K).
    """
    N, J = X.shape
    K = w_nk.shape[1]

    N_jk = np.zeros((J, K), dtype=np.float64)
    R_jk = np.zeros((J, K), dtype=np.float64)

    for j in range(J):
        N_jk[j], R_jk[j] = sufficient_stats_item(X[:, j], mask_obs[:, j], w_nk)

    return N_jk, R_jk


# =============================================================================
# Parameter Initialization
# =============================================================================


def initialize_params_mml(
    X: np.ndarray,
    mask_obs: np.ndarray,
    model: str,
    start: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Initialize item parameters for MML-EM.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    model : str
        Model type ("rasch", "2pl", or "3pl").
    start : dict or None
        User-provided starting values.

    Returns
    -------
    a : np.ndarray
        Initial discrimination parameters, shape (J,).
    b : np.ndarray
        Initial difficulty parameters, shape (J,).
    c : np.ndarray or None
        Initial guessing parameters for 3PL, shape (J,).

    Notes
    -----
    Default initialization:
    - a: 1.0 for all items (Rasch) or 1.0 (2PL)
    - b: logit(1 - p_value) where p_value is proportion correct
    """
    J = X.shape[1]

    # Default: a = 1 for all items
    if model == "rasch":
        a = np.ones(J, dtype=np.float64)
    else:
        # 2PL: start with a = 1, will be estimated
        if start is not None and "a" in start:
            a = start["a"].copy()
        else:
            a = np.ones(J, dtype=np.float64)

    # Default: b from p-values
    if start is not None and "b" in start:
        b = start["b"].copy()
    else:
        p_values = compute_item_pvalues(X, mask_obs)
        # b = -logit(p) = log((1-p)/p)
        # This gives higher b for harder items (lower p)
        b = np.log((1 - p_values) / p_values)
        # Clip to reasonable range
        b = np.clip(b, -5.0, 5.0)

    # Default: c for 3PL
    if model == "3pl":
        if start is not None and "c" in start:
            c = start["c"].copy()
        else:
            c = np.full(J, 0.2, dtype=np.float64)
    else:
        c = None

    return a, b, c


# =============================================================================
# Main EM Driver
# =============================================================================


def fit_mml_em(
    X: np.ndarray,
    mask_obs: np.ndarray,
    model: str,
    technical: dict[str, Any],
    start: dict | None,
    fixed: dict | None,
    priors: Any | None,
    constraints: dict[str, Any],
    item_names: list[str],
    person_names: list[str],
) -> FitResult:
    """
    Fit IRT model using MML-EM algorithm.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    model : str
        Model type ("rasch", "2pl", or "3pl").
    technical : dict
        Technical parameters (merged with defaults).
    start : dict or None
        Starting values.
    fixed : dict or None
        Fixed parameter masks.
    priors : dict or DataFrame or None
        Item parameter priors.
    constraints : dict
        Constraints (e.g., center_b).
    item_names : list[str]
        Item names.
    person_names : list[str]
        Person names.

    Returns
    -------
    FitResult
        Fitted model result.
    """
    # Import M-step functions here to avoid circular imports
    from ..mstep import (
        update_item_rasch_newton,
        update_item_2pl_lbfgsb,
        update_item_3pl_lbfgsb,
    )

    N, J = X.shape

    # Extract technical parameters
    quadpts = technical["quadpts"]
    theta_lo = technical["theta_lo"]
    theta_hi = technical["theta_hi"]
    prior_spec = technical["prior"]
    max_iter = technical["max_iter"]
    tol = technical["tol"]
    a_bounds = technical["a_bounds"]
    b_bounds = technical["b_bounds"]
    c_bounds = technical.get("c_bounds", (0.0, 0.35))
    mstep_max_iter = technical["mstep_max_iter"]

    # Parse prior specification
    prior_kind, prior_mean, prior_sd = prior_spec

    # Create theta grid and log-prior
    theta = make_theta_grid(quadpts, theta_lo, theta_hi)
    log_prior = prior_logpdf(theta, kind=prior_kind, mean=prior_mean, sd=prior_sd)

    # Normalize log-prior for integration (add log of grid spacing)
    dx = theta[1] - theta[0]
    log_prior = log_prior + np.log(dx)

    # Validate and process starting values
    validated_start = validate_start_params(start, J, N, model, "mml_em")

    # Initialize parameters
    a, b, c = initialize_params_mml(X, mask_obs, model, validated_start)

    # Validate and process fixed parameters
    validated_fixed = validate_fixed_params(fixed, J, validated_start)

    # Set up fixed masks
    if validated_fixed is not None and "b" in validated_fixed:
        fixed_b = validated_fixed["b"]
    else:
        fixed_b = np.zeros(J, dtype=bool)

    if validated_fixed is not None and "a" in validated_fixed:
        fixed_a = validated_fixed["a"]
    else:
        fixed_a = np.zeros(J, dtype=bool)

    if validated_fixed is not None and "c" in validated_fixed:
        if model != "3pl":
            warnings.warn(
                "Ignoring fixed['c'] for non-3PL model.",
                UserWarning,
                stacklevel=2,
            )
            fixed_c = np.zeros(J, dtype=bool)
        else:
            fixed_c = validated_fixed["c"]
    else:
        fixed_c = np.zeros(J, dtype=bool)

    # For Rasch, all a are fixed at 1
    if model == "rasch":
        fixed_a = np.ones(J, dtype=bool)

    # Parse and validate item priors
    parsed_priors = parse_item_priors(
        priors=priors,
        item_names=item_names,
        model=model,
        a_bounds=a_bounds,
        b_bounds=b_bounds,
        c_bounds=c_bounds,
    )

    if model == "3pl" and not any(parsed_priors["c"]):
        default_c_prior = {
            "kind": "gamma",
            "params": validate_prior_params("gamma", {"shape": 2.0, "scale": 0.05}),
        }
        parsed_priors["c"] = [default_c_prior for _ in range(J)]

    # History tracking
    history = {"loglik": [], "max_param_change": []}

    # EM iterations
    converged = False
    loglik_old = -np.inf
    loglik_change = np.inf
    max_param_change = np.inf

    for iteration in range(max_iter):
        # Store old parameters for convergence check
        a_old = a.copy()
        b_old = b.copy()
        c_old = c.copy() if c is not None else None

        # E-step
        logL_nk = compute_loglik_nk(X, mask_obs, theta, a, b, c=c)
        w_nk, log_evidence_n = posterior_weights(logL_nk, log_prior)

        # Compute marginal log-likelihood
        loglik = log_evidence_n.sum()
        history["loglik"].append(loglik)
        loglik_change = abs(loglik - loglik_old)

        # Check for NaN (numerical issues)
        if not np.isfinite(loglik):
            warnings.warn(
                f"MML-EM: Log-likelihood became non-finite at iteration {iteration}. "
                "This may indicate numerical issues. Try different starting values "
                "or check for extreme data patterns.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # M-step: compute sufficient statistics
        N_jk, R_jk = sufficient_stats_all_items(X, mask_obs, w_nk)

        # M-step: update item parameters
        for j in range(J):
            prior_a = parsed_priors["a"][j]
            prior_b = parsed_priors["b"][j]
            prior_c = parsed_priors["c"][j]
            if model == "rasch":
                # Update b only (a fixed at 1)
                if not fixed_b[j]:
                    b[j] = update_item_rasch_newton(
                        theta=theta,
                        N_k=N_jk[j],
                        R_k=R_jk[j],
                        b0=b[j],
                        bounds=b_bounds,
                        max_iter=mstep_max_iter,
                        prior=prior_b,
                    )
            elif model == "2pl":
                if not fixed_a[j] or not fixed_b[j]:
                    a_new, b_new = update_item_2pl_lbfgsb(
                        theta=theta,
                        N_k=N_jk[j],
                        R_k=R_jk[j],
                        a0=a[j] if not fixed_a[j] else a[j],
                        b0=b[j] if not fixed_b[j] else b[j],
                        a_bounds=a_bounds if not fixed_a[j] else (a[j], a[j]),
                        b_bounds=b_bounds if not fixed_b[j] else (b[j], b[j]),
                        prior_a=prior_a,
                        prior_b=prior_b,
                    )
                    if not fixed_a[j]:
                        a[j] = a_new
                    if not fixed_b[j]:
                        b[j] = b_new
            else:  # 3PL
                if not fixed_a[j] or not fixed_b[j] or not fixed_c[j]:
                    a_new, b_new, c_new = update_item_3pl_lbfgsb(
                        theta=theta,
                        N_k=N_jk[j],
                        R_k=R_jk[j],
                        a0=a[j],
                        b0=b[j],
                        c0=c[j],
                        a_bounds=a_bounds if not fixed_a[j] else (a[j], a[j]),
                        b_bounds=b_bounds if not fixed_b[j] else (b[j], b[j]),
                        c_bounds=c_bounds if not fixed_c[j] else (c[j], c[j]),
                        prior_a=prior_a,
                        prior_b=prior_b,
                        prior_c=prior_c,
                    )
                    if not fixed_a[j]:
                        a[j] = a_new
                    if not fixed_b[j]:
                        b[j] = b_new
                    if not fixed_c[j]:
                        c[j] = c_new

        # Apply constraints (e.g., center difficulties)
        if constraints.get("center_b", False):
            est_mask = ~fixed_b
            b = center_difficulties(b, est_mask)

        # Compute parameter change
        max_change_b = np.max(np.abs(b - b_old))
        if model == "2pl":
            max_change_a = np.max(np.abs(a - a_old))
            max_param_change = max(max_change_a, max_change_b)
        elif model == "3pl":
            max_change_a = np.max(np.abs(a - a_old))
            max_change_c = np.max(np.abs(c - c_old))
            max_param_change = max(max_change_a, max_change_b, max_change_c)
        else:
            max_param_change = max_change_b

        history["max_param_change"].append(max_param_change)

        # Check convergence
        if iteration > 0 and loglik_change < tol and max_param_change < tol:
            converged = True
            break

        loglik_old = loglik

    # Final E-step to get posterior weights for scoring
    logL_nk = compute_loglik_nk(X, mask_obs, theta, a, b, c=c)
    w_nk_final, _ = posterior_weights(logL_nk, log_prior)

    if not converged:
        warnings.warn(
            f"MML-EM did not converge in {max_iter} iterations. "
            f"Final log-likelihood change: {loglik_change:.2e}, "
            f"max parameter change: {max_param_change:.2e}",
            RuntimeWarning,
            stacklevel=2,
        )

    # Create scoring function - capture training data
    X_train_data = X.copy()
    mask_train_data = mask_obs.copy()
    theta_grid_data = theta.copy()
    log_prior_data = log_prior.copy()
    a_data = a.copy()
    b_data = b.copy()
    c_data = c.copy() if c is not None else None
    w_nk_data = w_nk_final.copy()
    person_names_data = list(person_names)

    def score_fn(
        X: np.ndarray | None = None,
        method: str = "eap",
        **kwargs: Any,
    ) -> ScoreResult:
        return _score_mml_em(
            X=X,
            method=method,
            X_train=X_train_data,
            mask_obs_train=mask_train_data,
            theta_grid=theta_grid_data,
            log_prior=log_prior_data,
            a=a_data,
            b=b_data,
            c=c_data,
            w_nk_train=w_nk_data,
            person_names_train=person_names_data,
            **kwargs,
        )

    return FitResult(
        model=model,
        estimator="mml_em",
        params={"a": a.copy(), "b": b.copy(), **({"c": c.copy()} if c is not None else {})},
        converged=converged,
        n_iter=iteration + 1,
        loglik=loglik,
        history=history,
        theta_grid=theta,
        log_prior=log_prior,
        posterior_weights=w_nk_final,
        item_names=item_names,
        person_names=person_names,
        mask_obs=mask_obs,
        X=X,
        _score_fn=score_fn,
    )


def _score_mml_em(
    X: np.ndarray | None,
    method: str,
    X_train: np.ndarray,
    mask_obs_train: np.ndarray,
    theta_grid: np.ndarray,
    log_prior: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None,
    w_nk_train: np.ndarray,
    person_names_train: list[str],
    **kwargs: Any,
) -> ScoreResult:
    """
    Score persons using MML-EM posterior.

    Parameters
    ----------
    X : np.ndarray or None
        Data to score. If None, use training data.
    method : str
        Scoring method: "eap", "map", or "mle".
    X_train, mask_obs_train : np.ndarray
        Training data and mask.
    theta_grid : np.ndarray
        Quadrature points.
    log_prior : np.ndarray
        Log-prior values.
    a, b : np.ndarray
        Item parameters.
    w_nk_train : np.ndarray
        Posterior weights from training.
    person_names_train : list[str]
        Training person names.
    **kwargs
        Additional arguments.

    Returns
    -------
    ScoreResult
        Scoring results.
    """
    # Import scoring functions here to avoid circular imports
    from ..scoring import score_eap_from_w, score_map_from_w, score_mle_newton

    method = method.lower()
    valid_methods = ("eap", "map", "mle")
    if method not in valid_methods:
        raise ValueError(
            f"Invalid scoring method '{method}'. Must be one of {valid_methods}."
        )

    if X is None:
        # Use training data
        X_use = X_train
        mask_use = mask_obs_train
        person_names = person_names_train
        w_nk = w_nk_train
    else:
        # Score new data
        from ..data import as_binary_matrix, filter_people_min_items
        X_use, mask_use, _, person_names_new = as_binary_matrix(X)
        
        # Filter persons with too few responses
        X_use, mask_use, person_names, _ = filter_people_min_items(
            X_use, mask_use, person_names_new, min_items=1
        )

        # Compute posterior for new data
        logL_nk = compute_loglik_nk(X_use, mask_use, theta_grid, a, b, c=c)
        w_nk, _ = posterior_weights(logL_nk, log_prior)

    if method == "eap":
        theta_hat, se = score_eap_from_w(w_nk, theta_grid)
    elif method == "map":
        theta_hat = score_map_from_w(w_nk, theta_grid)
        # MAP doesn't have a simple SE formula; compute from posterior
        # SE = sqrt(E[(theta - MAP)^2 | X])
        se = np.sqrt(
            (w_nk * (theta_grid[np.newaxis, :] - theta_hat[:, np.newaxis]) ** 2).sum(
                axis=1
            )
        )
    else:  # mle
        theta_hat, se = score_mle_newton(
            X=X_use,
            mask_obs=mask_use,
            a=a,
            b=b,
            c=c,
            theta0=kwargs.get("theta0", None),
        )

    return ScoreResult(
        theta=theta_hat,
        se=se,
        method=method,
        person_names=person_names,
    )
