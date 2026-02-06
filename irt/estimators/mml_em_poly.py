"""
MML-EM estimation for polytomous IRT models.

Supports PCM, RSM, GRM, GPCM, NRM.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.special import logsumexp, logit as _logit

from ..types import FitResult, ScoreResult
from ..core_poly import (
    prob_pcm,
    prob_rsm,
    prob_grm,
    prob_gpcm,
    prob_nrm,
    log_prob_pcm,
    log_prob_rsm,
    log_prob_grm,
    log_prob_gpcm,
    log_prob_nrm,
)
from ..data import (
    compute_item_category_proportions,
    filter_people_min_items,
)
from ..quadrature import make_theta_grid, prior_logpdf


# =============================================================================
# E-Step
# =============================================================================


def _prob_item_all_theta(
    theta: np.ndarray,
    params: dict[str, Any],
    model: str,
    j: int,
    n_cat: int,
) -> np.ndarray:
    """Compute P(X_j = c | theta_k) for all theta values at once.

    Returns shape (K, n_cat) where K = len(theta).
    """
    if model == "pcm":
        return prob_pcm(theta, params["b"][j], n_cat)
    elif model == "rsm":
        return prob_rsm(theta, params["b"][j], params["tau"], n_cat)
    elif model == "grm":
        return prob_grm(theta, params["a"][j], params["b"][j], n_cat)
    elif model == "gpcm":
        return prob_gpcm(theta, params["a"][j], params["b"][j], n_cat)
    elif model == "nrm":
        return prob_nrm(theta, params["a"][j], params["c"][j], n_cat)
    else:
        raise ValueError(f"Unknown model '{model}'")


def compute_loglik_nk_poly(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    params: dict[str, Any],
    model: str,
    n_categories: np.ndarray,
) -> np.ndarray:
    """
    Compute log P(X_n | theta_k) for polytomous data.

    Vectorized over quadrature points -- probabilities for all theta
    values are computed in a single call per item.

    Returns
    -------
    np.ndarray
        Log-likelihood matrix, shape (N, K).
    """
    N, J = X.shape
    K = len(theta)
    logL_nk = np.zeros((N, K), dtype=np.float64)

    for j in range(J):
        obs_j = mask_obs[:, j]
        if not np.any(obs_j):
            continue

        n_cat = int(n_categories[j])
        # Compute all category probabilities for all quadrature points at once
        p_all = _prob_item_all_theta(theta, params, model, j, n_cat)  # (K, n_cat)
        p_all = np.clip(p_all, 1e-12, 1.0)
        log_p_all = np.log(p_all)  # (K, n_cat)

        # Get observed category for each person who responded to item j
        obs_indices = np.where(obs_j)[0]
        x_j = X[obs_indices, j].astype(int)

        # Gather log-probabilities for observed categories: log_p_all[:, x_j] -> (K, N_obs)
        # We want logL_nk[n, k] += log P(X_nj = x_nj | theta_k)
        # Fancy-index: for each observed person i, pick column x_j[i] from log_p_all
        log_p_selected = log_p_all[:, x_j]  # (K, N_obs)
        logL_nk[obs_indices, :] += log_p_selected.T  # (N_obs, K)

    return logL_nk


def posterior_weights(
    logL_nk: np.ndarray,
    log_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute posterior weights from log-likelihood."""
    log_post_unnorm = logL_nk + log_prior[np.newaxis, :]
    log_evidence_n = logsumexp(log_post_unnorm, axis=1)
    log_w_nk = log_post_unnorm - log_evidence_n[:, np.newaxis]
    w_nk = np.exp(log_w_nk)
    w_nk = w_nk / w_nk.sum(axis=1, keepdims=True)
    return w_nk, log_evidence_n


# =============================================================================
# Sufficient Statistics
# =============================================================================


def sufficient_stats_poly(
    X: np.ndarray,
    mask_obs: np.ndarray,
    w_nk: np.ndarray,
    n_categories: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Compute sufficient statistics for polytomous items.

    Returns
    -------
    N_list : list of np.ndarray
        N_jk for each item j, shape (K,).
    R_list : list of np.ndarray
        R_jk_c for each item j, shape (K, n_cat_j).
    """
    N, J = X.shape
    K = w_nk.shape[1]
    N_list = []
    R_list = []

    for j in range(J):
        n_cat = int(n_categories[j])
        N_k = np.zeros(K, dtype=np.float64)
        R_k_c = np.zeros((K, n_cat), dtype=np.float64)

        obs_j = mask_obs[:, j]
        w_obs = w_nk[obs_j]  # (N_obs, K)
        x_obs = X[obs_j, j].astype(int)  # (N_obs,)

        N_k = w_obs.sum(axis=0)
        for c in range(n_cat):
            mask_c = x_obs == c
            R_k_c[:, c] = (w_obs[mask_c]).sum(axis=0)

        N_list.append(N_k)
        R_list.append(R_k_c)

    return N_list, R_list


# =============================================================================
# Parameter Initialization
# =============================================================================


def initialize_params_poly(
    X: np.ndarray,
    mask_obs: np.ndarray,
    n_categories: np.ndarray,
    model: str,
    start: dict | None = None,
) -> dict[str, Any]:
    """Initialize parameters for polytomous models."""
    J = X.shape[1]
    max_cat = int(np.max(n_categories))
    K_step = max_cat - 1

    props = compute_item_category_proportions(X, mask_obs, n_categories)

    params: dict[str, Any] = {}

    def _logit_cum(cum_prop: float, fallback: float = 0.0) -> float:
        """Convert a cumulative proportion to a logit, with fallback."""
        if cum_prop < 1e-6 or cum_prop > 1 - 1e-6:
            return fallback
        return float(_logit(np.clip(cum_prop, 0.01, 0.99)))

    if model == "pcm":
        b = np.zeros((J, K_step), dtype=np.float64)
        for j in range(J):
            n_cat = n_categories[j]
            cum = np.cumsum(props[j, :n_cat])
            for k in range(n_cat - 1):
                b[j, k] = _logit_cum(cum[k])
        params["b"] = b

    elif model == "rsm":
        b = np.zeros(J, dtype=np.float64)
        for j in range(J):
            n_cat = n_categories[j]
            cum = np.cumsum(props[j, :n_cat])
            for k in range(n_cat - 1):
                val = _logit_cum(cum[k])
                b[j] += val / (n_cat - 1)
        b = np.clip(b, -4, 4)
        tau = np.linspace(-0.5, 0.5, K_step)
        params["b"] = b
        params["tau"] = tau

    elif model == "grm":
        a = np.ones(J, dtype=np.float64)
        b = np.zeros((J, K_step), dtype=np.float64)
        for j in range(J):
            n_cat = n_categories[j]
            cum = np.cumsum(props[j, :n_cat])
            for k in range(n_cat - 1):
                b[j, k] = _logit_cum(cum[k], fallback=(k - K_step / 2) * 0.5)
            b[j, :] = np.sort(b[j, :])
        params["a"] = a
        params["b"] = b

    elif model == "gpcm":
        a = np.ones(J, dtype=np.float64)
        b = np.zeros((J, K_step), dtype=np.float64)
        for j in range(J):
            n_cat = n_categories[j]
            cum = np.cumsum(props[j, :n_cat])
            for k in range(n_cat - 1):
                b[j, k] = _logit_cum(cum[k])
        params["a"] = a
        params["b"] = b

    elif model == "nrm":
        a = np.zeros((J, max_cat), dtype=np.float64)
        c = np.zeros((J, max_cat), dtype=np.float64)
        for j in range(J):
            n_cat = n_categories[j]
            p0 = np.clip(props[j, 0], 1e-6, 1.0)
            for x in range(1, n_cat):
                p_x = np.clip(props[j, x], 0.01, 0.99)
                c[j, x] = np.log(p_x / p0)
            a[j, 1:n_cat] = np.linspace(0.3, 1.0, n_cat - 1)
        params["a"] = a
        params["c"] = c

    if start is not None:
        for key in ("a", "b", "c", "tau"):
            if key in start:
                params[key] = np.asarray(start[key], dtype=np.float64).copy()

    return params


# =============================================================================
# Main EM Driver
# =============================================================================


def fit_mml_em_poly(
    X: np.ndarray,
    mask_obs: np.ndarray,
    n_categories: np.ndarray,
    model: str,
    technical: dict[str, Any],
    start: dict | None,
    item_names: list[str],
    person_names: list[str],
) -> FitResult:
    """Fit polytomous IRT model using MML-EM."""
    from ..mstep_poly import (
        update_item_pcm,
        update_item_gpcm,
        update_item_grm,
        update_item_rsm,
        update_tau_rsm,
        update_item_nrm,
    )

    N, J = X.shape
    max_cat = int(np.max(n_categories))
    K_step = max_cat - 1

    quadpts = technical["quadpts"]
    theta_lo = technical["theta_lo"]
    theta_hi = technical["theta_hi"]
    prior_spec = technical["prior"]
    max_iter = technical["max_iter"]
    tol = technical["tol"]
    a_bounds = technical.get("a_bounds", (0.25, 4.0))
    b_bounds = technical.get("b_bounds", (-6.0, 6.0))
    mstep_max_iter = technical.get("mstep_max_iter", 100)

    prior_kind, prior_mean, prior_sd = prior_spec
    theta = make_theta_grid(quadpts, theta_lo, theta_hi)
    log_prior = prior_logpdf(theta, kind=prior_kind, mean=prior_mean, sd=prior_sd)
    dx = theta[1] - theta[0]
    log_prior = log_prior + np.log(dx)

    params = initialize_params_poly(X, mask_obs, n_categories, model, start)

    history = {"loglik": [], "max_param_change": []}
    converged = False
    loglik_old = -np.inf
    loglik_change = np.inf
    max_param_change = np.inf

    for iteration in range(max_iter):
        params_old = {k: v.copy() if hasattr(v, "copy") else v for k, v in params.items()}

        # E-step
        logL_nk = compute_loglik_nk_poly(
            X, mask_obs, theta, params, model, n_categories
        )
        w_nk, log_evidence_n = posterior_weights(logL_nk, log_prior)
        loglik = log_evidence_n.sum()
        history["loglik"].append(loglik)
        loglik_change = abs(loglik - loglik_old)

        if not np.isfinite(loglik):
            warnings.warn(
                f"MML-EM (polytomous): Log-likelihood non-finite at iteration {iteration}.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

        # M-step: sufficient statistics
        N_list, R_list = sufficient_stats_poly(X, mask_obs, w_nk, n_categories)

        # M-step: update parameters
        if model == "pcm":
            for j in range(J):
                n_cat = int(n_categories[j])
                k_step_j = n_cat - 1
                b_j = params["b"][j, :k_step_j]
                b_new = update_item_pcm(
                    theta, N_list[j], R_list[j], b_j, b_bounds, mstep_max_iter
                )
                params["b"][j, :k_step_j] = b_new
            max_param_change = np.max(np.abs(params["b"] - params_old["b"]))

        elif model == "rsm":
            for j in range(J):
                params["b"][j] = update_item_rsm(
                    theta, N_list[j], R_list[j],
                    params["b"][j], params["tau"],
                    b_bounds, mstep_max_iter,
                )
            params["tau"] = update_tau_rsm(
                theta, R_list, params["b"], params["tau"],
                (-4.0, 4.0), mstep_max_iter,
            )
            max_param_change = max(
                np.max(np.abs(params["b"] - params_old["b"])),
                np.max(np.abs(params["tau"] - params_old["tau"])),
            )

        elif model == "grm":
            for j in range(J):
                n_cat = int(n_categories[j])
                k_step_j = n_cat - 1
                a_new, b_new = update_item_grm(
                    theta, N_list[j], R_list[j],
                    params["a"][j], params["b"][j, :k_step_j],
                    a_bounds, b_bounds, mstep_max_iter,
                )
                params["a"][j] = a_new
                params["b"][j, :k_step_j] = b_new
            max_param_change = max(
                np.max(np.abs(params["a"] - params_old["a"])),
                np.max(np.abs(params["b"] - params_old["b"])),
            )

        elif model == "gpcm":
            for j in range(J):
                n_cat = int(n_categories[j])
                k_step_j = n_cat - 1
                a_new, b_new = update_item_gpcm(
                    theta, N_list[j], R_list[j],
                    params["a"][j], params["b"][j, :k_step_j],
                    a_bounds, b_bounds, mstep_max_iter,
                )
                params["a"][j] = a_new
                params["b"][j, :k_step_j] = b_new
            max_param_change = max(
                np.max(np.abs(params["a"] - params_old["a"])),
                np.max(np.abs(params["b"] - params_old["b"])),
            )

        elif model == "nrm":
            for j in range(J):
                n_cat = int(n_categories[j])
                a_new, c_new = update_item_nrm(
                    theta, N_list[j], R_list[j],
                    params["a"][j, :n_cat], params["c"][j, :n_cat],
                    n_cat, b_bounds, mstep_max_iter,
                )
                params["a"][j, :n_cat] = a_new
                params["c"][j, :n_cat] = c_new
            max_param_change = max(
                np.max(np.abs(params["a"] - params_old["a"])),
                np.max(np.abs(params["c"] - params_old["c"])),
            )

        history["max_param_change"].append(max_param_change)

        # RSM: center tau for identification
        if model == "rsm":
            params["tau"] = params["tau"] - np.mean(params["tau"])

        if iteration > 0 and loglik_change < tol and max_param_change < tol:
            converged = True
            break
        loglik_old = loglik

    # Final E-step
    logL_nk = compute_loglik_nk_poly(
        X, mask_obs, theta, params, model, n_categories
    )
    w_nk_final, _ = posterior_weights(logL_nk, log_prior)

    if not converged:
        warnings.warn(
            f"MML-EM (polytomous) did not converge in {max_iter} iterations.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Build params dict for FitResult (copy arrays)
    result_params: dict[str, np.ndarray] = {}
    for k, v in params.items():
        result_params[k] = np.asarray(v).copy()

    # Capture training data for scoring closure
    X_train = X.copy()
    mask_train = mask_obs.copy()
    theta_grid = theta.copy()
    log_prior_data = log_prior.copy()
    n_cat_data = n_categories.copy()
    w_nk_train = w_nk_final.copy()
    person_names_data = list(person_names)

    # Scoring function (X is the data to score, None = use training data)
    def score_fn(
        X: np.ndarray | None = None,
        method: str = "eap",
        **kwargs: Any,
    ) -> ScoreResult:
        return _score_mml_em_poly(
            X=X,
            method=method,
            X_train=X_train,
            mask_obs_train=mask_train,
            theta_grid=theta_grid,
            log_prior=log_prior_data,
            params=result_params,
            model=model,
            n_categories=n_cat_data,
            w_nk_train=w_nk_train,
            person_names_train=person_names_data,
            **kwargs,
        )

    return FitResult(
        model=model,
        estimator="mml_em",
        params=result_params,
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
        n_categories=n_categories,
    )


def _score_mml_em_poly(
    X: np.ndarray | None,
    method: str,
    X_train: np.ndarray,
    mask_obs_train: np.ndarray,
    theta_grid: np.ndarray,
    log_prior: np.ndarray,
    params: dict[str, np.ndarray],
    model: str,
    n_categories: np.ndarray,
    w_nk_train: np.ndarray,
    person_names_train: list[str],
    **kwargs: Any,
) -> ScoreResult:
    """Score persons for polytomous model."""
    from ..scoring import score_eap_from_w, score_map_from_w
    from ..scoring_poly import score_mle_newton_poly

    method = method.lower()
    if method not in ("eap", "map", "mle"):
        raise ValueError(f"Invalid scoring method '{method}'.")

    if X is None:
        X_use = X_train
        mask_use = mask_obs_train
        person_names = person_names_train
        w_nk = w_nk_train
    else:
        from ..data import as_polytomous_matrix
        X_use, mask_use, _, person_names_new, n_cat_use = as_polytomous_matrix(X)
        X_use, mask_use, person_names, _ = filter_people_min_items(
            X_use, mask_use, person_names_new, min_items=1
        )
        logL_nk = compute_loglik_nk_poly(
            X_use, mask_use, theta_grid, params, model, n_cat_use
        )
        w_nk, _ = posterior_weights(logL_nk, log_prior)

    if method == "eap":
        theta_hat, se = score_eap_from_w(w_nk, theta_grid)
    elif method == "map":
        theta_hat = score_map_from_w(w_nk, theta_grid)
        se = np.sqrt(
            (w_nk * (theta_grid[np.newaxis, :] - theta_hat[:, np.newaxis]) ** 2).sum(
                axis=1
            )
        )
    else:
        theta_hat, se = score_mle_newton_poly(
            X_use, mask_use, params, model, n_categories, **kwargs
        )

    return ScoreResult(
        theta=theta_hat,
        se=se,
        method=method,
        person_names=person_names,
    )
