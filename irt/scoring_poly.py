"""
Person ability scoring for polytomous IRT models.

MLE via Newton-Raphson on the polytomous log-likelihood.
"""

from __future__ import annotations

import numpy as np

from .estimators.mml_em_poly import _prob_item_all_theta


def _loglik_grad_hess_poly(
    theta: float,
    X_row: np.ndarray,
    mask_row: np.ndarray,
    params: dict,
    model: str,
    n_categories: np.ndarray,
    eps: float = 1e-7,
) -> tuple[float, float, float]:
    """Compute log-likelihood, gradient, and Hessian for one person.

    Uses central finite differences for the gradient and Hessian with
    respect to theta, leveraging the shared model dispatch helper.
    """
    logL = 0.0
    grad = 0.0
    hess = 0.0

    theta_arr = np.array([theta])
    theta_hi = np.array([theta + eps])
    theta_lo = np.array([theta - eps])
    J = len(mask_row)

    for j in range(J):
        if not mask_row[j]:
            continue
        n_cat = int(n_categories[j])
        c = int(X_row[j])

        p = np.clip(
            _prob_item_all_theta(theta_arr, params, model, j, n_cat)[0],
            1e-12, 1.0 - 1e-12,
        )
        p_hi = np.clip(
            _prob_item_all_theta(theta_hi, params, model, j, n_cat)[0],
            1e-12, 1.0 - 1e-12,
        )
        p_lo = np.clip(
            _prob_item_all_theta(theta_lo, params, model, j, n_cat)[0],
            1e-12, 1.0 - 1e-12,
        )

        logL += np.log(p[c])

        dp = (p_hi[c] - p_lo[c]) / (2 * eps)
        d2p = (p_hi[c] - 2 * p[c] + p_lo[c]) / (eps**2)

        grad += dp / p[c]
        hess += (d2p / p[c]) - (dp / p[c]) ** 2

    return logL, grad, hess


def score_mle_newton_poly(
    X: np.ndarray,
    mask_obs: np.ndarray,
    params: dict,
    model: str,
    n_categories: np.ndarray,
    theta0: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (-6.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MLE ability estimates for polytomous model."""
    N = X.shape[0]
    theta_lo, theta_hi = bounds

    if theta0 is None:
        theta = np.zeros(N, dtype=np.float64)
    else:
        theta = theta0.copy()

    theta_hat = np.zeros(N, dtype=np.float64)
    se = np.zeros(N, dtype=np.float64)

    for i in range(N):
        if not np.any(mask_obs[i]):
            theta_hat[i] = 0.0
            se[i] = np.inf
            continue

        th = theta[i]
        for _ in range(max_iter):
            _, grad, hess = _loglik_grad_hess_poly(
                th, X[i], mask_obs[i], params, model, n_categories
            )
            if abs(grad) < tol:
                break
            if abs(hess) < 1e-12:
                step = 0.1 * np.sign(grad)
            else:
                step = -grad / hess
            step = np.clip(step, -2.0, 2.0)
            th = np.clip(th + step, theta_lo, theta_hi)

        theta_hat[i] = th

        _, _, hess = _loglik_grad_hess_poly(
            th, X[i], mask_obs[i], params, model, n_categories
        )
        info = max(-hess, 1e-12)
        se[i] = 1.0 / np.sqrt(info)

    return theta_hat, se
