"""
M-step solvers for MML-EM algorithm.

This module implements parameter updates for the M-step:
- Rasch: Newton-Raphson for difficulty parameter b
- 2PL: L-BFGS-B optimization for (a, b) jointly
- 3PL: L-BFGS-B optimization for (a, b, c) jointly

The M-step maximizes the expected complete-data log-likelihood
Q(params) = sum_n sum_k w_nk * log P(X_n | theta_k, params)

Using sufficient statistics, this simplifies to optimizing each
item independently.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .item_priors import prior_logpdf_grad_hess

# =============================================================================
# Rasch M-Step (Newton-Raphson for b)
# =============================================================================


def rasch_Q_grad_hess(
    b: float,
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    prior: dict | None = None,
) -> tuple[float, float, float]:
    """
    Compute Q function, gradient, and Hessian for Rasch item.

    Parameters
    ----------
    b : float
        Current difficulty parameter.
    theta : np.ndarray
        Quadrature points, shape (K,).
    N_k : np.ndarray
        Expected counts at each quadrature point, shape (K,).
    R_k : np.ndarray
        Expected correct counts at each quadrature point, shape (K,).

    Returns
    -------
    Q : float
        Q function value (expected complete-data log-likelihood contribution).
    grad : float
        Gradient dQ/db.
    hess : float
        Hessian d^2Q/db^2.

    Notes
    -----
    For Rasch (a=1):
        P(X=1|theta, b) = 1 / (1 + exp(-(theta - b)))
                        = expit(theta - b)

    Q_j = sum_k [R_k * log(p_k) + (N_k - R_k) * log(1 - p_k)]

    Gradient:
        dQ/db = sum_k [R_k * (-1) + (N_k - R_k) * p_k / (1-p_k) * (-1)]
              = sum_k [-R_k + (N_k - R_k) * p_k * (1-p_k)^(-1) * (-1)]
              = sum_k [N_k * p_k - R_k]

    Hessian:
        d^2Q/db^2 = sum_k [N_k * dp/db]
                  = sum_k [N_k * (-1) * p_k * (1 - p_k)]
                  = -sum_k [N_k * p_k * (1 - p_k)]

    The Hessian is negative (Q is concave), so Newton converges.
    """
    # Probability at each quadrature point
    p_k = expit(theta - b)  # P(X=1 | theta_k, b)
    q_k = 1 - p_k

    # Clip for numerical stability
    eps = 1e-12
    p_k = np.clip(p_k, eps, 1 - eps)
    q_k = np.clip(q_k, eps, 1 - eps)

    # Q function
    Q = np.sum(R_k * np.log(p_k) + (N_k - R_k) * np.log(q_k))

    # Gradient: dQ/db = sum_k [N_k * p_k - R_k]
    grad = np.sum(N_k * p_k - R_k)

    # Hessian: d^2Q/db^2 = -sum_k [N_k * p_k * q_k]
    hess = -np.sum(N_k * p_k * q_k)

    if prior is not None:
        logp, grad_p, hess_p = prior_logpdf_grad_hess(
            prior["kind"], b, prior["params"]
        )
        Q += logp
        grad += grad_p
        hess += hess_p

    return Q, grad, hess


def update_item_rasch_newton(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    b0: float,
    bounds: tuple[float, float],
    max_iter: int = 25,
    tol: float = 1e-8,
    prior: dict | None = None,
) -> float:
    """
    Update Rasch difficulty parameter using Newton-Raphson.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature points, shape (K,).
    N_k : np.ndarray
        Expected counts at each quadrature point, shape (K,).
    R_k : np.ndarray
        Expected correct counts at each quadrature point, shape (K,).
    b0 : float
        Starting value for b.
    bounds : tuple[float, float]
        Lower and upper bounds for b.
    max_iter : int, default=25
        Maximum Newton iterations.
    tol : float, default=1e-8
        Convergence tolerance for gradient.

    Returns
    -------
    float
        Updated difficulty parameter.

    Notes
    -----
    Uses Newton-Raphson with step damping and bounds enforcement.
    The update is: b_new = b - grad / hess

    Since hess < 0 (Q is concave), this moves in the direction of
    increasing Q when grad > 0 (need to decrease b) or grad < 0
    (need to increase b).
    """
    b_lo, b_hi = bounds
    b = np.clip(b0, b_lo, b_hi)

    for _ in range(max_iter):
        _, grad, hess = rasch_Q_grad_hess(b, theta, N_k, R_k, prior=prior)

        # Check convergence
        if abs(grad) < tol:
            break

        # Newton step (hess is negative, so we need to handle signs)
        if abs(hess) < 1e-12:
            # Hessian too small, use gradient descent with small step
            step = -0.1 * np.sign(grad)
        else:
            step = -grad / hess

        # Damping for large steps
        max_step = 2.0
        if abs(step) > max_step:
            step = max_step * np.sign(step)

        # Update with bounds checking
        b_new = b + step
        b_new = np.clip(b_new, b_lo, b_hi)

        # If at boundary and gradient points outward, stop
        if (b_new == b_lo and grad < 0) or (b_new == b_hi and grad > 0):
            break

        b = b_new

    return float(b)


# =============================================================================
# 2PL M-Step (L-BFGS-B for a and b)
# =============================================================================


def twopl_Q_and_grad(
    params: np.ndarray,
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    prior_a: dict | None = None,
    prior_b: dict | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute negative Q and gradient for 2PL item.

    Parameters
    ----------
    params : np.ndarray
        Parameters [a, b].
    theta : np.ndarray
        Quadrature points, shape (K,).
    N_k : np.ndarray
        Expected counts, shape (K,).
    R_k : np.ndarray
        Expected correct counts, shape (K,).

    Returns
    -------
    neg_Q : float
        Negative Q function value (for minimization).
    neg_grad : np.ndarray
        Negative gradient [d(-Q)/da, d(-Q)/db].

    Notes
    -----
    For 2PL:
        P(X=1|theta, a, b) = 1 / (1 + exp(-a*(theta - b)))
                           = expit(a * (theta - b))

    Q_j = sum_k [R_k * log(p_k) + (N_k - R_k) * log(1 - p_k)]

    Gradient w.r.t. a:
        dQ/da = sum_k [(R_k - N_k * p_k) * (theta - b)]

    Gradient w.r.t. b:
        dQ/db = sum_k [(R_k - N_k * p_k) * (-a)]
              = -a * sum_k [R_k - N_k * p_k]
    """
    a, b = params

    # Probability at each quadrature point
    z = a * (theta - b)
    p_k = expit(z)
    q_k = 1 - p_k

    # Clip for numerical stability
    eps = 1e-12
    p_k = np.clip(p_k, eps, 1 - eps)
    q_k = np.clip(q_k, eps, 1 - eps)

    # Q function
    Q = np.sum(R_k * np.log(p_k) + (N_k - R_k) * np.log(q_k))

    # Common term: R_k - N_k * p_k (residual)
    residual = R_k - N_k * p_k

    # Gradient w.r.t. a
    dQ_da = np.sum(residual * (theta - b))

    # Gradient w.r.t. b
    dQ_db = -a * np.sum(residual)

    if prior_a is not None:
        logp_a, grad_a, _ = prior_logpdf_grad_hess(
            prior_a["kind"], a, prior_a["params"]
        )
        Q += logp_a
        dQ_da += grad_a
    if prior_b is not None:
        logp_b, grad_b, _ = prior_logpdf_grad_hess(
            prior_b["kind"], b, prior_b["params"]
        )
        Q += logp_b
        dQ_db += grad_b

    # Return negative for minimization
    return -Q, np.array([-dQ_da, -dQ_db])


def update_item_2pl_lbfgsb(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    a0: float,
    b0: float,
    a_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
    prior_a: dict | None = None,
    prior_b: dict | None = None,
) -> tuple[float, float]:
    """
    Update 2PL item parameters using L-BFGS-B.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature points, shape (K,).
    N_k : np.ndarray
        Expected counts, shape (K,).
    R_k : np.ndarray
        Expected correct counts, shape (K,).
    a0 : float
        Starting value for discrimination.
    b0 : float
        Starting value for difficulty.
    a_bounds : tuple[float, float]
        Bounds for discrimination parameter.
    b_bounds : tuple[float, float]
        Bounds for difficulty parameter.

    Returns
    -------
    a_hat : float
        Updated discrimination parameter.
    b_hat : float
        Updated difficulty parameter.

    Notes
    -----
    Uses scipy.optimize.minimize with L-BFGS-B method, which handles
    box constraints efficiently.
    """
    x0 = np.array([a0, b0])
    bounds = [a_bounds, b_bounds]

    result = minimize(
        twopl_Q_and_grad,
        x0,
        args=(theta, N_k, R_k, prior_a, prior_b),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-8, "gtol": 1e-6},
    )

    a_hat, b_hat = result.x
    return float(a_hat), float(b_hat)


# =============================================================================
# 3PL M-Step (L-BFGS-B for a, b, c)
# =============================================================================


def threepl_Q_and_grad(
    params: np.ndarray,
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    prior_a: dict | None = None,
    prior_b: dict | None = None,
    prior_c: dict | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute negative Q and gradient for 3PL item.
    """
    a, b, c = params

    z = a * (theta - b)
    p_2pl = expit(z)
    p = c + (1.0 - c) * p_2pl

    # Clip for numerical stability
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    Q = np.sum(R_k * np.log(p) + (N_k - R_k) * np.log(1 - p))

    residual = R_k - N_k * p
    denom = p * (1.0 - p)
    common = residual / denom

    dp2 = p_2pl * (1.0 - p_2pl)
    dpa = (1.0 - c) * dp2 * (theta - b)
    dpb = (1.0 - c) * dp2 * (-a)
    dpc = 1.0 - p_2pl

    dQ_da = np.sum(common * dpa)
    dQ_db = np.sum(common * dpb)
    dQ_dc = np.sum(common * dpc)

    if prior_a is not None:
        logp_a, grad_a, _ = prior_logpdf_grad_hess(
            prior_a["kind"], a, prior_a["params"]
        )
        Q += logp_a
        dQ_da += grad_a
    if prior_b is not None:
        logp_b, grad_b, _ = prior_logpdf_grad_hess(
            prior_b["kind"], b, prior_b["params"]
        )
        Q += logp_b
        dQ_db += grad_b
    if prior_c is not None:
        logp_c, grad_c, _ = prior_logpdf_grad_hess(
            prior_c["kind"], c, prior_c["params"]
        )
        Q += logp_c
        dQ_dc += grad_c

    return -Q, np.array([-dQ_da, -dQ_db, -dQ_dc])


def update_item_3pl_lbfgsb(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    a0: float,
    b0: float,
    c0: float,
    a_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
    c_bounds: tuple[float, float],
    prior_a: dict | None = None,
    prior_b: dict | None = None,
    prior_c: dict | None = None,
) -> tuple[float, float, float]:
    """
    Update 3PL item parameters using L-BFGS-B.
    """
    x0 = np.array([a0, b0, c0])
    bounds = [a_bounds, b_bounds, c_bounds]

    result = minimize(
        threepl_Q_and_grad,
        x0,
        args=(theta, N_k, R_k, prior_a, prior_b, prior_c),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-8, "gtol": 1e-6},
    )

    a_hat, b_hat, c_hat = result.x
    return float(a_hat), float(b_hat), float(c_hat)


# =============================================================================
# Alternative: Coordinate Descent for 2PL
# =============================================================================


def update_item_2pl_coordinate(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    a0: float,
    b0: float,
    a_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
    max_iter: int = 10,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """
    Update 2PL item parameters using coordinate descent.

    Alternative to L-BFGS-B that alternates between updating a and b.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature points, shape (K,).
    N_k : np.ndarray
        Expected counts, shape (K,).
    R_k : np.ndarray
        Expected correct counts, shape (K,).
    a0 : float
        Starting value for discrimination.
    b0 : float
        Starting value for difficulty.
    a_bounds : tuple[float, float]
        Bounds for discrimination.
    b_bounds : tuple[float, float]
        Bounds for difficulty.
    max_iter : int, default=10
        Maximum coordinate descent iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    a_hat : float
        Updated discrimination parameter.
    b_hat : float
        Updated difficulty parameter.
    """
    a = np.clip(a0, a_bounds[0], a_bounds[1])
    b = np.clip(b0, b_bounds[0], b_bounds[1])

    for _ in range(max_iter):
        a_old, b_old = a, b

        # Update b given a (similar to Rasch update)
        b = _update_b_given_a(theta, N_k, R_k, a, b, b_bounds)

        # Update a given b
        a = _update_a_given_b(theta, N_k, R_k, a, b, a_bounds)

        # Check convergence
        if abs(a - a_old) < tol and abs(b - b_old) < tol:
            break

    return float(a), float(b)


def _update_b_given_a(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    a: float,
    b0: float,
    bounds: tuple[float, float],
    max_iter: int = 15,
) -> float:
    """Update b given fixed a using Newton-Raphson."""
    b_lo, b_hi = bounds
    b = np.clip(b0, b_lo, b_hi)

    for _ in range(max_iter):
        z = a * (theta - b)
        p_k = expit(z)
        p_k = np.clip(p_k, 1e-12, 1 - 1e-12)

        # Gradient: dQ/db = -a * sum(R_k - N_k * p_k)
        grad = -a * np.sum(R_k - N_k * p_k)

        # Hessian: d2Q/db2 = -a^2 * sum(N_k * p_k * (1-p_k))
        hess = -a**2 * np.sum(N_k * p_k * (1 - p_k))

        if abs(grad) < 1e-8:
            break

        if abs(hess) < 1e-12:
            step = -0.1 * np.sign(grad)
        else:
            step = -grad / hess

        step = np.clip(step, -2.0, 2.0)
        b = np.clip(b + step, b_lo, b_hi)

    return float(b)


def _update_a_given_b(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k: np.ndarray,
    a0: float,
    b: float,
    bounds: tuple[float, float],
    max_iter: int = 15,
) -> float:
    """Update a given fixed b using Newton-Raphson."""
    a_lo, a_hi = bounds
    a = np.clip(a0, a_lo, a_hi)

    for _ in range(max_iter):
        z = a * (theta - b)
        p_k = expit(z)
        p_k = np.clip(p_k, 1e-12, 1 - 1e-12)

        diff = theta - b
        residual = R_k - N_k * p_k

        # Gradient: dQ/da = sum(residual * diff)
        grad = np.sum(residual * diff)

        # Hessian: d2Q/da2 = -sum(N_k * p_k * (1-p_k) * diff^2)
        hess = -np.sum(N_k * p_k * (1 - p_k) * diff**2)

        if abs(grad) < 1e-8:
            break

        if abs(hess) < 1e-12:
            step = 0.1 * np.sign(grad)
        else:
            step = -grad / hess

        step = np.clip(step, -0.5, 0.5)
        a = np.clip(a + step, a_lo, a_hi)

    return float(a)
