"""
Person ability scoring functions for IRT models.

This module provides functions for computing person ability estimates:
- EAP (Expected A Posteriori): Posterior mean
- MAP (Maximum A Posteriori): Posterior mode
- MLE (Maximum Likelihood Estimate): Mode of likelihood

For MML-EM fits:
- EAP and MAP can be computed directly from posterior weights
- MLE requires Newton-Raphson optimization

For JMLE fits:
- Theta estimates are computed during fitting
- Scoring new data uses MLE or MAP with Newton-Raphson

Scoring Methods Comparison
--------------------------
EAP:
    - Posterior mean: E[theta | X]
    - Always finite (within grid bounds for MML-EM)
    - Shrinks estimates toward prior mean
    - Recommended for most applications

MAP:
    - Posterior mode: argmax P(theta | X)
    - Can be computed from grid or refined with Newton
    - Less shrinkage than EAP
    - Useful when point estimate at mode is preferred

MLE:
    - Likelihood mode: argmax P(X | theta)
    - Infinite for extreme scores (all correct or all wrong)
    - No shrinkage
    - Standard errors from observed information
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def score_eap_from_w(
    w_nk: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute EAP (Expected A Posteriori) ability estimates from posterior weights.

    EAP = E[theta | X] = sum_k w_nk * theta_k

    Parameters
    ----------
    w_nk : np.ndarray
        Posterior weights, shape (N, K). Rows should sum to 1.
    theta : np.ndarray
        Quadrature points, shape (K,).

    Returns
    -------
    theta_hat : np.ndarray
        EAP estimates, shape (N,).
    se : np.ndarray
        Standard errors (posterior SD), shape (N,).

    Notes
    -----
    The standard error is the posterior standard deviation:
        SE = sqrt(Var[theta | X]) = sqrt(E[(theta - EAP)^2 | X])
           = sqrt(sum_k w_nk * (theta_k - EAP)^2)

    EAP estimates are always within the bounds of the theta grid.

    Examples
    --------
    >>> w_nk = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3]])
    >>> theta = np.array([-1, 0, 1])
    >>> theta_hat, se = score_eap_from_w(w_nk, theta)
    >>> theta_hat  # doctest: +SKIP
    array([0., 0.])
    """
    # EAP = sum_k w_nk * theta_k
    theta_hat = np.sum(w_nk * theta[np.newaxis, :], axis=1)

    # Posterior variance = E[(theta - EAP)^2 | X]
    deviations = theta[np.newaxis, :] - theta_hat[:, np.newaxis]
    posterior_var = np.sum(w_nk * deviations**2, axis=1)
    se = np.sqrt(posterior_var)

    return theta_hat, se


def score_map_from_w(
    w_nk: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute MAP (Maximum A Posteriori) ability estimates from posterior weights.

    MAP = argmax_k P(theta_k | X) = theta[argmax_k(w_nk)]

    Parameters
    ----------
    w_nk : np.ndarray
        Posterior weights, shape (N, K).
    theta : np.ndarray
        Quadrature points, shape (K,).

    Returns
    -------
    np.ndarray
        MAP estimates, shape (N,).

    Notes
    -----
    This is a grid-based MAP that returns the theta value with highest
    posterior weight. For a more refined estimate, use score_map_newton.

    For unimodal posteriors, this equals the true MAP within grid resolution.

    Examples
    --------
    >>> w_nk = np.array([[0.1, 0.8, 0.1], [0.3, 0.4, 0.3]])
    >>> theta = np.array([-1, 0, 1])
    >>> score_map_from_w(w_nk, theta)
    array([0., 0.])
    """
    # Find index of maximum weight for each person
    max_idx = np.argmax(w_nk, axis=1)
    return theta[max_idx]


def score_mle_newton(
    X: np.ndarray,
    mask_obs: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (-6.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MLE ability estimates using Newton-Raphson.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).
    theta0 : np.ndarray or None, optional
        Starting values, shape (N,). If None, starts from 0.
    max_iter : int, default=25
        Maximum Newton iterations per person.
    tol : float, default=1e-6
        Convergence tolerance.
    bounds : tuple[float, float], default=(-6.0, 6.0)
        Bounds for theta estimates.

    Returns
    -------
    theta_hat : np.ndarray
        MLE estimates, shape (N,).
    se : np.ndarray
        Standard errors, shape (N,).

    Notes
    -----
    For persons with extreme scores (all correct or all wrong), the MLE
    is at +/- infinity. We return the bound value and SE = inf.

    Standard errors are computed from the observed Fisher information:
        SE = 1 / sqrt(I(theta)) = 1 / sqrt(sum_j a_j^2 * p_j * (1-p_j))

    Examples
    --------
    >>> X = np.array([[1, 0, 1], [0, 1, 0]])
    >>> mask = np.ones_like(X, dtype=bool)
    >>> a = np.array([1.0, 1.0, 1.0])
    >>> b = np.array([-1.0, 0.0, 1.0])
    >>> theta, se = score_mle_newton(X, mask, a, b)
    """
    N = X.shape[0]
    theta_lo, theta_hi = bounds

    # Initialize
    if theta0 is None:
        theta = np.zeros(N, dtype=np.float64)
    else:
        theta = theta0.copy()

    theta_hat = np.zeros(N, dtype=np.float64)
    se = np.zeros(N, dtype=np.float64)

    for i in range(N):
        obs = mask_obs[i]
        if not np.any(obs):
            theta_hat[i] = 0.0
            se[i] = np.inf
            continue

        x_obs = X[i, obs]
        a_obs = a[obs]
        b_obs = b[obs]
        c_obs = c[obs] if c is not None else None

        # Check for extreme scores
        sum_score = x_obs.sum()
        max_score = len(x_obs)
        if sum_score == 0:
            theta_hat[i] = theta_lo
            se[i] = np.inf
            continue
        elif sum_score == max_score:
            theta_hat[i] = theta_hi
            se[i] = np.inf
            continue

        # Newton-Raphson
        th = theta[i]
        for _ in range(max_iter):
            z = a_obs * (th - b_obs)
            p_2pl = expit(z)
            if c_obs is None:
                p = p_2pl
                dp_dtheta = a_obs * p * (1 - p)
            else:
                p = c_obs + (1.0 - c_obs) * p_2pl
                dp_dtheta = (1.0 - c_obs) * a_obs * p_2pl * (1 - p_2pl)
            p = np.clip(p, 1e-12, 1 - 1e-12)

            denom = p * (1 - p)
            # Gradient: dL/dtheta = sum_j (x_j - p_j) * dp/dtheta / (p_j * (1-p_j))
            grad = np.sum((x_obs - p) * dp_dtheta / denom)

            # Hessian (Fisher information approximation)
            hess = -np.sum((dp_dtheta**2) / denom)

            if abs(grad) < tol:
                break

            if abs(hess) < 1e-12:
                step = 0.1 * np.sign(grad)
            else:
                step = -grad / hess

            step = np.clip(step, -2.0, 2.0)
            th = np.clip(th + step, theta_lo, theta_hi)

        theta_hat[i] = th

        # Compute SE from information at MLE
        z = a_obs * (th - b_obs)
        p_2pl = expit(z)
        if c_obs is None:
            p = p_2pl
            dp_dtheta = a_obs * p * (1 - p)
        else:
            p = c_obs + (1.0 - c_obs) * p_2pl
            dp_dtheta = (1.0 - c_obs) * a_obs * p_2pl * (1 - p_2pl)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        info = np.sum((dp_dtheta**2) / (p * (1 - p)))
        se[i] = 1.0 / np.sqrt(max(info, 1e-12))

    return theta_hat, se


def score_map_newton(
    X: np.ndarray,
    mask_obs: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    theta0: np.ndarray | None = None,
    max_iter: int = 25,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (-6.0, 6.0),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MAP ability estimates using Newton-Raphson with normal prior.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).
    prior_mean : float, default=0.0
        Mean of the normal prior on theta.
    prior_sd : float, default=1.0
        Standard deviation of the normal prior.
    theta0 : np.ndarray or None, optional
        Starting values, shape (N,).
    max_iter : int, default=25
        Maximum Newton iterations.
    tol : float, default=1e-6
        Convergence tolerance.
    bounds : tuple[float, float], default=(-6.0, 6.0)
        Bounds for theta estimates.

    Returns
    -------
    theta_hat : np.ndarray
        MAP estimates, shape (N,).
    se : np.ndarray
        Standard errors, shape (N,).

    Notes
    -----
    The MAP is the mode of the posterior:
        P(theta | X) ∝ P(X | theta) * P(theta)

    Log-posterior:
        log P(theta | X) = sum_j log P(X_j | theta) + log P(theta) + const

    With normal prior N(mu, sigma^2):
        log P(theta) = -(theta - mu)^2 / (2 * sigma^2) + const

    The prior contribution to gradient and Hessian:
        d(log P(theta))/dtheta = -(theta - mu) / sigma^2
        d^2(log P(theta))/dtheta^2 = -1 / sigma^2

    Standard errors are computed from the posterior information
    (likelihood info + prior info).

    Examples
    --------
    >>> X = np.array([[1, 1, 1], [0, 0, 0]])  # Extreme scores
    >>> mask = np.ones_like(X, dtype=bool)
    >>> a = np.ones(3)
    >>> b = np.zeros(3)
    >>> theta, se = score_map_newton(X, mask, a, b)
    >>> # MAP will be finite due to prior, unlike MLE
    """
    N = X.shape[0]
    theta_lo, theta_hi = bounds

    # Initialize
    if theta0 is None:
        theta = np.full(N, prior_mean, dtype=np.float64)
    else:
        theta = theta0.copy()

    theta_hat = np.zeros(N, dtype=np.float64)
    se = np.zeros(N, dtype=np.float64)

    prior_var = prior_sd**2

    for i in range(N):
        obs = mask_obs[i]
        if not np.any(obs):
            # No observations - posterior equals prior
            theta_hat[i] = prior_mean
            se[i] = prior_sd
            continue

        x_obs = X[i, obs]
        a_obs = a[obs]
        b_obs = b[obs]
        c_obs = c[obs] if c is not None else None

        # Newton-Raphson
        th = theta[i]
        for _ in range(max_iter):
            z = a_obs * (th - b_obs)
            p_2pl = expit(z)
            if c_obs is None:
                p = p_2pl
                dp_dtheta = a_obs * p * (1 - p)
            else:
                p = c_obs + (1.0 - c_obs) * p_2pl
                dp_dtheta = (1.0 - c_obs) * a_obs * p_2pl * (1 - p_2pl)
            p = np.clip(p, 1e-12, 1 - 1e-12)

            denom = p * (1 - p)
            # Likelihood gradient
            grad_lik = np.sum((x_obs - p) * dp_dtheta / denom)

            # Prior gradient
            grad_prior = -(th - prior_mean) / prior_var

            # Total gradient
            grad = grad_lik + grad_prior

            # Likelihood Hessian (Fisher information approximation)
            hess_lik = -np.sum((dp_dtheta**2) / denom)

            # Prior Hessian
            hess_prior = -1.0 / prior_var

            # Total Hessian
            hess = hess_lik + hess_prior

            if abs(grad) < tol:
                break

            if abs(hess) < 1e-12:
                step = 0.1 * np.sign(grad)
            else:
                step = -grad / hess

            step = np.clip(step, -2.0, 2.0)
            th = np.clip(th + step, theta_lo, theta_hi)

        theta_hat[i] = th

        # Compute SE from posterior information
        z = a_obs * (th - b_obs)
        p_2pl = expit(z)
        if c_obs is None:
            p = p_2pl
            dp_dtheta = a_obs * p * (1 - p)
        else:
            p = c_obs + (1.0 - c_obs) * p_2pl
            dp_dtheta = (1.0 - c_obs) * a_obs * p_2pl * (1 - p_2pl)
        p = np.clip(p, 1e-12, 1 - 1e-12)
        info_lik = np.sum((dp_dtheta**2) / (p * (1 - p)))
        info_prior = 1.0 / prior_var
        info_total = info_lik + info_prior
        se[i] = 1.0 / np.sqrt(max(info_total, 1e-12))

    return theta_hat, se


def score_eap_numerical(
    X_row: np.ndarray,
    mask_row: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None,
    theta_grid: np.ndarray,
    log_prior: np.ndarray,
) -> tuple[float, float]:
    """
    Compute EAP for a single person using numerical integration.

    Parameters
    ----------
    X_row : np.ndarray
        Responses for one person, shape (J,).
    mask_row : np.ndarray
        Observation mask, shape (J,).
    a : np.ndarray
        Discrimination parameters, shape (J,).
    b : np.ndarray
        Difficulty parameters, shape (J,).
    c : np.ndarray or None
        Guessing parameters for 3PL, shape (J,).
    theta_grid : np.ndarray
        Quadrature points, shape (K,).
    log_prior : np.ndarray
        Log-prior values, shape (K,).

    Returns
    -------
    eap : float
        EAP estimate.
    se : float
        Posterior standard deviation.

    Notes
    -----
    This is useful for scoring new data without recomputing the full
    posterior weight matrix.
    """
    from scipy.special import logsumexp

    K = len(theta_grid)
    obs = mask_row
    if not np.any(obs):
        # Return prior mean and SD
        prior_weights = np.exp(log_prior - logsumexp(log_prior))
        eap = np.sum(prior_weights * theta_grid)
        se = np.sqrt(np.sum(prior_weights * (theta_grid - eap)**2))
        return eap, se

    x_obs = X_row[obs]
    a_obs = a[obs]
    b_obs = b[obs]
    c_obs = c[obs] if c is not None else None

    # Compute log-likelihood at each theta
    logL_k = np.zeros(K, dtype=np.float64)
    for k in range(K):
        z = a_obs * (theta_grid[k] - b_obs)
        p_2pl = expit(z)
        if c_obs is None:
            p = p_2pl
        else:
            p = c_obs + (1.0 - c_obs) * p_2pl
        p = np.clip(p, 1e-12, 1 - 1e-12)
        logL_k[k] = np.sum(x_obs * np.log(p) + (1 - x_obs) * np.log(1 - p))

    # Log-posterior (unnormalized)
    log_post = logL_k + log_prior

    # Normalize
    log_norm = logsumexp(log_post)
    w_k = np.exp(log_post - log_norm)

    # EAP
    eap = np.sum(w_k * theta_grid)

    # SE
    se = np.sqrt(np.sum(w_k * (theta_grid - eap)**2))

    return float(eap), float(se)


def expected_score(
    theta: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """
    Compute expected raw score at a given theta.

    E[X | theta] = sum_j P_j(theta)

    Parameters
    ----------
    theta : float
        Ability value.
    a : np.ndarray
        Discrimination parameters, shape (J,).
    b : np.ndarray
        Difficulty parameters, shape (J,).
    c : np.ndarray or None
        Guessing parameters for 3PL, shape (J,).
    mask : np.ndarray or None, optional
        Boolean mask indicating which items to include.

    Returns
    -------
    float
        Expected raw score.
    """
    if mask is not None:
        a = a[mask]
        b = b[mask]
        if c is not None:
            c = c[mask]

    z = a * (theta - b)
    p_2pl = expit(z)
    if c is None:
        p = p_2pl
    else:
        p = c + (1.0 - c) * p_2pl
    return float(p.sum())


def invert_expected_score(
    target_score: float,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    bounds: tuple[float, float] = (-6.0, 6.0),
    tol: float = 1e-6,
) -> float:
    """
    Find theta that gives a target expected score.

    Useful for converting raw scores to ability estimates on the
    expected score scale.

    Parameters
    ----------
    target_score : float
        Target expected raw score.
    a : np.ndarray
        Discrimination parameters, shape (J,).
    b : np.ndarray
        Difficulty parameters, shape (J,).
    c : np.ndarray or None
        Guessing parameters for 3PL, shape (J,).
    mask : np.ndarray or None, optional
        Items to include.
    bounds : tuple[float, float], default=(-6.0, 6.0)
        Search bounds.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    float
        Theta value giving the target expected score.
    """
    from scipy.optimize import brentq

    if mask is not None:
        a = a[mask]
        b = b[mask]
        if c is not None:
            c = c[mask]

    # Expected score function minus target
    def f(theta):
        z = a * (theta - b)
        p_2pl = expit(z)
        if c is None:
            p = p_2pl
        else:
            p = c + (1.0 - c) * p_2pl
        return p.sum() - target_score

    # Check bounds
    f_lo = f(bounds[0])
    f_hi = f(bounds[1])

    if f_lo > 0:
        return bounds[0]
    if f_hi < 0:
        return bounds[1]

    return float(brentq(f, bounds[0], bounds[1], xtol=tol))
