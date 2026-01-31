"""
Core mathematical utilities for IRT computations.

This module provides numerically stable implementations of:
- Probability functions (2PL/Rasch ICC)
- Log-likelihood computations
- Parameter transformations

All functions are designed to be vectorized and handle edge cases
(extreme values, missing data) gracefully.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


def clip_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Clip probabilities to avoid numerical issues with log(0) or log(1).

    Parameters
    ----------
    p : np.ndarray
        Array of probabilities.
    eps : float, default=1e-12
        Small constant for clipping. Probabilities will be clipped to [eps, 1-eps].

    Returns
    -------
    np.ndarray
        Clipped probabilities.

    Examples
    --------
    >>> p = np.array([0.0, 0.5, 1.0])
    >>> clip_prob(p)
    array([1.e-12, 5.e-01, 1.e+00])
    """
    return np.clip(p, eps, 1.0 - eps)


def bernoulli_loglik_from_p(
    p: np.ndarray,
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute Bernoulli log-likelihood given probabilities.

    Computes: x * log(p) + (1 - x) * log(1 - p)

    This implementation is numerically stable, clipping probabilities
    to avoid log(0).

    Parameters
    ----------
    p : np.ndarray
        Probabilities P(X=1), any shape.
    x : np.ndarray
        Observed responses (0 or 1), same shape as p.
    eps : float, default=1e-12
        Small constant for probability clipping.

    Returns
    -------
    np.ndarray
        Log-likelihood values, same shape as p.

    Notes
    -----
    For numerical stability, probabilities are clipped to [eps, 1-eps]
    before taking logarithms.

    Examples
    --------
    >>> p = np.array([0.8, 0.2, 0.5])
    >>> x = np.array([1, 0, 1])
    >>> bernoulli_loglik_from_p(p, x)
    array([-0.22314355, -0.22314355, -0.69314718])
    """
    p_safe = clip_prob(p, eps)
    return x * np.log(p_safe) + (1 - x) * np.log(1 - p_safe)


def prob_2pl(
    theta: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """
    Compute 2PL item response probability.

    P(X=1|theta) = 1 / (1 + exp(-a * (theta - b)))

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a : float
        Item discrimination parameter (a > 0).
    b : float
        Item difficulty parameter.

    Returns
    -------
    np.ndarray
        Probabilities of correct response, shape (K,).

    Notes
    -----
    Uses scipy.special.expit for numerical stability, which computes
    1 / (1 + exp(-x)) accurately for both large positive and negative x.

    Examples
    --------
    >>> theta = np.array([-2, 0, 2])
    >>> prob_2pl(theta, a=1.0, b=0.0)
    array([0.11920292, 0.5       , 0.88079708])
    """
    z = a * (theta - b)
    return expit(z)


def prob_3pl(
    theta: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """
    Compute 3PL item response probability.

    P(X=1|theta) = c + (1 - c) * 1 / (1 + exp(-a * (theta - b)))
    """
    p_2pl = prob_2pl(theta, a, b)
    return c + (1.0 - c) * p_2pl


def prob_matrix_2pl(
    theta: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute probability matrix for all theta values and items.

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a_vec : np.ndarray
        Item discrimination parameters, shape (J,).
    b_vec : np.ndarray
        Item difficulty parameters, shape (J,).

    Returns
    -------
    np.ndarray
        Probability matrix of shape (K, J) where entry (k, j) is
        P(X=1|theta_k, a_j, b_j).

    Examples
    --------
    >>> theta = np.array([-1, 0, 1])
    >>> a_vec = np.array([1.0, 1.5])
    >>> b_vec = np.array([0.0, 0.5])
    >>> P = prob_matrix_2pl(theta, a_vec, b_vec)
    >>> P.shape
    (3, 2)
    """
    # theta: (K,), a_vec: (J,), b_vec: (J,)
    # Result: (K, J)
    # z[k, j] = a_j * (theta_k - b_j)
    z = a_vec[np.newaxis, :] * (theta[:, np.newaxis] - b_vec[np.newaxis, :])
    return expit(z)


def prob_matrix_3pl(
    theta: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    c_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute probability matrix for 3PL model.
    """
    p_2pl = prob_matrix_2pl(theta, a_vec, b_vec)
    return c_vec[np.newaxis, :] + (1.0 - c_vec[np.newaxis, :]) * p_2pl


def log_prob_2pl(
    theta: np.ndarray,
    a: float,
    b: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log(p) and log(1-p) for 2PL model, numerically stable.

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a : float
        Item discrimination parameter.
    b : float
        Item difficulty parameter.
    eps : float, default=1e-12
        Small constant for numerical stability.

    Returns
    -------
    log_p : np.ndarray
        log P(X=1|theta), shape (K,).
    log_q : np.ndarray
        log P(X=0|theta) = log(1-p), shape (K,).

    Notes
    -----
    For numerical stability, we use:
    - log(p) = -log(1 + exp(-z)) = -softplus(-z)
    - log(1-p) = -log(1 + exp(z)) = -softplus(z)

    where z = a * (theta - b).
    """
    z = a * (theta - b)
    # log(p) = -log(1 + exp(-z))
    # log(1-p) = -log(1 + exp(z))
    # Use stable computation via log1p for moderate values
    log_p = np.where(
        z >= 0,
        -np.log1p(np.exp(-z)),
        z - np.log1p(np.exp(z)),
    )
    log_q = np.where(
        z >= 0,
        -z - np.log1p(np.exp(-z)),
        -np.log1p(np.exp(z)),
    )
    return log_p, log_q


def log_prob_3pl(
    theta: np.ndarray,
    a: float,
    b: float,
    c: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log(p) and log(1-p) for 3PL model, numerically stable.
    """
    p = prob_3pl(theta, a, b, c)
    p_safe = clip_prob(p, eps)
    log_p = np.log(p_safe)
    log_q = np.log(1.0 - p_safe)
    return log_p, log_q


def center_difficulties(
    b: np.ndarray,
    est_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Center item difficulties to have mean zero.

    Parameters
    ----------
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    est_mask : np.ndarray or None, optional
        Boolean array of shape (J,) indicating which items are being estimated
        (True = estimated, False = fixed). If None, all items are considered.
        Only estimated items contribute to the mean.

    Returns
    -------
    np.ndarray
        Centered difficulty parameters, shape (J,).

    Notes
    -----
    This is used for identification in JMLE, where the mean of estimated
    difficulties is constrained to zero.

    Examples
    --------
    >>> b = np.array([1.0, 2.0, 3.0])
    >>> center_difficulties(b)
    array([-1.,  0.,  1.])
    >>> # With fixed items
    >>> mask = np.array([True, True, False])  # Third item fixed
    >>> center_difficulties(b, mask)
    array([-0.5,  0.5,  3. ])
    """
    if est_mask is None:
        est_mask = np.ones(len(b), dtype=bool)

    if not np.any(est_mask):
        # No items to center
        return b.copy()

    mean_b = np.mean(b[est_mask])
    b_centered = b.copy()
    b_centered[est_mask] = b[est_mask] - mean_b
    return b_centered


def logit_transform(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute logit transformation with safeguards for extreme values.

    Parameters
    ----------
    p : np.ndarray
        Probabilities to transform.
    eps : float, default=1e-6
        Clipping value for extreme probabilities.

    Returns
    -------
    np.ndarray
        Logit values: log(p / (1-p)).

    Examples
    --------
    >>> p = np.array([0.1, 0.5, 0.9])
    >>> logit_transform(p)
    array([-2.19722458,  0.        ,  2.19722458])
    """
    p_safe = np.clip(p, eps, 1 - eps)
    return logit(p_safe)


def info_2pl(
    theta: np.ndarray,
    a: float,
    b: float,
) -> np.ndarray:
    """
    Compute Fisher information for 2PL model at given theta values.

    I(theta) = a^2 * p * (1 - p)

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a : float
        Item discrimination parameter.
    b : float
        Item difficulty parameter.

    Returns
    -------
    np.ndarray
        Information values, shape (K,).

    Notes
    -----
    The information function is maximized at theta = b, where it equals a^2/4.
    For Rasch (a=1), the maximum information is 0.25.

    Examples
    --------
    >>> theta = np.array([-1, 0, 1])
    >>> info_2pl(theta, a=1.0, b=0.0)
    array([0.19661193, 0.25      , 0.19661193])
    """
    p = prob_2pl(theta, a, b)
    return a**2 * p * (1 - p)


def info_3pl(
    theta: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    """
    Compute Fisher information for 3PL model at given theta values.
    """
    p_2pl = prob_2pl(theta, a, b)
    p = c + (1.0 - c) * p_2pl
    dp = (1.0 - c) * a * p_2pl * (1.0 - p_2pl)
    denom = clip_prob(p) * clip_prob(1.0 - p)
    return (dp ** 2) / denom


def info_matrix_2pl(
    theta: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute information matrix for all theta values and items.

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a_vec : np.ndarray
        Item discrimination parameters, shape (J,).
    b_vec : np.ndarray
        Item difficulty parameters, shape (J,).

    Returns
    -------
    np.ndarray
        Information matrix of shape (K, J).

    Examples
    --------
    >>> theta = np.array([-1, 0, 1])
    >>> a_vec = np.array([1.0, 1.5])
    >>> b_vec = np.array([0.0, 0.5])
    >>> I = info_matrix_2pl(theta, a_vec, b_vec)
    >>> I.shape
    (3, 2)
    """
    P = prob_matrix_2pl(theta, a_vec, b_vec)
    a_sq = a_vec[np.newaxis, :] ** 2
    return a_sq * P * (1 - P)


def info_matrix_3pl(
    theta: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    c_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute information matrix for 3PL model.
    """
    p_2pl = prob_matrix_2pl(theta, a_vec, b_vec)
    p = c_vec[np.newaxis, :] + (1.0 - c_vec[np.newaxis, :]) * p_2pl
    dp = (1.0 - c_vec[np.newaxis, :]) * a_vec[np.newaxis, :] * p_2pl * (1.0 - p_2pl)
    denom = clip_prob(p) * clip_prob(1.0 - p)
    return (dp ** 2) / denom


def test_response_function(
    theta: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    c_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute expected test score (Test Response Function).

    TRF(theta) = sum_j P_j(theta)

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a_vec : np.ndarray
        Item discrimination parameters, shape (J,).
    b_vec : np.ndarray
        Item difficulty parameters, shape (J,).

    Returns
    -------
    np.ndarray
        Expected test scores, shape (K,).

    Examples
    --------
    >>> theta = np.array([-1, 0, 1])
    >>> a_vec = np.array([1.0, 1.0])
    >>> b_vec = np.array([-0.5, 0.5])
    >>> test_response_function(theta, a_vec, b_vec)
    array([0.5       , 1.        , 1.5       ])
    """
    if c_vec is None:
        P = prob_matrix_2pl(theta, a_vec, b_vec)
    else:
        P = prob_matrix_3pl(theta, a_vec, b_vec, c_vec)
    return P.sum(axis=1)


def test_information_function(
    theta: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    c_vec: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute test information function.

    TIF(theta) = sum_j I_j(theta)

    Parameters
    ----------
    theta : np.ndarray
        Person ability parameters, shape (K,).
    a_vec : np.ndarray
        Item discrimination parameters, shape (J,).
    b_vec : np.ndarray
        Item difficulty parameters, shape (J,).

    Returns
    -------
    np.ndarray
        Test information values, shape (K,).

    Examples
    --------
    >>> theta = np.array([-1, 0, 1])
    >>> a_vec = np.array([1.0, 1.0])
    >>> b_vec = np.array([-0.5, 0.5])
    >>> test_information_function(theta, a_vec, b_vec)
    array([0.43879269, 0.46211716, 0.43879269])
    """
    if c_vec is None:
        I = info_matrix_2pl(theta, a_vec, b_vec)
    else:
        I = info_matrix_3pl(theta, a_vec, b_vec, c_vec)
    return I.sum(axis=1)
