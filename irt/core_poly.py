"""
Core probability functions for polytomous IRT models.

This module provides numerically stable implementations of:
- PCM (Partial Credit Model)
- RSM (Rating Scale Model)
- GRM (Graded Response Model)
- GPCM (Generalized Partial Credit Model)
- NRM (Nominal Response Model)

All functions use 0-based categories: 0, 1, ..., m_j - 1.
All probability functions return shape (len(theta), n_categories).
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from .core import clip_prob


def _softmax_2d(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over axis=1. Input/output: (N, C)."""
    log_denom = logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits - log_denom)


# =============================================================================
# PCM - Partial Credit Model (Masters 1982)
# P(X=x|θ) = exp(Σ_{k=1}^x (θ - b_k)) / Σ_{h=0}^{m} exp(Σ_{k=1}^h (θ - b_k))
# b: step parameters, shape (K,) where K = n_categories - 1
# =============================================================================


def prob_pcm(
    theta: np.ndarray,
    b: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """
    Compute PCM category probabilities for one item.

    Parameters
    ----------
    theta : np.ndarray
        Ability values, shape (K,) or scalar.
    b : np.ndarray
        Step parameters, shape (n_categories - 1,).
    n_categories : int
        Number of categories (0, 1, ..., n_categories-1).

    Returns
    -------
    np.ndarray
        P(X=x|θ) for x=0..n_categories-1, shape (len(theta), n_categories).
    """
    theta = np.atleast_1d(theta).astype(np.float64).ravel()
    N = len(theta)
    K = n_categories - 1
    b = np.asarray(b, dtype=np.float64).ravel()[:K]

    # s_h = Σ_{k=1}^h (θ - b_k) = h*θ - Σ_{k=1}^h b_k
    # Use log-space via softmax for numerical stability
    b_cumsum = np.concatenate([[0.0], np.cumsum(b)])  # length n_categories
    logits = np.zeros((N, n_categories), dtype=np.float64)
    for h in range(1, n_categories):
        if h <= len(b):
            logits[:, h] = h * theta - b_cumsum[h]
    return _softmax_2d(logits)


def log_prob_pcm(
    theta: np.ndarray,
    b: np.ndarray,
    n_categories: int,
    category: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Log P(X=category|θ) for PCM."""
    p = prob_pcm(theta, b, n_categories)
    p_cat = p[:, category] if p.ndim > 1 else p[category]
    return np.log(np.clip(p_cat, eps, 1.0 - eps))


# =============================================================================
# RSM - Rating Scale Model (Andrich 1978)
# b_jk = b_j + tau_k, common tau across items
# =============================================================================


def prob_rsm(
    theta: np.ndarray,
    b: float,
    tau: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """
    Compute RSM category probabilities for one item.

    The RSM is a constrained PCM where step parameters are decomposed as
    b_jk = b_j + tau_k, with tau common across all items.

    Parameters
    ----------
    theta : np.ndarray
        Ability values.
    b : float
        Item location parameter.
    tau : np.ndarray
        Common step parameters, shape (n_categories - 1,).
    n_categories : int
        Number of categories.

    Returns
    -------
    np.ndarray
        Category probabilities, shape (len(theta), n_categories).
    """
    K = n_categories - 1
    tau = np.asarray(tau, dtype=np.float64)[:K]
    # b_jk = b + tau_k
    b_full = b + tau
    return prob_pcm(theta, b_full, n_categories)


def log_prob_rsm(
    theta: np.ndarray,
    b: float,
    tau: np.ndarray,
    n_categories: int,
    category: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Log P(X=category|θ) for RSM."""
    p = prob_rsm(theta, b, tau, n_categories)
    p_cat = p[:, category] if p.ndim > 1 else p[category]
    return np.log(np.clip(p_cat, eps, 1.0 - eps))


# =============================================================================
# GRM - Graded Response Model (Samejima 1969)
# P*(k) = P(X≥k|θ) = expit(a(θ - b_k))
# P(X=x|θ) = P*(x) - P*(x+1)
# =============================================================================


def prob_grm(
    theta: np.ndarray,
    a: float,
    b: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """
    Compute GRM category probabilities for one item.

    Unlike the other models in this module, the GRM defines probabilities via
    cumulative boundary curves: P*(k) = expit(a(θ − b_k)), with
    P(X=x) = P*(x) − P*(x+1).

    Parameters
    ----------
    theta : np.ndarray
        Ability values.
    a : float
        Discrimination parameter.
    b : np.ndarray
        Category boundary parameters, shape (n_categories - 1,).
        Must be in ascending order. b[0] is the boundary between
        categories 0 and 1, etc.
    n_categories : int
        Number of categories.

    Returns
    -------
    np.ndarray
        Category probabilities, shape (len(theta), n_categories).
    """
    theta = np.atleast_1d(theta).astype(np.float64).ravel()
    N = len(theta)
    K = n_categories - 1
    b = np.asarray(b, dtype=np.float64).ravel()[:K]

    # P*(k) = P(X >= k) = expit(a * (theta - b_{k-1}))
    # P*(0) = 1, P*(m+1) = 0
    # P(X=x) = P*(x) - P*(x+1)
    p_star = np.ones((N, n_categories + 1), dtype=np.float64)
    for k in range(1, n_categories):
        if k - 1 < len(b):
            p_star[:, k] = expit(a * (theta - b[k - 1]))
    p_star[:, n_categories] = 0.0

    p = p_star[:, :-1] - p_star[:, 1:]
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def log_prob_grm(
    theta: np.ndarray,
    a: float,
    b: np.ndarray,
    n_categories: int,
    category: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Log P(X=category|θ) for GRM."""
    p = prob_grm(theta, a, b, n_categories)
    p_cat = p[:, category] if p.ndim > 1 else p[category]
    return np.log(np.clip(p_cat, eps, 1.0 - eps))


# =============================================================================
# GPCM - Generalized Partial Credit Model (Muraki 1992)
# P(X=x|θ) = exp(Σ_{k=1}^x a(θ - b_k)) / Σ_h exp(...)
# =============================================================================


def prob_gpcm(
    theta: np.ndarray,
    a: float,
    b: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """
    Compute GPCM category probabilities for one item.

    Parameters
    ----------
    theta : np.ndarray
        Ability values.
    a : float
        Discrimination parameter.
    b : np.ndarray
        Step parameters, shape (n_categories - 1,).
    n_categories : int
        Number of categories.
    """
    theta = np.atleast_1d(theta).astype(np.float64).ravel()
    N = len(theta)
    K = n_categories - 1
    b = np.asarray(b, dtype=np.float64).ravel()[:K]

    # logits[h] = Σ_{k=1}^h a(θ - b_k) = a * (h*θ - Σ b_k)
    b_cumsum = np.concatenate([[0.0], np.cumsum(b)])
    logits = np.zeros((N, n_categories), dtype=np.float64)
    for h in range(1, n_categories):
        if h <= len(b):
            logits[:, h] = a * (h * theta - b_cumsum[h])
    return _softmax_2d(logits)


def log_prob_gpcm(
    theta: np.ndarray,
    a: float,
    b: np.ndarray,
    n_categories: int,
    category: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Log P(X=category|θ) for GPCM."""
    p = prob_gpcm(theta, a, b, n_categories)
    p_cat = p[:, category] if p.ndim > 1 else p[category]
    return np.log(np.clip(p_cat, eps, 1.0 - eps))


# =============================================================================
# NRM - Nominal Response Model (Bock 1972)
# P(X=x|θ) = exp(a_x*θ + c_x) / Σ_h exp(a_h*θ + c_h)
# Fix a_0=0, c_0=0 for identification
# =============================================================================


def prob_nrm(
    theta: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    n_categories: int,
) -> np.ndarray:
    """
    Compute NRM category probabilities for one item.

    Parameters
    ----------
    theta : np.ndarray
        Ability values.
    a : np.ndarray
        Slope parameters per category, shape (n_categories,). a[0]=0 for identification.
    c : np.ndarray
        Intercept parameters per category, shape (n_categories,). c[0]=0 for identification.
    n_categories : int
        Number of categories.
    """
    theta = np.atleast_1d(theta).astype(np.float64).ravel()
    a = np.asarray(a, dtype=np.float64).ravel()[:n_categories].copy()
    c = np.asarray(c, dtype=np.float64).ravel()[:n_categories].copy()
    a[0] = 0.0
    c[0] = 0.0

    # logits[n, x] = a_x * theta_n + c_x
    logits = theta[:, np.newaxis] * a[np.newaxis, :] + c[np.newaxis, :]
    return _softmax_2d(logits)


def log_prob_nrm(
    theta: np.ndarray,
    a: np.ndarray,
    c: np.ndarray,
    n_categories: int,
    category: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """Log P(X=category|θ) for NRM."""
    p = prob_nrm(theta, a, c, n_categories)
    p_cat = p[:, category] if p.ndim > 1 else p[category]
    return np.log(np.clip(p_cat, eps, 1.0 - eps))


# =============================================================================
# Expected score and information (polytomous)
# =============================================================================


def expected_score_poly(
    p: np.ndarray,
    categories: np.ndarray | None = None,
) -> np.ndarray:
    """
    Expected score E[X|θ] = Σ_x x * P(X=x|θ).

    Parameters
    ----------
    p : np.ndarray
        Category probabilities, shape (..., n_categories).
    categories : np.ndarray or None
        Category values (default 0, 1, ..., n_categories-1).
    """
    n_cat = p.shape[-1]
    if categories is None:
        categories = np.arange(n_cat, dtype=np.float64)
    return np.sum(p * categories, axis=-1)


def info_poly(
    p: np.ndarray,
    dp_dtheta: np.ndarray,
    categories: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fisher information for polytomous item.
    I(θ) = Σ_x (dP/dθ)^2 / P
    """
    n_cat = p.shape[-1]
    if categories is None:
        categories = np.arange(n_cat, dtype=np.float64)
    p_safe = clip_prob(p)
    return np.sum(dp_dtheta**2 / p_safe, axis=-1)
