"""
M-step solvers for polytomous IRT models (MML-EM).

Each function maximizes the expected complete-data log-likelihood
Q = sum_k sum_c R_k_c * log P(X=c | theta_k, params)

for a single item, given sufficient statistics N_k and R_k_c.

All optimizations use L-BFGS-B with numerical gradients (approx_grad=True)
which lets scipy compute finite-difference gradients internally, avoiding
redundant evaluations.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .core_poly import prob_pcm, prob_gpcm, prob_grm, prob_rsm, prob_nrm


def _neg_Q_value(
    prob_fn,
    theta: np.ndarray,
    R_k_c: np.ndarray,
    *prob_args,
) -> float:
    """Compute negative expected log-likelihood for any polytomous model."""
    n_cat = R_k_c.shape[1]
    p = prob_fn(theta, *prob_args, n_cat)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -float(np.sum(R_k_c * np.log(p)))


def _pad_to(arr: np.ndarray, length: int, fill: float = 0.0) -> np.ndarray:
    """Pad 1-D array to `length` if shorter."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), constant_values=fill)
    return arr[:length]


def update_item_pcm(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k_c: np.ndarray,
    b0: np.ndarray,
    b_bounds: tuple[float, float],
    max_iter: int = 100,
) -> np.ndarray:
    """Update PCM step parameters for one item."""
    n_cat = R_k_c.shape[1]
    K_step = n_cat - 1
    b0 = _pad_to(b0, K_step)

    result = minimize(
        lambda b: _neg_Q_value(prob_pcm, theta, R_k_c, b),
        b0,
        method="L-BFGS-B",
        bounds=[b_bounds] * K_step,
        options={"maxiter": max_iter, "ftol": 1e-8},
    )
    return result.x


def update_item_gpcm(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k_c: np.ndarray,
    a0: float,
    b0: np.ndarray,
    a_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
    max_iter: int = 100,
) -> tuple[float, np.ndarray]:
    """Update GPCM discrimination and step parameters for one item."""
    n_cat = R_k_c.shape[1]
    K_step = n_cat - 1
    b0 = _pad_to(b0, K_step)

    x0 = np.concatenate([[a0], b0])
    bounds = [a_bounds] + [b_bounds] * K_step

    def neg_Q(params):
        a, b = params[0], params[1 : 1 + K_step]
        return _neg_Q_value(prob_gpcm, theta, R_k_c, a, b)

    result = minimize(neg_Q, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iter, "ftol": 1e-8})
    return float(result.x[0]), result.x[1 : 1 + K_step]


def update_item_grm(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k_c: np.ndarray,
    a0: float,
    b0: np.ndarray,
    a_bounds: tuple[float, float],
    b_bounds: tuple[float, float],
    max_iter: int = 100,
) -> tuple[float, np.ndarray]:
    """Update GRM parameters for one item. Boundary params are sorted at end."""
    n_cat = R_k_c.shape[1]
    K_step = n_cat - 1
    b0 = np.sort(_pad_to(b0, K_step))

    x0 = np.concatenate([[a0], b0])
    bounds = [a_bounds] + [b_bounds] * K_step

    def neg_Q(params):
        a, b = params[0], params[1 : 1 + K_step]
        return _neg_Q_value(prob_grm, theta, R_k_c, a, b)

    result = minimize(neg_Q, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": max_iter, "ftol": 1e-8})
    a_hat = float(result.x[0])
    b_hat = np.sort(result.x[1 : 1 + K_step])  # GRM requires ordered boundaries
    return a_hat, b_hat


def update_item_rsm(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k_c: np.ndarray,
    b0: float,
    tau: np.ndarray,
    b_bounds: tuple[float, float],
    max_iter: int = 100,
) -> float:
    """Update RSM item location b_j for one item, with tau fixed."""
    n_cat = R_k_c.shape[1]
    tau = np.asarray(tau, dtype=np.float64)[: n_cat - 1]

    def neg_Q(b_val):
        return _neg_Q_value(prob_rsm, theta, R_k_c, b_val, tau)

    result = minimize(
        neg_Q,
        np.atleast_1d(b0),
        method="L-BFGS-B",
        bounds=[b_bounds],
        options={"maxiter": max_iter, "ftol": 1e-8},
    )
    return float(result.x[0])


def update_tau_rsm(
    theta: np.ndarray,
    R_all: list[np.ndarray],
    b_all: np.ndarray,
    tau0: np.ndarray,
    tau_bounds: tuple[float, float],
    max_iter: int = 100,
) -> np.ndarray:
    """Update RSM common tau given all items' b and sufficient statistics."""
    n_items = len(R_all)
    n_cat = R_all[0].shape[1]
    K_step = n_cat - 1
    tau0 = _pad_to(tau0, K_step)

    def neg_Q(tau_params):
        total = 0.0
        for j in range(n_items):
            p = prob_rsm(theta, b_all[j], tau_params, n_cat)
            p = np.clip(p, 1e-12, 1.0 - 1e-12)
            total += np.sum(R_all[j] * np.log(p))
        return -total

    result = minimize(
        neg_Q,
        tau0,
        method="L-BFGS-B",
        bounds=[tau_bounds] * K_step,
        options={"maxiter": max_iter, "ftol": 1e-8},
    )
    return result.x


def update_item_nrm(
    theta: np.ndarray,
    N_k: np.ndarray,
    R_k_c: np.ndarray,
    a0: np.ndarray,
    c0: np.ndarray,
    n_categories: int,
    bounds: tuple[float, float] = (-4.0, 4.0),
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Update NRM slope/intercept parameters for one item. a[0]=c[0]=0 fixed."""
    n_cat = n_categories
    K_free = n_cat - 1
    a0 = np.asarray(a0, dtype=np.float64)[:n_cat].copy()
    c0 = np.asarray(c0, dtype=np.float64)[:n_cat].copy()
    a0[0], c0[0] = 0.0, 0.0

    x0 = np.concatenate([a0[1:], c0[1:]])
    bds = [bounds] * (2 * K_free)

    def neg_Q(params):
        a = np.zeros(n_cat)
        c = np.zeros(n_cat)
        a[1:] = params[:K_free]
        c[1:] = params[K_free : 2 * K_free]
        return _neg_Q_value(prob_nrm, theta, R_k_c, a, c)

    result = minimize(neg_Q, x0, method="L-BFGS-B", bounds=bds,
                      options={"maxiter": max_iter, "ftol": 1e-8})

    a_hat = np.zeros(n_cat)
    c_hat = np.zeros(n_cat)
    a_hat[1:] = result.x[:K_free]
    c_hat[1:] = result.x[K_free:]
    return a_hat, c_hat
