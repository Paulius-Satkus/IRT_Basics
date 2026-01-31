"""
Quadrature and prior functions for MML-EM estimation.

This module provides:
- Theta grids for numerical integration
- Prior distribution functions
- Gauss-Hermite quadrature (optional)

The MML-EM algorithm integrates over the latent ability distribution
using numerical quadrature. This module provides the infrastructure
for that integration.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import roots_hermite


def make_theta_grid(
    n_points: int = 61,
    lo: float = -4.0,
    hi: float = 4.0,
) -> np.ndarray:
    """
    Create an equally-spaced grid of theta (ability) values.

    Parameters
    ----------
    n_points : int, default=61
        Number of quadrature points.
    lo : float, default=-4.0
        Lower bound of the theta grid.
    hi : float, default=4.0
        Upper bound of the theta grid.

    Returns
    -------
    np.ndarray
        Array of theta values, shape (n_points,).

    Raises
    ------
    ValueError
        If n_points < 3 or lo >= hi.

    Notes
    -----
    A finer grid (more points) gives more accurate integration but
    increases computation time. 61 points is a good balance for most
    applications.

    Examples
    --------
    >>> theta = make_theta_grid(n_points=5, lo=-2, hi=2)
    >>> theta
    array([-2., -1.,  0.,  1.,  2.])
    """
    if n_points < 3:
        raise ValueError(f"n_points must be at least 3, got {n_points}")
    if lo >= hi:
        raise ValueError(f"lo must be less than hi, got lo={lo}, hi={hi}")

    return np.linspace(lo, hi, n_points)


def prior_logpdf(
    theta: np.ndarray,
    kind: str = "normal",
    mean: float = 0.0,
    sd: float = 1.0,
) -> np.ndarray:
    """
    Compute log-prior density at given theta values.

    Parameters
    ----------
    theta : np.ndarray
        Theta values at which to evaluate the prior, shape (K,).
    kind : {"normal", "uniform"}, default="normal"
        Type of prior distribution:
        - "normal": Normal (Gaussian) distribution
        - "uniform": Uniform distribution over [mean-3*sd, mean+3*sd]
    mean : float, default=0.0
        Mean of the prior distribution.
    sd : float, default=1.0
        Standard deviation of the prior (or half-width for uniform).

    Returns
    -------
    np.ndarray
        Log-density values, shape (K,).

    Raises
    ------
    ValueError
        If kind is not recognized or sd <= 0.

    Notes
    -----
    For MML-EM, the prior provides identification of the ability scale.
    A standard normal prior N(0,1) is most common, which anchors the
    ability scale to have mean 0 and variance 1 in the population.

    Examples
    --------
    >>> theta = np.array([-2, 0, 2])
    >>> prior_logpdf(theta, kind="normal", mean=0, sd=1)
    array([-2.91893853, -0.91893853, -2.91893853])
    """
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")

    kind = kind.lower()

    if kind == "normal":
        return stats.norm.logpdf(theta, loc=mean, scale=sd)
    elif kind == "uniform":
        # Uniform over [mean - 3*sd, mean + 3*sd]
        lo = mean - 3 * sd
        hi = mean + 3 * sd
        return stats.uniform.logpdf(theta, loc=lo, scale=hi - lo)
    else:
        raise ValueError(
            f"Unknown prior kind '{kind}'. Must be 'normal' or 'uniform'."
        )


def gauss_hermite_grid(
    n_points: int = 21,
    mean: float = 0.0,
    sd: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Gauss-Hermite quadrature nodes and weights for normal distribution.

    Parameters
    ----------
    n_points : int, default=21
        Number of quadrature points.
    mean : float, default=0.0
        Mean of the normal distribution.
    sd : float, default=1.0
        Standard deviation of the normal distribution.

    Returns
    -------
    nodes : np.ndarray
        Quadrature nodes (theta values), shape (n_points,).
    weights : np.ndarray
        Quadrature weights, shape (n_points,).

    Notes
    -----
    Gauss-Hermite quadrature is optimal for integrating functions against
    the standard normal distribution:

        integral f(x) * exp(-x^2) dx ≈ sum_k w_k * f(x_k)

    We transform to integrate against N(mean, sd^2):

        integral f(x) * phi(x; mean, sd) dx ≈ sum_k w_k * f(nodes_k)

    where phi is the normal pdf.

    This is more efficient than rectangular quadrature for smooth
    functions, requiring fewer points for the same accuracy.

    Examples
    --------
    >>> nodes, weights = gauss_hermite_grid(n_points=5)
    >>> # Weights should sum to 1 (approximately, for transformed version)
    >>> weights.sum()  # doctest: +SKIP
    1.0
    """
    if n_points < 1:
        raise ValueError(f"n_points must be at least 1, got {n_points}")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")

    # Get standard Hermite quadrature nodes and weights
    # These are for the weight function exp(-x^2)
    nodes_std, weights_std = roots_hermite(n_points)

    # Transform nodes: x_std -> x = mean + sd * sqrt(2) * x_std
    # Transform weights: account for the change of variables
    # The standard Gauss-Hermite weights integrate against exp(-x^2)
    # We need to integrate against (1/sqrt(2*pi)) * exp(-x^2/2)
    nodes = mean + sd * np.sqrt(2) * nodes_std

    # Adjust weights: divide by sqrt(pi) to normalize
    # (since Hermite weights are for exp(-x^2), not for standard normal)
    weights = weights_std / np.sqrt(np.pi)

    return nodes, weights


def compute_integration_weights(
    theta: np.ndarray,
    prior_kind: str = "normal",
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
) -> np.ndarray:
    """
    Compute integration weights for rectangular quadrature with prior.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature points, shape (K,).
    prior_kind : str, default="normal"
        Type of prior distribution.
    prior_mean : float, default=0.0
        Mean of the prior.
    prior_sd : float, default=1.0
        Standard deviation of the prior.

    Returns
    -------
    np.ndarray
        Integration weights, shape (K,). These are the prior densities
        times the grid spacing, normalized to sum to 1.

    Notes
    -----
    For rectangular (midpoint) quadrature:
        integral f(x) * p(x) dx ≈ sum_k f(x_k) * p(x_k) * dx

    We compute w_k = p(x_k) * dx and normalize so sum(w_k) = 1.

    Examples
    --------
    >>> theta = make_theta_grid(n_points=5)
    >>> weights = compute_integration_weights(theta)
    >>> weights.sum()  # Should be close to 1
    1.0
    """
    if len(theta) < 2:
        return np.ones(1)

    # Grid spacing (assumed uniform)
    dx = theta[1] - theta[0]

    # Prior density at each point
    log_prior = prior_logpdf(theta, kind=prior_kind, mean=prior_mean, sd=prior_sd)
    prior_density = np.exp(log_prior)

    # Weights = density * spacing
    weights = prior_density * dx

    # Normalize to sum to 1
    weights = weights / weights.sum()

    return weights


def validate_theta_grid(theta: np.ndarray) -> None:
    """
    Validate a theta grid for use in MML-EM.

    Parameters
    ----------
    theta : np.ndarray
        Quadrature points to validate.

    Raises
    ------
    ValueError
        If theta is invalid (wrong shape, not sorted, etc.).
    """
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-dimensional, got {theta.ndim} dimensions")

    if len(theta) < 3:
        raise ValueError(f"theta must have at least 3 points, got {len(theta)}")

    if not np.all(np.diff(theta) > 0):
        raise ValueError("theta must be strictly increasing")

    if not np.all(np.isfinite(theta)):
        raise ValueError("theta contains non-finite values")
