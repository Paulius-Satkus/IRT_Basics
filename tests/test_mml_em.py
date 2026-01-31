"""
Tests for MML-EM estimator.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from irt.estimators.mml_em import (
    compute_loglik_nk,
    posterior_weights,
    sufficient_stats_item,
    sufficient_stats_all_items,
    initialize_params_mml,
)
from irt.quadrature import make_theta_grid, prior_logpdf


class TestComputeLoglikNk:
    """Tests for compute_loglik_nk function."""

    def test_shape(self):
        """Output has correct shape (N, K)."""
        N, J, K = 10, 5, 21
        X = np.random.binomial(1, 0.5, size=(N, J)).astype(float)
        mask = np.ones((N, J), dtype=bool)
        theta = make_theta_grid(n_points=K)
        a = np.ones(J)
        b = np.zeros(J)
        
        logL = compute_loglik_nk(X, mask, theta, a, b)
        assert logL.shape == (N, K)

    def test_finite_values(self):
        """Log-likelihood values are finite."""
        X = np.array([[1, 0, 1], [0, 1, 0]])
        mask = np.ones_like(X, dtype=bool)
        theta = make_theta_grid(n_points=21)
        a = np.ones(3)
        b = np.array([-1, 0, 1])
        
        logL = compute_loglik_nk(X, mask, theta, a, b)
        assert np.all(np.isfinite(logL))

    def test_handles_missing(self):
        """Missing data is handled correctly."""
        X = np.array([[1, np.nan, 1], [0, 1, np.nan]])
        mask = ~np.isnan(X)
        theta = make_theta_grid(n_points=21)
        a = np.ones(3)
        b = np.zeros(3)
        
        logL = compute_loglik_nk(X, mask, theta, a, b)
        assert np.all(np.isfinite(logL))

    def test_monotone_in_correct_direction(self):
        """Higher ability -> higher loglik for correct responses."""
        X = np.array([[1, 1, 1]])  # All correct
        mask = np.ones_like(X, dtype=bool)
        theta = make_theta_grid(n_points=21)
        a = np.ones(3)
        b = np.zeros(3)
        
        logL = compute_loglik_nk(X, mask, theta, a, b)
        # Log-likelihood should increase with theta for all-correct pattern
        assert np.all(np.diff(logL[0]) > 0)


class TestPosteriorWeights:
    """Tests for posterior_weights function."""

    def test_rows_sum_to_one(self):
        """Posterior weights sum to 1 for each person."""
        N, K = 10, 21
        logL = np.random.randn(N, K)
        log_prior = np.random.randn(K)
        
        w_nk, _ = posterior_weights(logL, log_prior)
        row_sums = w_nk.sum(axis=1)
        assert_allclose(row_sums, np.ones(N), rtol=1e-10)

    def test_non_negative(self):
        """All weights are non-negative."""
        N, K = 10, 21
        logL = np.random.randn(N, K)
        log_prior = np.random.randn(K)
        
        w_nk, _ = posterior_weights(logL, log_prior)
        assert np.all(w_nk >= 0)

    def test_no_nan(self):
        """Weights don't contain NaN."""
        N, K = 10, 21
        logL = np.random.randn(N, K) * 10  # Larger variance
        log_prior = np.random.randn(K) * 5
        
        w_nk, log_ev = posterior_weights(logL, log_prior)
        assert not np.any(np.isnan(w_nk))
        assert not np.any(np.isnan(log_ev))


class TestSufficientStats:
    """Tests for sufficient statistics functions."""

    def test_item_stats_shape(self):
        """Item sufficient stats have correct shape."""
        N, K = 10, 21
        X_col = np.random.binomial(1, 0.5, size=N).astype(float)
        mask_col = np.ones(N, dtype=bool)
        w_nk = np.ones((N, K)) / K  # Uniform weights
        
        N_k, R_k = sufficient_stats_item(X_col, mask_col, w_nk)
        assert N_k.shape == (K,)
        assert R_k.shape == (K,)

    def test_all_items_shape(self):
        """All items sufficient stats have correct shape."""
        N, J, K = 10, 5, 21
        X = np.random.binomial(1, 0.5, size=(N, J)).astype(float)
        mask = np.ones((N, J), dtype=bool)
        w_nk = np.ones((N, K)) / K
        
        N_jk, R_jk = sufficient_stats_all_items(X, mask, w_nk)
        assert N_jk.shape == (J, K)
        assert R_jk.shape == (J, K)

    def test_R_leq_N(self):
        """Correct count R_k <= total count N_k."""
        N, K = 10, 21
        X_col = np.random.binomial(1, 0.5, size=N).astype(float)
        mask_col = np.ones(N, dtype=bool)
        w_nk = np.abs(np.random.randn(N, K))
        w_nk = w_nk / w_nk.sum(axis=1, keepdims=True)
        
        N_k, R_k = sufficient_stats_item(X_col, mask_col, w_nk)
        assert np.all(R_k <= N_k + 1e-10)

    def test_all_zeros_gives_R_zero(self):
        """All zero responses give R_k = 0."""
        N, K = 10, 21
        X_col = np.zeros(N)
        mask_col = np.ones(N, dtype=bool)
        w_nk = np.ones((N, K)) / K
        
        N_k, R_k = sufficient_stats_item(X_col, mask_col, w_nk)
        assert_allclose(R_k, np.zeros(K), atol=1e-10)


class TestInitializeParamsMml:
    """Tests for initialize_params_mml function."""

    def test_rasch_a_is_one(self):
        """Rasch model has a = 1 for all items."""
        X = np.random.binomial(1, 0.5, size=(20, 5)).astype(float)
        mask = np.ones_like(X, dtype=bool)
        
        a, b, c = initialize_params_mml(X, mask, model="rasch")
        assert_array_close(a, np.ones(5))
        assert c is None

    def test_b_within_bounds(self):
        """Difficulty estimates are within reasonable bounds."""
        X = np.random.binomial(1, 0.5, size=(20, 5)).astype(float)
        mask = np.ones_like(X, dtype=bool)
        
        a, b, _ = initialize_params_mml(X, mask, model="rasch")
        assert np.all(b >= -5.0)
        assert np.all(b <= 5.0)

    def test_uses_start_values(self):
        """Start values are used when provided."""
        X = np.random.binomial(1, 0.5, size=(20, 5)).astype(float)
        mask = np.ones_like(X, dtype=bool)
        start = {"b": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        
        a, b, _ = initialize_params_mml(X, mask, model="rasch", start=start)
        assert_allclose(b, start["b"])


def assert_array_close(a, b, rtol=1e-10):
    """Helper to check arrays are close."""
    assert_allclose(a, b, rtol=rtol)
