"""
Tests for scoring functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from irt.scoring import (
    score_eap_from_w,
    score_map_from_w,
    score_mle_newton,
    score_map_newton,
    expected_score,
)
from irt.quadrature import make_theta_grid


class TestScoreEapFromW:
    """Tests for score_eap_from_w function."""

    def test_eap_is_posterior_mean(self):
        """EAP equals weighted mean of theta."""
        theta = np.array([-1, 0, 1])
        w_nk = np.array([
            [0.2, 0.6, 0.2],  # Centered
            [0.8, 0.15, 0.05],  # Left-skewed
        ])
        
        theta_hat, se = score_eap_from_w(w_nk, theta)
        
        # Manual calculation
        expected = np.array([
            0.2 * (-1) + 0.6 * 0 + 0.2 * 1,
            0.8 * (-1) + 0.15 * 0 + 0.05 * 1,
        ])
        assert_allclose(theta_hat, expected)

    def test_eap_within_grid(self):
        """EAP is always within theta grid bounds."""
        theta = make_theta_grid(n_points=61)
        N = 100
        
        # Random weights
        w_nk = np.random.rand(N, 61)
        w_nk = w_nk / w_nk.sum(axis=1, keepdims=True)
        
        theta_hat, _ = score_eap_from_w(w_nk, theta)
        
        assert np.all(theta_hat >= theta.min())
        assert np.all(theta_hat <= theta.max())

    def test_se_positive(self):
        """Standard errors are positive."""
        theta = make_theta_grid(n_points=61)
        N = 10
        
        w_nk = np.random.rand(N, 61)
        w_nk = w_nk / w_nk.sum(axis=1, keepdims=True)
        
        _, se = score_eap_from_w(w_nk, theta)
        assert np.all(se > 0)


class TestScoreMapFromW:
    """Tests for score_map_from_w function."""

    def test_map_equals_argmax(self):
        """MAP returns theta with highest weight."""
        theta = np.array([-2, -1, 0, 1, 2])
        w_nk = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0],  # Peak at 0
            [0.1, 0.8, 0.05, 0.03, 0.02],  # Peak at -1
        ])
        
        theta_hat = score_map_from_w(w_nk, theta)
        assert_allclose(theta_hat, [0.0, -1.0])

    def test_map_on_grid(self):
        """MAP is always on the theta grid."""
        theta = make_theta_grid(n_points=61)
        N = 10
        
        w_nk = np.random.rand(N, 61)
        w_nk = w_nk / w_nk.sum(axis=1, keepdims=True)
        
        theta_hat = score_map_from_w(w_nk, theta)
        
        for t in theta_hat:
            assert t in theta


class TestScoreMleNewton:
    """Tests for score_mle_newton function."""

    def test_returns_correct_shape(self):
        """Returns correct shape for theta and SE."""
        N, J = 10, 5
        X = np.random.binomial(1, 0.5, size=(N, J)).astype(float)
        mask = np.ones((N, J), dtype=bool)
        a = np.ones(J)
        b = np.zeros(J)
        
        theta_hat, se = score_mle_newton(X, mask, a, b)
        
        assert theta_hat.shape == (N,)
        assert se.shape == (N,)

    def test_extreme_scores_at_bounds(self):
        """Extreme scores return bound values."""
        # All correct
        X = np.array([[1, 1, 1], [0, 0, 0]])
        mask = np.ones_like(X, dtype=bool)
        a = np.ones(3)
        b = np.zeros(3)
        
        theta_hat, se = score_mle_newton(X, mask, a, b, bounds=(-4, 4))
        
        assert theta_hat[0] == 4.0  # Perfect score -> upper bound
        assert theta_hat[1] == -4.0  # Zero score -> lower bound

    def test_3pl_scoring_runs(self):
        """3PL MLE scoring runs and returns finite values."""
        N, J = 10, 5
        X = np.random.binomial(1, 0.5, size=(N, J)).astype(float)
        mask = np.ones((N, J), dtype=bool)
        a = np.ones(J)
        b = np.zeros(J)
        c = np.full(J, 0.2)

        theta_hat, se = score_mle_newton(X, mask, a, b, c=c)

        assert theta_hat.shape == (N,)
        assert se.shape == (N,)
        assert np.all(np.isfinite(theta_hat))
        assert np.any(np.isfinite(se))

    def test_moderate_scores_within_bounds(self):
        """Moderate scores give estimates within bounds."""
        X = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        mask = np.ones_like(X, dtype=bool)
        a = np.ones(4)
        b = np.zeros(4)
        
        theta_hat, se = score_mle_newton(X, mask, a, b, bounds=(-4, 4))
        
        assert np.all(theta_hat > -4)
        assert np.all(theta_hat < 4)
        assert np.all(np.isfinite(se))


class TestScoreMapNewton:
    """Tests for score_map_newton function."""

    def test_extreme_scores_finite(self):
        """Extreme scores give finite estimates due to prior."""
        # All correct and all wrong
        X = np.array([[1, 1, 1], [0, 0, 0]])
        mask = np.ones_like(X, dtype=bool)
        a = np.ones(3)
        b = np.zeros(3)
        
        theta_hat, se = score_map_newton(
            X, mask, a, b,
            prior_mean=0.0,
            prior_sd=1.0,
        )
        
        # MAP should be finite (pulled toward prior)
        assert np.all(np.isfinite(theta_hat))
        assert np.all(np.isfinite(se))

    def test_shrinkage_toward_prior(self):
        """MAP estimates are shrunk toward prior mean."""
        X = np.array([[1, 1, 1]])  # High raw score
        mask = np.ones_like(X, dtype=bool)
        a = np.ones(3)
        b = np.zeros(3)
        
        theta_mle, _ = score_mle_newton(X, mask, a, b, bounds=(-10, 10))
        theta_map, _ = score_map_newton(X, mask, a, b, prior_mean=0.0, prior_sd=1.0)
        
        # MAP should be between MLE and prior mean
        assert theta_map[0] < theta_mle[0]  # Shrunk from MLE
        assert theta_map[0] > 0  # But still positive


class TestExpectedScore:
    """Tests for expected_score function."""

    def test_increases_with_theta(self):
        """Expected score increases with ability."""
        a = np.ones(5)
        b = np.array([-2, -1, 0, 1, 2])
        
        score_low = expected_score(-2.0, a, b)
        score_mid = expected_score(0.0, a, b)
        score_high = expected_score(2.0, a, b)
        
        assert score_low < score_mid < score_high

    def test_bounded_by_item_count(self):
        """Expected score is between 0 and number of items."""
        a = np.random.rand(10) + 0.5
        b = np.random.randn(10)
        
        for theta in np.linspace(-4, 4, 20):
            score = expected_score(theta, a, b)
            assert 0 <= score <= 10
