"""
Tests for core mathematical utilities.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from irt.core import (
    clip_prob,
    bernoulli_loglik_from_p,
    prob_2pl,
    prob_matrix_2pl,
    log_prob_2pl,
    center_difficulties,
    info_2pl,
)


class TestClipProb:
    """Tests for clip_prob function."""

    def test_clips_zeros(self):
        """Probabilities of 0 are clipped to eps."""
        p = np.array([0.0, 0.5, 1.0])
        result = clip_prob(p, eps=1e-10)
        assert result[0] == 1e-10
        assert result[1] == 0.5
        assert result[2] == 1 - 1e-10

    def test_preserves_valid_probs(self):
        """Valid probabilities are unchanged."""
        p = np.array([0.1, 0.5, 0.9])
        result = clip_prob(p)
        assert_allclose(result, p)

    def test_custom_eps(self):
        """Custom epsilon is respected."""
        p = np.array([0.0, 1.0])
        result = clip_prob(p, eps=1e-6)
        assert result[0] == 1e-6
        assert result[1] == 1 - 1e-6


class TestBernoulliLoglik:
    """Tests for bernoulli_loglik_from_p function."""

    def test_basic_computation(self):
        """Basic log-likelihood computation is correct."""
        p = np.array([0.8, 0.2, 0.5])
        x = np.array([1, 0, 1])
        result = bernoulli_loglik_from_p(p, x)
        expected = np.array([np.log(0.8), np.log(0.8), np.log(0.5)])
        assert_allclose(result, expected, rtol=1e-10)

    def test_finite_at_extremes(self):
        """Log-likelihood is finite even for extreme probabilities."""
        p = np.array([0.0, 1.0, 1e-15, 1 - 1e-15])
        x = np.array([0, 1, 0, 1])
        result = bernoulli_loglik_from_p(p, x)
        assert np.all(np.isfinite(result))

    def test_vectorized(self):
        """Function works with 2D arrays."""
        p = np.array([[0.6, 0.4], [0.7, 0.3]])
        x = np.array([[1, 0], [1, 1]])
        result = bernoulli_loglik_from_p(p, x)
        assert result.shape == (2, 2)
        assert np.all(np.isfinite(result))


class TestProb2pl:
    """Tests for prob_2pl function."""

    def test_basic_computation(self):
        """Basic probability computation is correct."""
        theta = np.array([0.0])
        result = prob_2pl(theta, a=1.0, b=0.0)
        assert_allclose(result, [0.5])

    def test_monotone_in_theta(self):
        """Probability is monotonically increasing in theta."""
        theta = np.linspace(-4, 4, 100)
        p = prob_2pl(theta, a=1.0, b=0.0)
        assert np.all(np.diff(p) > 0)

    def test_monotone_in_discrimination(self):
        """Steepness increases with discrimination."""
        theta = np.linspace(-2, 2, 100)
        p_low = prob_2pl(theta, a=0.5, b=0.0)
        p_high = prob_2pl(theta, a=2.0, b=0.0)
        
        # Higher discrimination = steeper curve
        # Variance of p should be higher for steeper curves
        assert np.var(p_high) > np.var(p_low)

    def test_difficulty_shift(self):
        """Difficulty shifts the curve along theta."""
        theta = np.array([0.0, 1.0, 2.0])
        p1 = prob_2pl(theta, a=1.0, b=0.0)
        p2 = prob_2pl(theta, a=1.0, b=1.0)
        
        # p2 at theta=1 should equal p1 at theta=0
        assert_allclose(p2[1], p1[0], rtol=1e-10)

    def test_symmetric(self):
        """P(theta=b) = 0.5 for any b."""
        for b in [-2, 0, 2]:
            theta = np.array([b])
            p = prob_2pl(theta, a=1.5, b=b)
            assert_allclose(p, [0.5])


class TestProbMatrix2pl:
    """Tests for prob_matrix_2pl function."""

    def test_shape(self):
        """Output has correct shape (K, J)."""
        theta = np.array([-1, 0, 1])
        a_vec = np.array([1.0, 1.5, 2.0])
        b_vec = np.array([0.0, 0.5, -0.5])
        P = prob_matrix_2pl(theta, a_vec, b_vec)
        assert P.shape == (3, 3)

    def test_consistency_with_scalar(self):
        """Matrix version matches scalar version."""
        theta = np.array([-1, 0, 1])
        a_vec = np.array([1.0, 1.5])
        b_vec = np.array([0.0, 0.5])
        P = prob_matrix_2pl(theta, a_vec, b_vec)
        
        for k, t in enumerate(theta):
            for j in range(len(a_vec)):
                expected = prob_2pl(np.array([t]), a_vec[j], b_vec[j])[0]
                assert_allclose(P[k, j], expected)


class TestLogProb2pl:
    """Tests for log_prob_2pl function."""

    def test_consistency(self):
        """log(p) and log(1-p) are consistent with prob_2pl."""
        theta = np.array([-2, 0, 2])
        a, b = 1.5, 0.5
        
        log_p, log_q = log_prob_2pl(theta, a, b)
        p = prob_2pl(theta, a, b)
        
        assert_allclose(np.exp(log_p), p, rtol=1e-10)
        assert_allclose(np.exp(log_q), 1 - p, rtol=1e-10)

    def test_numerical_stability(self):
        """Stable for extreme theta values."""
        theta = np.array([-50, 50])
        log_p, log_q = log_prob_2pl(theta, a=1.0, b=0.0)
        
        assert np.all(np.isfinite(log_p))
        assert np.all(np.isfinite(log_q))


class TestCenterDifficulties:
    """Tests for center_difficulties function."""

    def test_centers_to_zero(self):
        """Centered difficulties have mean zero."""
        b = np.array([1.0, 2.0, 3.0])
        b_centered = center_difficulties(b)
        assert_allclose(b_centered.mean(), 0.0, atol=1e-10)

    def test_with_fixed_items(self):
        """Only estimated items contribute to mean."""
        b = np.array([1.0, 2.0, 10.0])
        mask = np.array([True, True, False])  # Third item fixed
        b_centered = center_difficulties(b, est_mask=mask)
        
        # Fixed item unchanged
        assert b_centered[2] == 10.0
        # Estimated items have mean 0
        assert_allclose(b_centered[mask].mean(), 0.0, atol=1e-10)

    def test_preserves_differences(self):
        """Centering preserves differences between items."""
        b = np.array([1.0, 3.0, 5.0])
        b_centered = center_difficulties(b)
        
        # Differences should be preserved
        assert_allclose(b_centered[1] - b_centered[0], 2.0)
        assert_allclose(b_centered[2] - b_centered[1], 2.0)


class TestInfo2pl:
    """Tests for info_2pl function."""

    def test_maximum_at_difficulty(self):
        """Information is maximized at theta = b."""
        theta = np.linspace(-3, 3, 1000)
        b = 0.5
        info = info_2pl(theta, a=1.0, b=b)
        
        max_idx = np.argmax(info)
        assert_allclose(theta[max_idx], b, atol=0.01)

    def test_max_value_formula(self):
        """Maximum information equals a^2/4."""
        theta = np.array([0.0])  # At difficulty
        for a in [0.5, 1.0, 2.0]:
            info = info_2pl(theta, a=a, b=0.0)
            assert_allclose(info[0], a**2 / 4, rtol=1e-10)

    def test_positive(self):
        """Information is always positive."""
        theta = np.linspace(-4, 4, 100)
        info = info_2pl(theta, a=1.0, b=0.0)
        assert np.all(info > 0)
