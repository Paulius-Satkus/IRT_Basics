"""
Tests for quadrature functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from irt.quadrature import (
    make_theta_grid,
    prior_logpdf,
    gauss_hermite_grid,
    compute_integration_weights,
)


class TestMakeThetaGrid:
    """Tests for make_theta_grid function."""

    def test_basic_grid(self):
        """Creates equally-spaced grid."""
        theta = make_theta_grid(n_points=5, lo=-2, hi=2)
        expected = np.array([-2, -1, 0, 1, 2])
        assert_allclose(theta, expected)

    def test_default_parameters(self):
        """Default parameters work correctly."""
        theta = make_theta_grid()
        assert len(theta) == 61
        assert theta[0] == -4.0
        assert theta[-1] == 4.0

    def test_too_few_points_raises(self):
        """Fewer than 3 points raises ValueError."""
        with pytest.raises(ValueError, match="at least 3"):
            make_theta_grid(n_points=2)

    def test_invalid_bounds_raises(self):
        """lo >= hi raises ValueError."""
        with pytest.raises(ValueError, match="less than hi"):
            make_theta_grid(lo=2.0, hi=1.0)


class TestPriorLogpdf:
    """Tests for prior_logpdf function."""

    def test_normal_prior(self):
        """Normal prior has correct shape."""
        theta = np.array([-2, 0, 2])
        log_prior = prior_logpdf(theta, kind="normal", mean=0.0, sd=1.0)
        
        # Standard normal: log p(0) = -0.5 * log(2*pi)
        expected_at_zero = -0.5 * np.log(2 * np.pi)
        assert_allclose(log_prior[1], expected_at_zero, rtol=1e-10)
        
        # Symmetric around mean
        assert_allclose(log_prior[0], log_prior[2])

    def test_uniform_prior(self):
        """Uniform prior is constant within bounds."""
        theta = np.array([-2, 0, 2])
        log_prior = prior_logpdf(theta, kind="uniform", mean=0.0, sd=1.0)
        
        # Should be constant for theta within [-3, 3]
        assert_allclose(log_prior[0], log_prior[1])
        assert_allclose(log_prior[1], log_prior[2])

    def test_invalid_kind_raises(self):
        """Unknown prior kind raises ValueError."""
        with pytest.raises(ValueError, match="Unknown prior kind"):
            prior_logpdf(np.array([0]), kind="invalid")

    def test_negative_sd_raises(self):
        """Non-positive sd raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            prior_logpdf(np.array([0]), sd=-1.0)

    def test_prior_shape_matches_theta(self):
        """Output shape matches input theta."""
        theta = np.linspace(-3, 3, 100)
        log_prior = prior_logpdf(theta)
        assert log_prior.shape == theta.shape


class TestGaussHermiteGrid:
    """Tests for gauss_hermite_grid function."""

    def test_weights_sum_to_one(self):
        """Weights sum to approximately 1."""
        nodes, weights = gauss_hermite_grid(n_points=21)
        assert_allclose(weights.sum(), 1.0, rtol=1e-6)

    def test_nodes_centered(self):
        """Nodes are centered at mean."""
        nodes, _ = gauss_hermite_grid(n_points=21, mean=0.0)
        # Nodes should be roughly symmetric
        assert_allclose(nodes.mean(), 0.0, atol=1e-10)

    def test_custom_mean_sd(self):
        """Custom mean and sd shift and scale nodes."""
        nodes1, _ = gauss_hermite_grid(n_points=21, mean=0.0, sd=1.0)
        nodes2, _ = gauss_hermite_grid(n_points=21, mean=2.0, sd=2.0)
        
        # nodes2 should be shifted and scaled
        expected = 2.0 + 2.0 * (nodes1 - 0.0) / 1.0
        assert_allclose(nodes2, expected, rtol=1e-10)


class TestComputeIntegrationWeights:
    """Tests for compute_integration_weights function."""

    def test_weights_sum_to_one(self):
        """Integration weights sum to 1."""
        theta = make_theta_grid(n_points=61)
        weights = compute_integration_weights(theta)
        assert_allclose(weights.sum(), 1.0, rtol=1e-4)

    def test_weights_positive(self):
        """All weights are positive."""
        theta = make_theta_grid(n_points=61)
        weights = compute_integration_weights(theta)
        assert np.all(weights > 0)

    def test_shape_matches_theta(self):
        """Output shape matches input theta."""
        theta = make_theta_grid(n_points=31)
        weights = compute_integration_weights(theta)
        assert weights.shape == theta.shape
