"""
Tests for M-step solvers.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from irt.mstep import (
    rasch_Q_grad_hess,
    update_item_rasch_newton,
    twopl_Q_and_grad,
    update_item_2pl_lbfgsb,
)
from irt.quadrature import make_theta_grid


class TestRaschQGradHess:
    """Tests for rasch_Q_grad_hess function."""

    def test_gradient_at_optimum_near_zero(self):
        """Gradient is near zero at the optimum."""
        # Create data where b* ≈ 0
        theta = make_theta_grid(n_points=61)
        
        # Symmetric N_k and R_k around theta=0
        from scipy.special import expit
        p_k = expit(theta - 0.0)  # True b = 0
        N_k = np.ones_like(theta) * 100
        R_k = N_k * p_k
        
        # Evaluate at optimum
        Q, grad, hess = rasch_Q_grad_hess(b=0.0, theta=theta, N_k=N_k, R_k=R_k)
        
        # Gradient should be near zero
        assert abs(grad) < 0.1

    def test_hessian_negative(self):
        """Hessian is negative (Q is concave)."""
        theta = make_theta_grid(n_points=61)
        N_k = np.ones_like(theta) * 10
        R_k = N_k * 0.5
        
        Q, grad, hess = rasch_Q_grad_hess(b=0.0, theta=theta, N_k=N_k, R_k=R_k)
        assert hess < 0

    def test_Q_finite(self):
        """Q function value is finite."""
        theta = make_theta_grid(n_points=61)
        N_k = np.random.rand(61) * 10
        R_k = np.random.rand(61) * N_k
        
        Q, grad, hess = rasch_Q_grad_hess(b=1.5, theta=theta, N_k=N_k, R_k=R_k)
        assert np.isfinite(Q)
        assert np.isfinite(grad)
        assert np.isfinite(hess)


class TestUpdateItemRaschNewton:
    """Tests for update_item_rasch_newton function."""

    def test_converges(self):
        """Newton iteration converges."""
        theta = make_theta_grid(n_points=61)
        
        # Create data with true b = 1.0
        from scipy.special import expit
        p_k = expit(theta - 1.0)
        N_k = np.ones_like(theta) * 100
        R_k = N_k * p_k
        
        b_hat = update_item_rasch_newton(
            theta=theta,
            N_k=N_k,
            R_k=R_k,
            b0=0.0,  # Start far from true
            bounds=(-6, 6),
        )
        
        # Should recover true value approximately
        assert_allclose(b_hat, 1.0, atol=0.1)

    def test_respects_bounds(self):
        """Result stays within bounds."""
        theta = make_theta_grid(n_points=61)
        N_k = np.ones_like(theta) * 10
        
        # Data that would push b very negative
        R_k = N_k * 0.99
        
        b_hat = update_item_rasch_newton(
            theta=theta,
            N_k=N_k,
            R_k=R_k,
            b0=0.0,
            bounds=(-3, 3),
        )
        
        assert b_hat >= -3
        assert b_hat <= 3


class TestTwoplQAndGrad:
    """Tests for twopl_Q_and_grad function."""

    def test_gradient_shape(self):
        """Gradient has correct shape."""
        theta = make_theta_grid(n_points=61)
        N_k = np.ones_like(theta) * 10
        R_k = N_k * 0.5
        
        params = np.array([1.0, 0.0])  # [a, b]
        neg_Q, neg_grad = twopl_Q_and_grad(params, theta, N_k, R_k)
        
        assert np.isscalar(neg_Q)
        assert neg_grad.shape == (2,)

    def test_finite_values(self):
        """Q and gradient are finite."""
        theta = make_theta_grid(n_points=61)
        N_k = np.random.rand(61) * 10 + 1
        R_k = np.random.rand(61) * N_k
        
        params = np.array([1.5, -0.5])
        neg_Q, neg_grad = twopl_Q_and_grad(params, theta, N_k, R_k)
        
        assert np.isfinite(neg_Q)
        assert np.all(np.isfinite(neg_grad))


class TestUpdateItem2plLbfgsb:
    """Tests for update_item_2pl_lbfgsb function."""

    def test_returns_finite(self):
        """Returns finite values."""
        theta = make_theta_grid(n_points=61)
        
        from scipy.special import expit
        true_a, true_b = 1.5, 0.5
        p_k = expit(true_a * (theta - true_b))
        N_k = np.ones_like(theta) * 50
        R_k = N_k * p_k
        
        a_hat, b_hat = update_item_2pl_lbfgsb(
            theta=theta,
            N_k=N_k,
            R_k=R_k,
            a0=1.0,
            b0=0.0,
            a_bounds=(0.25, 4.0),
            b_bounds=(-6.0, 6.0),
        )
        
        assert np.isfinite(a_hat)
        assert np.isfinite(b_hat)

    def test_improves_Q(self):
        """Optimization improves (decreases negative) Q."""
        theta = make_theta_grid(n_points=61)
        
        from scipy.special import expit
        true_a, true_b = 1.5, 0.5
        p_k = expit(true_a * (theta - true_b))
        N_k = np.ones_like(theta) * 50
        R_k = N_k * p_k
        
        # Starting values
        a0, b0 = 1.0, 0.0
        neg_Q_start, _ = twopl_Q_and_grad(np.array([a0, b0]), theta, N_k, R_k)
        
        # Optimized values
        a_hat, b_hat = update_item_2pl_lbfgsb(
            theta=theta,
            N_k=N_k,
            R_k=R_k,
            a0=a0,
            b0=b0,
            a_bounds=(0.25, 4.0),
            b_bounds=(-6.0, 6.0),
        )
        
        neg_Q_end, _ = twopl_Q_and_grad(np.array([a_hat, b_hat]), theta, N_k, R_k)
        
        # neg_Q should decrease (Q increases)
        assert neg_Q_end <= neg_Q_start + 1e-6

    def test_respects_bounds(self):
        """Result stays within specified bounds."""
        theta = make_theta_grid(n_points=61)
        N_k = np.ones_like(theta) * 10
        R_k = N_k * 0.99  # Would push a and b extreme
        
        a_hat, b_hat = update_item_2pl_lbfgsb(
            theta=theta,
            N_k=N_k,
            R_k=R_k,
            a0=1.0,
            b0=0.0,
            a_bounds=(0.5, 2.0),
            b_bounds=(-2.0, 2.0),
        )
        
        assert a_hat >= 0.5
        assert a_hat <= 2.0
        assert b_hat >= -2.0
        assert b_hat <= 2.0
