"""
Tests for item parameter priors.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit

from irt.data import parse_item_priors
from irt.item_priors import prior_logpdf_grad_hess
from irt.mstep import rasch_Q_grad_hess, twopl_Q_and_grad, update_item_rasch_newton
from irt import fit


def _finite_diff_grad_hess(func, x, h=1e-5):
    f0 = func(x)
    f1 = func(x + h)
    f2 = func(x - h)
    grad = (f1 - f2) / (2 * h)
    hess = (f1 - 2 * f0 + f2) / (h * h)
    return grad, hess


class TestPriorDerivatives:
    def test_normal(self):
        params = {"mean": 0.5, "sd": 1.2}
        x = 0.3
        logp, grad, hess = prior_logpdf_grad_hess("normal", x, params)
        grad_fd, hess_fd = _finite_diff_grad_hess(
            lambda v: prior_logpdf_grad_hess("normal", v, params)[0], x
        )
        assert_allclose(grad, grad_fd, atol=1e-5)
        assert_allclose(hess, hess_fd, atol=1e-5)

    def test_lognormal(self):
        params = {"mean": 0.0, "sd": 0.6}
        x = 1.2
        logp, grad, hess = prior_logpdf_grad_hess("lognormal", x, params)
        grad_fd, hess_fd = _finite_diff_grad_hess(
            lambda v: prior_logpdf_grad_hess("lognormal", v, params)[0], x
        )
        assert_allclose(grad, grad_fd, atol=1e-5)
        assert_allclose(hess, hess_fd, atol=1e-5)

    def test_gamma(self):
        params = {"shape": 2.0, "scale": 0.7}
        x = 1.1
        logp, grad, hess = prior_logpdf_grad_hess("gamma", x, params)
        grad_fd, hess_fd = _finite_diff_grad_hess(
            lambda v: prior_logpdf_grad_hess("gamma", v, params)[0], x
        )
        assert_allclose(grad, grad_fd, atol=1e-5)
        assert_allclose(hess, hess_fd, atol=1e-5)

    def test_beta(self):
        params = {"alpha": 2.0, "beta": 4.0}
        x = 0.3
        logp, grad, hess = prior_logpdf_grad_hess("beta", x, params)
        grad_fd, hess_fd = _finite_diff_grad_hess(
            lambda v: prior_logpdf_grad_hess("beta", v, params)[0], x
        )
        assert_allclose(grad, grad_fd, atol=1e-5)
        assert_allclose(hess, hess_fd, atol=1e-5)


class TestParseItemPriors:
    def test_dict_grouped_and_per_item(self):
        item_names = ["i1", "i2", "i3"]
        priors = {
            "b": {"dist": "normal", "mean": 0.0, "sd": 1.0, "items": ["i1", "i3"]},
            "a": [{"dist": "lognormal", "mean": 0.0, "sd": 0.5, "item": "i2"}],
        }
        parsed = parse_item_priors(priors, item_names, model="2pl")
        assert parsed["b"][0] is not None
        assert parsed["b"][1] is None
        assert parsed["b"][2] is not None
        assert parsed["a"][1] is not None

    def test_dataframe(self):
        pd = pytest.importorskip("pandas")
        item_names = ["i1", "i2"]
        df = pd.DataFrame(
            [{"item": "i2", "param": "b", "dist": "normal", "mean": 0.2, "sd": 0.7}]
        )
        parsed = parse_item_priors(df, item_names, model="rasch")
        assert parsed["b"][1]["kind"] == "normal"
        assert parsed["b"][0] is None

    def test_rasch_ignores_a_priors(self):
        item_names = ["i1", "i2"]
        priors = {"a": {"dist": "lognormal", "mean": 0.0, "sd": 0.5}}
        with pytest.warns(UserWarning, match="Ignoring item priors"):
            parsed = parse_item_priors(priors, item_names, model="rasch")
        assert all(p is None for p in parsed["a"])

    def test_bounds_validation(self):
        item_names = ["i1"]
        priors = {"b": {"dist": "beta", "alpha": 2.0, "beta": 3.0}}
        with pytest.raises(ValueError, match="requires bounds above"):
            parse_item_priors(priors, item_names, model="rasch", b_bounds=(-6.0, 6.0))

    def test_3pl_c_gamma_prior(self):
        item_names = ["i1", "i2"]
        priors = {"c": {"dist": "gamma", "shape": 2.0, "scale": 0.05}}
        parsed = parse_item_priors(
            priors,
            item_names,
            model="3pl",
            c_bounds=(1e-6, 0.35),
        )
        assert parsed["c"][0]["kind"] == "gamma"
        assert parsed["c"][1]["kind"] == "gamma"


class TestMstepPriors:
    def test_rasch_grad_includes_prior(self):
        theta = np.linspace(-3, 3, 31)
        p_k = expit(theta - 0.5)
        N_k = np.ones_like(theta) * 50
        R_k = N_k * p_k
        prior = {"kind": "normal", "params": {"mean": 0.0, "sd": 1.0}}

        _, grad_no, _ = rasch_Q_grad_hess(0.5, theta, N_k, R_k)
        _, grad_yes, _ = rasch_Q_grad_hess(0.5, theta, N_k, R_k, prior=prior)
        _, prior_grad, _ = prior_logpdf_grad_hess("normal", 0.5, prior["params"])

        assert_allclose(grad_yes - grad_no, prior_grad, atol=1e-8)

    def test_rasch_update_shrinks_to_prior(self):
        theta = np.linspace(-3, 3, 61)
        true_b = 2.0
        p_k = expit(theta - true_b)
        N_k = np.ones_like(theta) * 200
        R_k = N_k * p_k

        b_no = update_item_rasch_newton(
            theta=theta, N_k=N_k, R_k=R_k, b0=0.0, bounds=(-6, 6)
        )
        prior = {"kind": "normal", "params": {"mean": 0.0, "sd": 0.3}}
        b_yes = update_item_rasch_newton(
            theta=theta, N_k=N_k, R_k=R_k, b0=0.0, bounds=(-6, 6), prior=prior
        )
        assert abs(b_yes) < abs(b_no)

    def test_twopl_grad_includes_priors(self):
        theta = np.linspace(-3, 3, 31)
        a, b = 1.2, -0.4
        p_k = expit(a * (theta - b))
        N_k = np.ones_like(theta) * 60
        R_k = N_k * p_k
        prior_a = {"kind": "lognormal", "params": {"mean": 0.0, "sd": 0.5}}
        prior_b = {"kind": "normal", "params": {"mean": 0.0, "sd": 1.0}}

        neg_Q_no, grad_no = twopl_Q_and_grad(
            np.array([a, b]), theta, N_k, R_k
        )
        neg_Q_yes, grad_yes = twopl_Q_and_grad(
            np.array([a, b]), theta, N_k, R_k, prior_a=prior_a, prior_b=prior_b
        )
        _, grad_a, _ = prior_logpdf_grad_hess("lognormal", a, prior_a["params"])
        _, grad_b, _ = prior_logpdf_grad_hess("normal", b, prior_b["params"])

        assert_allclose(grad_yes[0] - grad_no[0], -grad_a, atol=1e-8)
        assert_allclose(grad_yes[1] - grad_no[1], -grad_b, atol=1e-8)


class TestFitWithPriors:
    def test_rasch_prior_shrinks_difficulty(self):
        rng = np.random.default_rng(123)
        n_persons, n_items = 300, 6
        theta = rng.standard_normal(n_persons)
        b_true = np.array([1.5, 0.2, -0.4, 0.0, 0.5, -0.2])
        p = expit(theta[:, None] - b_true[None, :])
        X = (rng.random((n_persons, n_items)) < p).astype(float)

        result_no = fit(X, model="rasch", estimator="mml_em")
        priors = {"b": {"dist": "normal", "mean": 0.0, "sd": 0.25, "item": 0}}
        result_yes = fit(X, model="rasch", estimator="mml_em", priors=priors)

        assert result_yes.params["b"][0] < result_no.params["b"][0]
