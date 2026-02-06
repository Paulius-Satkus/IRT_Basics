"""Integration tests for polytomous model fitting."""

import numpy as np
import pytest

from irt import fit
from irt.core_poly import prob_pcm, prob_gpcm, prob_grm
from irt.data import as_polytomous_matrix


def _simulate_pcm(n_persons, n_items, n_cat, theta, b):
    X = np.zeros((n_persons, n_items))
    for i in range(n_persons):
        for j in range(n_items):
            p = prob_pcm(np.array([theta[i]]), b[j], n_cat)[0]
            X[i, j] = np.random.choice(n_cat, p=p)
    return X.astype(float)


class TestPolytomousFit:
    def test_pcm_converges(self):
        np.random.seed(42)
        theta = np.random.randn(100) * 0.8
        b = [np.array([-0.5, 0, 0.5]) for _ in range(5)]
        X = _simulate_pcm(100, 5, 4, theta, b)
        result = fit(X, model="pcm")
        assert result.converged
        assert result.loglik is not None

    def test_gpcm_converges(self):
        np.random.seed(43)
        X = np.random.randint(0, 4, size=(80, 6)).astype(float)
        result = fit(X, model="gpcm")
        assert result.converged

    def test_grm_converges(self):
        np.random.seed(44)
        X = np.random.randint(0, 4, size=(80, 6)).astype(float)
        result = fit(X, model="grm")
        assert result.converged

    def test_rsm_converges(self):
        np.random.seed(45)
        X = np.random.randint(0, 4, size=(80, 6)).astype(float)
        result = fit(X, model="rsm")
        assert result.converged

    def test_nrm_fits(self):
        np.random.seed(46)
        X = np.random.randint(0, 3, size=(80, 5)).astype(float)
        result = fit(X, model="nrm")
        assert "a" in result.params and "c" in result.params

    def test_eap_map_mle_finite(self):
        np.random.seed(47)
        X = np.random.randint(0, 4, size=(50, 5)).astype(float)
        result = fit(X, model="pcm")
        eap = result.score(method="eap")
        map_s = result.score(method="map")
        mle = result.score(method="mle")
        assert np.all(np.isfinite(eap.theta))
        assert np.all(np.isfinite(map_s.theta))
        assert np.all(np.isfinite(mle.theta))

    def test_fit_accepts_model_names(self):
        X = np.random.randint(0, 4, size=(30, 4)).astype(float)
        for model in ["pcm", "rsm", "grm", "gpcm", "nrm"]:
            result = fit(X, model=model)
            assert result.model == model

    def test_fit_result_params_shape(self):
        X = np.random.randint(0, 4, size=(40, 5)).astype(float)
        result = fit(X, model="pcm")
        assert "b" in result.params
        assert result.params["b"].shape[0] == 5

    def test_binary_unchanged(self):
        X = np.random.binomial(1, 0.5, size=(50, 10)).astype(float)
        result = fit(X, model="rasch")
        assert result.model == "rasch"
        assert "a" in result.params
        assert "b" in result.params
