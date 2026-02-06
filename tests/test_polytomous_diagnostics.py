"""Integration tests for polytomous diagnostics."""

import numpy as np
import pytest

from irt import fit


class TestPolytomousDiagnostics:
    @pytest.fixture
    def result(self):
        np.random.seed(42)
        X = np.random.randint(0, 4, size=(80, 6)).astype(float)
        return fit(X, model="pcm")

    def test_item_fit_returns_dataframe(self, result):
        df = result.item_fit()
        assert "item" in df.columns
        assert "infit_ms" in df.columns
        assert "outfit_ms" in df.columns
        assert len(df) == 6

    def test_person_fit_works(self, result):
        df = result.person_fit()
        assert "person" in df.columns
        assert "theta" in df.columns
        assert "infit_ms" in df.columns
        assert len(df) == 80

    def test_model_fit_returns_dict(self, result):
        d = result.model_fit()
        assert "n_persons" in d
        assert "n_items" in d
        assert "loglik" in d
        assert "aic" in d
        assert "bic" in d
