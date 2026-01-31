"""
Integration tests for the public API.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from irt import fit, FitResult, ScoreResult


def generate_rasch_data(n_persons=100, n_items=20, seed=42):
    """Generate simulated Rasch data for testing."""
    np.random.seed(seed)
    
    # True parameters
    theta = np.random.randn(n_persons)
    b = np.random.randn(n_items)
    
    # Generate responses
    from scipy.special import expit
    p = expit(theta[:, None] - b[None, :])
    X = (np.random.rand(n_persons, n_items) < p).astype(float)
    
    return X, theta, b


def generate_2pl_data(n_persons=100, n_items=20, seed=42):
    """Generate simulated 2PL data for testing."""
    np.random.seed(seed)
    
    # True parameters
    theta = np.random.randn(n_persons)
    a = np.random.rand(n_items) * 1.5 + 0.5  # 0.5 to 2.0
    b = np.random.randn(n_items)
    
    # Generate responses
    from scipy.special import expit
    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    X = (np.random.rand(n_persons, n_items) < p).astype(float)
    
    return X, theta, a, b


def generate_3pl_data(n_persons=100, n_items=20, seed=42):
    """Generate simulated 3PL data for testing."""
    np.random.seed(seed)

    theta = np.random.randn(n_persons)
    a = np.random.rand(n_items) * 1.5 + 0.5  # 0.5 to 2.0
    b = np.random.randn(n_items)
    c = np.random.rand(n_items) * 0.2 + 0.05  # 0.05 to 0.25

    from scipy.special import expit
    p_2pl = expit(a[None, :] * (theta[:, None] - b[None, :]))
    p = c[None, :] + (1.0 - c[None, :]) * p_2pl
    X = (np.random.rand(n_persons, n_items) < p).astype(float)

    return X, theta, a, b, c


class TestFitRaschMmlEm:
    """Tests for fitting Rasch model with MML-EM."""

    def test_basic_fit(self):
        """Basic fit returns correct types."""
        X, _, _ = generate_rasch_data()
        
        result = fit(X, model="rasch", estimator="mml_em")
        
        assert isinstance(result, FitResult)
        assert result.model == "rasch"
        assert result.estimator == "mml_em"
        assert "a" in result.params
        assert "b" in result.params

    def test_converges(self):
        """Model converges on reasonable data."""
        X, _, _ = generate_rasch_data()
        
        result = fit(X, model="rasch", estimator="mml_em")
        
        assert result.converged

    def test_a_is_one_for_rasch(self):
        """Discrimination is 1 for all items in Rasch."""
        X, _, _ = generate_rasch_data()
        
        result = fit(X, model="rasch", estimator="mml_em")
        
        assert_allclose(result.params["a"], np.ones(20))

    def test_loglik_increases(self):
        """Log-likelihood is non-decreasing across iterations."""
        X, _, _ = generate_rasch_data()
        
        result = fit(X, model="rasch", estimator="mml_em")
        
        loglik_history = result.history["loglik"]
        for i in range(1, len(loglik_history)):
            # Allow tiny numerical noise
            assert loglik_history[i] >= loglik_history[i-1] - 1e-6

    def test_recovers_difficulty_order(self):
        """Estimated difficulties correlate with true difficulties."""
        X, _, b_true = generate_rasch_data(seed=123)
        
        result = fit(X, model="rasch", estimator="mml_em")
        
        # Correlation should be high
        corr = np.corrcoef(result.params["b"], b_true)[0, 1]
        assert corr > 0.7  # Should be quite correlated


class TestFit2plMmlEm:
    """Tests for fitting 2PL model with MML-EM."""

    def test_basic_fit(self):
        """Basic fit returns correct types."""
        X, _, _, _ = generate_2pl_data()
        
        result = fit(X, model="2pl", estimator="mml_em")
        
        assert isinstance(result, FitResult)
        assert result.model == "2pl"
        assert result.params["a"].shape == (20,)

    def test_a_within_bounds(self):
        """Discrimination estimates are within default bounds."""
        X, _, _, _ = generate_2pl_data()
        
        result = fit(X, model="2pl", estimator="mml_em")
        
        assert np.all(result.params["a"] >= 0.25)
        assert np.all(result.params["a"] <= 4.0)


class TestFitJmle:
    """Tests for fitting with JMLE."""

    def test_basic_fit(self):
        """Basic JMLE fit works."""
        X, _, _ = generate_rasch_data()
        
        result = fit(X, model="rasch", estimator="jmle")
        
        assert isinstance(result, FitResult)
        assert result.estimator == "jmle"

    def test_2pl_jmle_raises(self):
        """JMLE with 2PL raises ValueError."""
        X, _, _, _ = generate_2pl_data()
        
        with pytest.raises(ValueError, match="JMLE.*only supports Rasch"):
            fit(X, model="2pl", estimator="jmle")


class TestFit3plMmlEm:
    """Tests for fitting 3PL model with MML-EM."""

    def test_basic_fit(self):
        """Basic 3PL fit returns correct types."""
        X, _, _, _, _ = generate_3pl_data()

        result = fit(X, model="3pl", estimator="mml_em")

        assert isinstance(result, FitResult)
        assert result.model == "3pl"
        assert "c" in result.params
        assert result.params["c"].shape == (20,)

    def test_c_within_bounds(self):
        """Guessing estimates respect bounds."""
        X, _, _, _, _ = generate_3pl_data()

        result = fit(
            X,
            model="3pl",
            estimator="mml_em",
            technical={"c_bounds": (0.05, 0.3)},
        )

        assert np.all(result.params["c"] >= 0.05)
        assert np.all(result.params["c"] <= 0.3)

    def test_fixed_c(self):
        """Fixed guessing parameters remain unchanged."""
        X, _, _, _, _ = generate_3pl_data()
        c_fixed = np.full(X.shape[1], 0.15)

        result = fit(
            X,
            model="3pl",
            estimator="mml_em",
            start={"c": c_fixed},
            fixed={"c": np.ones_like(c_fixed, dtype=bool)},
        )

        assert_allclose(result.params["c"], c_fixed)


class TestScoring:
    """Tests for scoring functionality."""

    def test_eap_scoring(self):
        """EAP scoring returns correct types."""
        X, _, _ = generate_rasch_data()
        result = fit(X, model="rasch", estimator="mml_em")
        
        scores = result.score(method="eap")
        
        assert isinstance(scores, ScoreResult)
        assert scores.theta.shape == (100,)
        assert scores.se.shape == (100,)
        assert scores.method == "eap"

    def test_map_scoring(self):
        """MAP scoring works."""
        X, _, _ = generate_rasch_data()
        result = fit(X, model="rasch", estimator="mml_em")
        
        scores = result.score(method="map")
        
        assert scores.method == "map"
        assert np.all(np.isfinite(scores.theta))

    def test_mle_scoring(self):
        """MLE scoring works."""
        X, _, _ = generate_rasch_data()
        result = fit(X, model="rasch", estimator="mml_em")
        
        scores = result.score(method="mle")
        
        assert scores.method == "mle"

    def test_score_correlates_with_raw(self):
        """Ability estimates correlate with raw scores."""
        X, _, _ = generate_rasch_data()
        result = fit(X, model="rasch", estimator="mml_em")
        
        scores = result.score(method="eap")
        raw_scores = np.nansum(X, axis=1)
        
        corr = np.corrcoef(scores.theta, raw_scores)[0, 1]
        assert corr > 0.8

    def test_score_new_data(self):
        """Can score new data."""
        X, _, _ = generate_rasch_data(n_persons=100, seed=1)
        X_new, _, _ = generate_rasch_data(n_persons=50, seed=2)
        
        result = fit(X, model="rasch", estimator="mml_em")
        scores = result.score(X=X_new, method="eap")
        
        assert scores.theta.shape == (50,)


class TestValidation:
    """Tests for input validation."""

    def test_invalid_model_raises(self):
        """Invalid model name raises ValueError."""
        X, _, _ = generate_rasch_data()
        
        with pytest.raises(ValueError, match="Invalid model"):
            fit(X, model="invalid")

    def test_invalid_estimator_raises(self):
        """Invalid estimator name raises ValueError."""
        X, _, _ = generate_rasch_data()
        
        with pytest.raises(ValueError, match="Invalid estimator"):
            fit(X, estimator="invalid")

    def test_invalid_data_raises(self):
        """Invalid data values raise ValueError."""
        X = np.array([[1, 0, 2], [0, 1, 0]])
        
        with pytest.raises(ValueError, match="invalid values"):
            fit(X)


class TestMissingData:
    """Tests for handling missing data."""

    def test_handles_nan(self):
        """Model handles NaN values."""
        X, _, _ = generate_rasch_data()
        
        # Introduce missing data
        mask = np.random.rand(*X.shape) > 0.1
        X_missing = X.copy()
        X_missing[~mask] = np.nan
        
        result = fit(X_missing, model="rasch", estimator="mml_em")
        
        assert result.converged


class TestTechnicalParameters:
    """Tests for technical parameter customization."""

    def test_custom_quadpts(self):
        """Custom quadrature points work."""
        X, _, _ = generate_rasch_data()
        
        result = fit(
            X,
            model="rasch",
            technical={"quadpts": 81, "max_iter": 100}
        )
        
        assert len(result.theta_grid) == 81

    def test_invalid_technical_raises(self):
        """Invalid technical parameter raises ValueError."""
        X, _, _ = generate_rasch_data()
        
        with pytest.raises(ValueError, match="Invalid technical"):
            fit(X, technical={"invalid_param": 10})


class TestReports:
    """Tests for report generation."""

    def test_item_report(self):
        """Item report works (requires pandas)."""
        pytest.importorskip("pandas")
        
        X, _, _ = generate_rasch_data()
        result = fit(X, model="rasch")
        
        report = result.item_report()
        
        assert "a" in report.columns
        assert "b" in report.columns
        assert len(report) == 20

    def test_score_result_to_dataframe(self):
        """ScoreResult.to_dataframe works."""
        pytest.importorskip("pandas")
        
        X, _, _ = generate_rasch_data()
        result = fit(X, model="rasch")
        scores = result.score(method="eap")
        
        df = scores.to_dataframe()
        
        assert "theta" in df.columns
        assert "se" in df.columns
        assert len(df) == 100
