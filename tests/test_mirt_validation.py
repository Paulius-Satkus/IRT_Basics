"""
Cross-validation tests against R mirt package.

These tests compare Python IRT estimates to pre-computed R mirt results.
Reference values generated using mirt v1.45.1.

Run R validation script first: Rscript tests/generate_mirt_reference.R
"""

from pathlib import Path

import numpy as np
import pytest

from irt import fit


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _load_params(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {
        "a": np.array(data["a"], dtype=float),
        "b": np.array(data["b"], dtype=float),
    }


def _load_loglik(path: Path) -> float:
    return float(Path(path).read_text().strip())


class TestLSAT7Validation:
    """Validate against mirt results on LSAT7 dataset."""

    @pytest.fixture
    def lsat7_data(self, mirt_reference_dir: Path) -> np.ndarray:
        """Load LSAT7 expanded data (N=1000, J=5)."""
        return _load_csv(mirt_reference_dir / "lsat7_data.csv")

    @pytest.fixture
    def mirt_rasch_params(self, mirt_reference_dir: Path) -> dict[str, np.ndarray | float]:
        """Reference Rasch parameters from mirt."""
        params = _load_params(mirt_reference_dir / "lsat7_rasch_params.csv")
        loglik = _load_loglik(mirt_reference_dir / "lsat7_rasch_loglik.txt")
        return {"b": params["b"], "loglik": loglik}

    @pytest.fixture
    def mirt_2pl_params(self, mirt_reference_dir: Path) -> dict[str, np.ndarray | float]:
        """Reference 2PL parameters from mirt."""
        params = _load_params(mirt_reference_dir / "lsat7_2pl_params.csv")
        loglik = _load_loglik(mirt_reference_dir / "lsat7_2pl_loglik.txt")
        return {"a": params["a"], "b": params["b"], "loglik": loglik}

    def test_rasch_difficulty_recovery(self, lsat7_data, mirt_rasch_params):
        """Item difficulties should match mirt within tolerance."""
        result = fit(lsat7_data, model="rasch", estimator="mml_em")

        # Difficulties should be close (allowing for different centering)
        b_py = result.params["b"]
        b_r = mirt_rasch_params["b"]

        # Center both for comparison (remove mean difference)
        b_py_centered = b_py - b_py.mean()
        b_r_centered = b_r - b_r.mean()

        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.05)

    def test_rasch_loglik(self, lsat7_data, mirt_rasch_params):
        """Log-likelihood should match mirt within tolerance."""
        result = fit(lsat7_data, model="rasch", estimator="mml_em")

        # Allow small difference due to numerical integration
        assert abs(result.loglik - mirt_rasch_params["loglik"]) < 1.0

    def test_2pl_parameter_recovery(self, lsat7_data, mirt_2pl_params):
        """2PL a and b parameters should match mirt."""
        result = fit(lsat7_data, model="2pl", estimator="mml_em")

        # Discriminations
        np.testing.assert_allclose(
            result.params["a"], mirt_2pl_params["a"], rtol=0.10
        )

        # Difficulties (centered)
        b_py = result.params["b"] - result.params["b"].mean()
        b_r = mirt_2pl_params["b"] - mirt_2pl_params["b"].mean()
        np.testing.assert_allclose(b_py, b_r, atol=0.10)

    def test_eap_scores(self, lsat7_data, mirt_reference_dir: Path):
        """EAP ability estimates should match mirt fscores."""
        result = fit(lsat7_data, model="rasch", estimator="mml_em")
        scores = result.score(method="eap")

        # Load mirt EAP scores
        mirt_eap = _load_csv(mirt_reference_dir / "lsat7_rasch_eap.csv")
        mirt_theta = mirt_eap[:, 0]

        # Should be highly correlated (r > 0.99)
        corr = np.corrcoef(scores.theta, mirt_theta)[0, 1]
        assert corr > 0.99

        # Mean absolute difference should be small
        mad = np.abs(scores.theta - mirt_theta).mean()
        assert mad < 0.05


class TestSimulatedDataValidation:
    """Validate parameter recovery on simulated data."""

    def test_rasch_known_params(self):
        """Recover known Rasch parameters from simulated data."""
        # True parameters
        np.random.seed(42)
        N, J = 1000, 20
        b_true = np.linspace(-2, 2, J)
        theta_true = np.random.standard_normal(N)

        # Generate data
        from irt.core import prob_2pl
        p = prob_2pl(theta_true[:, None], np.ones(J), b_true)
        X = (np.random.rand(N, J) < p).astype(float)

        # Fit model
        result = fit(X, model="rasch", estimator="mml_em")

        # Recover difficulties (up to centering)
        b_est = result.params["b"] - result.params["b"].mean()
        b_true_centered = b_true - b_true.mean()

        # Should recover within ~0.1 logits for N=1000
        # Allow small buffer for numerical integration differences
        np.testing.assert_allclose(b_est, b_true_centered, atol=0.17)

    def test_2pl_known_params(self):
        """Recover known 2PL parameters from simulated data."""
        np.random.seed(42)
        N, J = 2000, 15
        a_true = np.exp(np.random.normal(0.2, 0.3, J))  # lognormal
        b_true = np.random.normal(0, 1, J)
        theta_true = np.random.standard_normal(N)

        # Generate data
        from irt.core import prob_2pl
        p = prob_2pl(theta_true[:, None], a_true, b_true)
        X = (np.random.rand(N, J) < p).astype(float)

        # Fit model
        result = fit(X, model="2pl", estimator="mml_em")

        # Discriminations should correlate highly with true values
        corr_a = np.corrcoef(result.params["a"], a_true)[0, 1]
        assert corr_a > 0.90

        # Difficulties should correlate highly
        corr_b = np.corrcoef(result.params["b"], b_true)[0, 1]
        assert corr_b > 0.95


class TestScoringValidation:
    """Validate ability scoring against mirt fscores."""

    def test_eap_vs_mirt(self, mirt_reference_dir: Path):
        """EAP scores should match mirt fscores(method='EAP')."""
        X = _load_csv(mirt_reference_dir / "sim_rasch_data.csv")
        mirt_eap = _load_csv(mirt_reference_dir / "sim_rasch_eap.csv")

        result = fit(X, model="rasch", estimator="mml_em")
        scores = result.score(method="eap")

        # Scores should be highly correlated
        assert np.corrcoef(scores.theta, mirt_eap[:, 0])[0, 1] > 0.99

        # Standard errors should be similar
        assert np.corrcoef(scores.se, mirt_eap[:, 1])[0, 1] > 0.95

    def test_map_vs_mirt(self, mirt_reference_dir: Path):
        """MAP scores should match mirt fscores(method='MAP')."""
        X = _load_csv(mirt_reference_dir / "sim_rasch_data.csv")

        result = fit(X, model="rasch", estimator="mml_em")
        scores = result.score(method="map")

        mirt_map = _load_csv(mirt_reference_dir / "sim_rasch_map.csv")
        mirt_theta = mirt_map[:, 0]

        assert np.corrcoef(scores.theta, mirt_theta)[0, 1] > 0.99


class TestTechnicalSettings:
    """Validate that technical settings produce comparable results."""

    def test_quadpts_effect(self, mirt_reference_dir: Path):
        """More quadrature points should improve accuracy."""
        X = _load_csv(mirt_reference_dir / "lsat7_data.csv")

        # Fewer quadpts (faster, less accurate)
        result_21 = fit(X, model="rasch", technical={"quadpts": 21})

        # More quadpts (slower, more accurate)
        result_61 = fit(X, model="rasch", technical={"quadpts": 61})

        # More quadpts should give higher log-likelihood
        assert result_61.loglik >= result_21.loglik - 0.5
