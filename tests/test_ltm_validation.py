"""
Cross-validation tests against R ltm package.

These tests compare Python IRT estimates to pre-computed R ltm results.
Reference values generated using ltm v1.2-0.

Run R validation script first: Rscript tests/generate_ltm_reference.R
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


class TestLSATValidationLtm:
    """Validate against ltm results on LSAT dataset."""

    @pytest.fixture
    def lsat_data(self, ltm_reference_dir: Path):
        """Load LSAT data (N=1000, J=5)."""
        return _load_csv(ltm_reference_dir / "lsat_data.csv")

    @pytest.fixture
    def ltm_rasch_params(self, ltm_reference_dir: Path):
        """Reference Rasch parameters from ltm (discrimination fixed at 1)."""
        params = _load_params(ltm_reference_dir / "lsat_rasch_fixed_params.csv")
        loglik = _load_loglik(ltm_reference_dir / "lsat_rasch_fixed_loglik.txt")
        return {
            "b": params["b"],
            "loglik": loglik,
        }

    @pytest.fixture
    def ltm_2pl_params(self, ltm_reference_dir: Path):
        """Reference 2PL parameters from ltm."""
        params = _load_params(ltm_reference_dir / "lsat_2pl_params.csv")
        loglik = _load_loglik(ltm_reference_dir / "lsat_2pl_loglik.txt")
        return {
            "a": params["a"],
            "b": params["b"],
            "loglik": loglik,
        }

    def test_rasch_difficulty_vs_ltm(self, lsat_data, ltm_rasch_params):
        """Item difficulties should match ltm rasch() within tolerance."""
        result = fit(lsat_data, model="rasch", estimator="mml_em")

        # Center both for comparison (ltm may use different centering)
        b_py = result.params["b"]
        b_r = ltm_rasch_params["b"]

        b_py_centered = b_py - b_py.mean()
        b_r_centered = b_r - b_r.mean()

        # ltm uses 21 quadrature points vs our 61, so allow slightly more tolerance
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.08)

    def test_rasch_loglik_vs_ltm(self, lsat_data, ltm_rasch_params):
        """Log-likelihood should be similar to ltm (within numerical tolerance)."""
        result = fit(lsat_data, model="rasch", estimator="mml_em")

        # Different quadrature settings may cause larger differences
        assert abs(result.loglik - ltm_rasch_params["loglik"]) < 2.0

    def test_2pl_parameters_vs_ltm(self, lsat_data, ltm_2pl_params):
        """2PL a and b parameters should match ltm ltm()."""
        result = fit(lsat_data, model="2pl", estimator="mml_em")

        # Discriminations should correlate highly
        corr_a = np.corrcoef(result.params["a"], ltm_2pl_params["a"])[0, 1]
        assert corr_a > 0.95

        # Difficulties (centered comparison)
        b_py = result.params["b"] - result.params["b"].mean()
        b_r = ltm_2pl_params["b"] - ltm_2pl_params["b"].mean()

        np.testing.assert_allclose(b_py, b_r, atol=0.15)

    def test_eap_scores_vs_ltm(self, lsat_data, ltm_reference_dir: Path):
        """EAP ability estimates should match ltm factor.scores(method='EAP')."""
        result = fit(lsat_data, model="rasch", estimator="mml_em")
        scores = result.score(method="eap")

        # Load ltm EAP scores
        ltm_eap = _load_csv(ltm_reference_dir / "lsat_rasch_fixed_eap.csv")
        ltm_theta = ltm_eap[:, 0]

        # Scores should be highly correlated (r > 0.98)
        corr = np.corrcoef(scores.theta, ltm_theta)[0, 1]
        assert corr > 0.98

        # Mean absolute difference should be reasonable
        mad = np.abs(scores.theta - ltm_theta).mean()
        assert mad < 0.10

    def test_map_scores_vs_ltm(self, lsat_data, ltm_reference_dir: Path):
        """MAP ability estimates should match ltm factor.scores(method='EB')."""
        result = fit(lsat_data, model="rasch", estimator="mml_em")
        scores = result.score(method="map")

        # Load ltm EB (Empirical Bayes = MAP) scores
        ltm_eb = _load_csv(ltm_reference_dir / "lsat_rasch_fixed_eb.csv")
        ltm_theta = ltm_eb[:, 0]

        # MAP scores should correlate highly with ltm EB scores
        corr = np.corrcoef(scores.theta, ltm_theta)[0, 1]
        assert corr > 0.97


class TestQuadratureComparison:
    """Compare effects of different quadrature settings matching ltm."""

    def test_ltm_quadrature_settings(self, ltm_reference_dir: Path):
        """Test with ltm-like quadrature settings (21 points)."""
        lsat_data = _load_csv(ltm_reference_dir / "lsat_data.csv")
        ltm_params = _load_params(ltm_reference_dir / "lsat_rasch_fixed_params.csv")

        # Fit with ltm-like settings: 21 quadrature points
        result = fit(
            lsat_data,
            model="rasch",
            estimator="mml_em",
            technical={"quadpts": 21},
        )

        b_py = result.params["b"] - result.params["b"].mean()
        b_r = ltm_params["b"] - ltm_params["b"].mean()

        # With matching quadrature, should be closer
        np.testing.assert_allclose(b_py, b_r, atol=0.05)
