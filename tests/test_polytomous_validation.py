"""
Cross-validation tests for polytomous IRT models against R mirt.

These tests compare Python polytomous estimates to pre-computed R mirt results.
Reference files are pre-committed; tests run without R.
To regenerate: make refs (requires R with mirt).

Science dataset: 4 items, 4 categories. R uses 1-4; we convert to 0-3.
"""

from pathlib import Path

import numpy as np
import pytest

from irt import fit


def _load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def _load_science_data(path: Path) -> np.ndarray:
    """Load Science data and convert from R 1-based (1,2,3,4) to Python 0-based (0,1,2,3)."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return (data - 1).astype(float)  # 1,2,3,4 -> 0,1,2,3


def _load_loglik(path: Path) -> float:
    return float(Path(path).read_text().strip())


def _science_ref_available(ref_dir: Path) -> bool:
    """Check if polytomous reference files exist."""
    return (ref_dir / "science_data.csv").exists()


def _ltm_science_available() -> bool:
    return _science_ref_available(Path(__file__).parent / "ltm_reference")


def _mirt_science_available() -> bool:
    return _science_ref_available(Path(__file__).parent / "mirt_reference")


# ---------------------------------------------------------------------------
# ltm validation
# ---------------------------------------------------------------------------


class TestScienceValidationLtm:
    """Validate polytomous models against ltm on Science dataset."""

    @pytest.fixture
    def science_data_ltm(self, ltm_reference_dir: Path) -> np.ndarray:
        """Load Science data (0-based)."""
        return _load_science_data(ltm_reference_dir / "science_data.csv")

    @pytest.fixture
    def ltm_grm_ref(self, ltm_reference_dir: Path) -> dict:
        """Load ltm GRM reference."""
        data = np.genfromtxt(
            ltm_reference_dir / "science_grm_params.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        a = np.array(data["a"], dtype=float)
        b = np.column_stack([data["b1"], data["b2"], data["b3"]])
        loglik = _load_loglik(ltm_reference_dir / "science_grm_loglik.txt")
        return {"a": a, "b": b, "loglik": loglik}

    @pytest.fixture
    def ltm_gpcm_ref(self, ltm_reference_dir: Path) -> dict:
        """Load ltm GPCM reference."""
        data = np.genfromtxt(
            ltm_reference_dir / "science_gpcm_params.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        a = np.array(data["a"], dtype=float)
        b = np.column_stack([data["b1"], data["b2"], data["b3"]])
        loglik = _load_loglik(ltm_reference_dir / "science_gpcm_loglik.txt")
        return {"a": a, "b": b, "loglik": loglik}

    @pytest.fixture
    def ltm_pcm_ref(self, ltm_reference_dir: Path) -> dict:
        """Load ltm PCM reference."""
        data = np.genfromtxt(
            ltm_reference_dir / "science_pcm_params.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        b = np.column_stack([data["b1"], data["b2"], data["b3"]])
        loglik = _load_loglik(ltm_reference_dir / "science_pcm_loglik.txt")
        return {"b": b, "loglik": loglik}

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_grm_params_vs_ltm(self, science_data_ltm, ltm_grm_ref, ltm_reference_dir):
        """GRM parameters should match ltm within tolerance."""
        result = fit(science_data_ltm, model="grm", estimator="mml_em")

        a_py = result.params["a"]
        b_py = result.params["b"]
        a_r = ltm_grm_ref["a"]
        b_r = ltm_grm_ref["b"]

        # Discrimination: allow correlation or relative tolerance
        np.testing.assert_allclose(a_py, a_r, rtol=0.15)

        # Boundaries: center for identification, then compare
        b_py_centered = b_py - np.nanmean(b_py)
        b_r_centered = b_r - np.nanmean(b_r)
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.15)

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_grm_loglik_vs_ltm(self, science_data_ltm, ltm_grm_ref):
        """GRM log-likelihood should match ltm within tolerance."""
        result = fit(science_data_ltm, model="grm", estimator="mml_em")
        assert abs(result.loglik - ltm_grm_ref["loglik"]) < 2.0

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_gpcm_params_vs_ltm(self, science_data_ltm, ltm_gpcm_ref):
        """GPCM parameters should match ltm within tolerance."""
        result = fit(science_data_ltm, model="gpcm", estimator="mml_em")

        a_py = result.params["a"]
        b_py = result.params["b"]
        a_r = ltm_gpcm_ref["a"]
        b_r = ltm_gpcm_ref["b"]

        np.testing.assert_allclose(a_py, a_r, rtol=0.15)
        b_py_centered = b_py - np.nanmean(b_py)
        b_r_centered = b_r - np.nanmean(b_r)
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.15)

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_gpcm_loglik_vs_ltm(self, science_data_ltm, ltm_gpcm_ref):
        """GPCM log-likelihood should match ltm within tolerance."""
        result = fit(science_data_ltm, model="gpcm", estimator="mml_em")
        assert abs(result.loglik - ltm_gpcm_ref["loglik"]) < 2.0

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_pcm_params_vs_ltm(self, science_data_ltm, ltm_pcm_ref):
        """PCM step parameters should match ltm within tolerance."""
        result = fit(science_data_ltm, model="pcm", estimator="mml_em")

        b_py = result.params["b"]
        b_r = ltm_pcm_ref["b"]

        b_py_centered = b_py - np.nanmean(b_py)
        b_r_centered = b_r - np.nanmean(b_r)
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.15)

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_pcm_loglik_vs_ltm(self, science_data_ltm, ltm_pcm_ref):
        """PCM log-likelihood should match ltm within tolerance."""
        result = fit(science_data_ltm, model="pcm", estimator="mml_em")
        assert abs(result.loglik - ltm_pcm_ref["loglik"]) < 2.0

    @pytest.mark.skipif(
        not _ltm_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_grm_eap_vs_ltm(self, science_data_ltm, ltm_reference_dir):
        """EAP scores should correlate highly with ltm GRM EAP."""
        result = fit(science_data_ltm, model="grm", estimator="mml_em")
        scores = result.score(method="eap")

        ltm_eap = _load_csv(ltm_reference_dir / "science_grm_eap.csv")
        ltm_theta = ltm_eap[:, 0]

        corr = np.corrcoef(scores.theta, ltm_theta)[0, 1]
        assert corr > 0.98


# ---------------------------------------------------------------------------
# mirt validation
# ---------------------------------------------------------------------------


class TestScienceValidationMirt:
    """Validate polytomous models against mirt on Science dataset."""

    @pytest.fixture
    def science_data_mirt(self, mirt_reference_dir: Path) -> np.ndarray:
        """Load Science data (0-based)."""
        return _load_science_data(mirt_reference_dir / "science_data.csv")

    @pytest.fixture
    def mirt_grm_ref(self, mirt_reference_dir: Path) -> dict:
        """Load mirt GRM reference."""
        data = np.genfromtxt(
            mirt_reference_dir / "science_grm_params.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        a = np.array(data["a"], dtype=float)
        b = np.column_stack([data["b1"], data["b2"], data["b3"]])
        loglik = _load_loglik(mirt_reference_dir / "science_grm_loglik.txt")
        return {"a": a, "b": b, "loglik": loglik}

    @pytest.fixture
    def mirt_gpcm_ref(self, mirt_reference_dir: Path) -> dict:
        """Load mirt GPCM reference."""
        data = np.genfromtxt(
            mirt_reference_dir / "science_gpcm_params.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        a = np.array(data["a"], dtype=float)
        b = np.column_stack([data["b1"], data["b2"], data["b3"]])
        loglik = _load_loglik(mirt_reference_dir / "science_gpcm_loglik.txt")
        return {"a": a, "b": b, "loglik": loglik}

    @pytest.fixture
    def mirt_pcm_ref(self, mirt_reference_dir: Path) -> dict:
        """Load mirt PCM reference."""
        data = np.genfromtxt(
            mirt_reference_dir / "science_pcm_params.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        b = np.column_stack([data["b1"], data["b2"], data["b3"]])
        loglik = _load_loglik(mirt_reference_dir / "science_pcm_loglik.txt")
        return {"b": b, "loglik": loglik}

    @pytest.mark.skipif(
        not _mirt_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_grm_params_vs_mirt(self, science_data_mirt, mirt_grm_ref):
        """GRM parameters should match mirt within tolerance."""
        result = fit(science_data_mirt, model="grm", estimator="mml_em")

        a_py = result.params["a"]
        b_py = result.params["b"]
        a_r = mirt_grm_ref["a"]
        b_r = mirt_grm_ref["b"]

        np.testing.assert_allclose(a_py, a_r, rtol=0.15)
        b_py_centered = b_py - np.nanmean(b_py)
        b_r_centered = b_r - np.nanmean(b_r)
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.15)

    @pytest.mark.skipif(
        not _mirt_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_grm_loglik_vs_mirt(self, science_data_mirt, mirt_grm_ref):
        """GRM log-likelihood should match mirt within tolerance."""
        result = fit(science_data_mirt, model="grm", estimator="mml_em")
        assert abs(result.loglik - mirt_grm_ref["loglik"]) < 2.0

    @pytest.mark.skipif(
        not _mirt_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_gpcm_params_vs_mirt(self, science_data_mirt, mirt_gpcm_ref):
        """GPCM parameters should match mirt within tolerance."""
        result = fit(science_data_mirt, model="gpcm", estimator="mml_em")

        a_py = result.params["a"]
        b_py = result.params["b"]
        a_r = mirt_gpcm_ref["a"]
        b_r = mirt_gpcm_ref["b"]

        np.testing.assert_allclose(a_py, a_r, rtol=0.15)
        b_py_centered = b_py - np.nanmean(b_py)
        b_r_centered = b_r - np.nanmean(b_r)
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.15)

    @pytest.mark.skipif(
        not _mirt_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_gpcm_loglik_vs_mirt(self, science_data_mirt, mirt_gpcm_ref):
        """GPCM log-likelihood should match mirt within tolerance."""
        result = fit(science_data_mirt, model="gpcm", estimator="mml_em")
        assert abs(result.loglik - mirt_gpcm_ref["loglik"]) < 2.0

    @pytest.mark.skipif(
        not _mirt_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_pcm_params_vs_mirt(self, science_data_mirt, mirt_pcm_ref):
        """PCM step parameters should match mirt within tolerance."""
        result = fit(science_data_mirt, model="pcm", estimator="mml_em")

        b_py = result.params["b"]
        b_r = mirt_pcm_ref["b"]

        b_py_centered = b_py - np.nanmean(b_py)
        b_r_centered = b_r - np.nanmean(b_r)
        np.testing.assert_allclose(b_py_centered, b_r_centered, atol=0.15)

    @pytest.mark.skipif(
        not _mirt_science_available(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_pcm_loglik_vs_mirt(self, science_data_mirt, mirt_pcm_ref):
        """PCM log-likelihood should match mirt within tolerance."""
        result = fit(science_data_mirt, model="pcm", estimator="mml_em")
        assert abs(result.loglik - mirt_pcm_ref["loglik"]) < 2.0

    @pytest.mark.skipif(
        not (Path(__file__).parent / "mirt_reference" / "science_rsm_params.csv").exists(),
        reason="Run Rscript tests/generate_polytomous_reference.R first",
    )
    def test_rsm_converges_and_loglik_reasonable(self, science_data_mirt, mirt_reference_dir):
        """RSM should converge and loglik should be in same ballpark as mirt."""
        result = fit(science_data_mirt, model="rsm", estimator="mml_em")
        assert result.converged
        mirt_loglik = _load_loglik(mirt_reference_dir / "science_rsm_loglik.txt")
        # Allow larger tolerance for RSM (different parameterizations)
        assert abs(result.loglik - mirt_loglik) < 5.0
