"""
Tests for diagnostic functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from irt.diagnostics import (
    compute_residuals,
    infit_outfit_items,
    infit_outfit_persons,
    infit_outfit,
    point_biserial_items,
    item_discrimination_empirical,
    reliability_marginal,
    separation_index,
    item_chi_square,
    compute_aic_bic,
    likelihood_ratio_test,
    model_fit_summary,
    item_fit_table,
    q3_residual_correlation,
    q3_summary,
    srmsr_statistic,
    m2_baseline_statistic,
    cfi_tli_from_m2,
    m2_statistic,
)


def generate_test_data(n_persons=100, n_items=20, seed=42):
    """Generate simulated IRT data for testing."""
    np.random.seed(seed)
    from scipy.special import expit

    theta = np.random.randn(n_persons)
    a = np.ones(n_items)  # Rasch
    b = np.random.randn(n_items)

    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    X = (np.random.rand(n_persons, n_items) < p).astype(float)
    mask = np.ones_like(X, dtype=bool)

    return X, mask, theta, a, b


class TestComputeResiduals:
    """Tests for compute_residuals function."""

    def test_output_shapes(self):
        """Output arrays have correct shapes."""
        X, mask, theta, a, b = generate_test_data()
        E, residual, std_residual = compute_residuals(X, mask, theta, a, b)

        N, J = X.shape
        assert E.shape == (N, J)
        assert residual.shape == (N, J)
        assert std_residual.shape == (N, J)

    def test_expected_in_range(self):
        """Expected values are between 0 and 1."""
        X, mask, theta, a, b = generate_test_data()
        E, _, _ = compute_residuals(X, mask, theta, a, b)

        assert np.all(E >= 0)
        assert np.all(E <= 1)

    def test_residual_bounded(self):
        """Raw residuals are between -1 and 1."""
        X, mask, theta, a, b = generate_test_data()
        _, residual, _ = compute_residuals(X, mask, theta, a, b)

        assert np.all(residual >= -1)
        assert np.all(residual <= 1)

    def test_mean_residual_near_zero(self):
        """Mean residual across persons is near zero for well-fitting model."""
        X, mask, theta, a, b = generate_test_data(n_persons=500)
        _, residual, _ = compute_residuals(X, mask, theta, a, b)

        mean_residual = residual.mean(axis=0)
        assert np.all(np.abs(mean_residual) < 0.15)  # Allow some sampling error


class TestInfitOutfit:
    """Tests for infit/outfit functions."""

    def test_item_fit_shapes(self):
        """Item fit statistics have correct shapes."""
        X, mask, theta, a, b = generate_test_data()
        infit_ms, infit_z, outfit_ms, outfit_z = infit_outfit_items(
            X, mask, theta, a, b
        )

        J = X.shape[1]
        assert infit_ms.shape == (J,)
        assert infit_z.shape == (J,)
        assert outfit_ms.shape == (J,)
        assert outfit_z.shape == (J,)

    def test_person_fit_shapes(self):
        """Person fit statistics have correct shapes."""
        X, mask, theta, a, b = generate_test_data()
        infit_ms, infit_z, outfit_ms, outfit_z = infit_outfit_persons(
            X, mask, theta, a, b
        )

        N = X.shape[0]
        assert infit_ms.shape == (N,)
        assert outfit_ms.shape == (N,)

    def test_fit_positive(self):
        """Mean-square fit statistics are positive."""
        X, mask, theta, a, b = generate_test_data()

        infit_ms, _, outfit_ms, _ = infit_outfit_items(X, mask, theta, a, b)
        assert np.all(infit_ms > 0)
        assert np.all(outfit_ms > 0)

        infit_ms_p, _, outfit_ms_p, _ = infit_outfit_persons(X, mask, theta, a, b)
        assert np.all(infit_ms_p > 0)
        assert np.all(outfit_ms_p > 0)

    def test_good_fit_near_one(self):
        """For well-fitting data, mean-squares are near 1.0."""
        X, mask, theta, a, b = generate_test_data(n_persons=500, n_items=30)
        infit_ms, _, outfit_ms, _ = infit_outfit_items(X, mask, theta, a, b)

        # Most items should have fit near 1
        assert np.median(infit_ms) > 0.7
        assert np.median(infit_ms) < 1.3
        assert np.median(outfit_ms) > 0.7
        assert np.median(outfit_ms) < 1.3

    def test_dataframe_output(self):
        """infit_outfit returns DataFrames."""
        pytest.importorskip("pandas")
        X, mask, theta, a, b = generate_test_data()

        item_fit, person_fit = infit_outfit(X, mask, theta, a, b)

        assert "infit_ms" in item_fit.columns
        assert "outfit_ms" in item_fit.columns
        assert len(item_fit) == X.shape[1]
        assert len(person_fit) == X.shape[0]


class TestPointBiserial:
    """Tests for point_biserial_items function."""

    def test_shape(self):
        """Output has correct shape."""
        X, mask, theta, a, b = generate_test_data()
        rpb = point_biserial_items(X, mask, theta)

        assert rpb.shape == (X.shape[1],)

    def test_positive_for_good_items(self):
        """Point-biserial is positive for well-discriminating items."""
        X, mask, theta, a, b = generate_test_data(n_persons=500)
        rpb = point_biserial_items(X, mask, theta)

        # Most items should have positive rpb
        assert (rpb > 0).sum() > len(rpb) * 0.8

    def test_bounded(self):
        """Point-biserial is between -1 and 1."""
        X, mask, theta, a, b = generate_test_data()
        rpb = point_biserial_items(X, mask, theta)

        valid = ~np.isnan(rpb)
        assert np.all(rpb[valid] >= -1)
        assert np.all(rpb[valid] <= 1)


class TestItemDiscrimination:
    """Tests for item_discrimination_empirical function."""

    def test_shape(self):
        """Output has correct shape."""
        X, mask, theta, a, b = generate_test_data()
        disc = item_discrimination_empirical(X, mask, theta)

        assert disc.shape == (X.shape[1],)

    def test_positive(self):
        """Discrimination is positive for good items."""
        X, mask, theta, a, b = generate_test_data(n_persons=300)
        disc = item_discrimination_empirical(X, mask, theta)

        valid = ~np.isnan(disc)
        # Most items should have positive discrimination
        assert (disc[valid] > 0).sum() > valid.sum() * 0.7


class TestReliability:
    """Tests for reliability functions."""

    def test_reliability_bounded(self):
        """Reliability is between 0 and 1."""
        X, mask, theta, a, b = generate_test_data()
        # Generate realistic SEs
        se = np.abs(np.random.randn(len(theta)) * 0.3 + 0.5)

        rel = reliability_marginal(theta, se)
        assert 0 <= rel <= 1

    def test_separation_positive(self):
        """Separation index is non-negative."""
        X, mask, theta, a, b = generate_test_data()
        se = np.abs(np.random.randn(len(theta)) * 0.3 + 0.5)

        sep, strata = separation_index(theta, se)
        assert sep >= 0
        assert strata >= 0


class TestChiSquare:
    """Tests for chi-square item fit."""

    def test_shapes(self):
        """Chi-square outputs have correct shapes."""
        X, mask, theta, a, b = generate_test_data()
        chi_sq, df, p_values = item_chi_square(X, mask, theta, a, b)

        J = X.shape[1]
        assert chi_sq.shape == (J,)
        assert df.shape == (J,)
        assert p_values.shape == (J,)

    def test_p_values_bounded(self):
        """P-values are between 0 and 1."""
        X, mask, theta, a, b = generate_test_data()
        chi_sq, df, p_values = item_chi_square(X, mask, theta, a, b)

        valid = ~np.isnan(p_values)
        assert np.all(p_values[valid] >= 0)
        assert np.all(p_values[valid] <= 1)


class TestInformationCriteria:
    """Tests for AIC/BIC computation."""

    def test_aic_bic_ordering(self):
        """BIC penalizes more than AIC for large samples."""
        loglik = -1000
        n_persons = 500
        n_items = 20

        aic, bic = compute_aic_bic(loglik, n_persons, n_items, "rasch")

        # BIC should be larger for large n
        assert bic > aic

    def test_2pl_more_params(self):
        """2PL has higher penalty than Rasch."""
        loglik = -1000
        n_persons = 100
        n_items = 20

        aic_rasch, bic_rasch = compute_aic_bic(loglik, n_persons, n_items, "rasch")
        aic_2pl, bic_2pl = compute_aic_bic(loglik, n_persons, n_items, "2pl")

        # 2PL has more parameters, so higher IC
        assert aic_2pl > aic_rasch


class TestResidualDiagnostics:
    """Tests for residual-based diagnostics (Q3, SRMSR)."""

    def test_q3_shape_and_diag(self):
        """Q3 matrix has correct shape and NaN diagonal."""
        X, mask, theta, a, b = generate_test_data(n_persons=200, n_items=10)
        q3 = q3_residual_correlation(X, mask, theta, a, b)

        assert q3.shape == (X.shape[1], X.shape[1])
        assert np.all(np.isnan(np.diag(q3)))

    def test_q3_summary(self):
        """Q3 summary returns expected keys."""
        X, mask, theta, a, b = generate_test_data(n_persons=200, n_items=8)
        q3 = q3_residual_correlation(X, mask, theta, a, b)
        summary = q3_summary(q3, threshold=0.2)

        assert "q3_mean" in summary
        assert "q3_max_abs" in summary
        assert "q3_n_pairs" in summary
        assert "q3_n_flagged" in summary

    def test_srmsr_non_negative(self):
        """SRMSR should be non-negative when computable."""
        X, mask, theta, a, b = generate_test_data(n_persons=300, n_items=12)
        srmsr = srmsr_statistic(X, mask, theta, a, b)

        assert np.isnan(srmsr) or srmsr >= 0


class TestFitIndices:
    """Tests for CFI/TLI and model fit summary outputs."""

    def test_cfi_tli_bounds(self):
        """CFI/TLI should be within [0, 1] when defined."""
        X, mask, theta, a, b = generate_test_data(n_persons=400, n_items=10)
        m2, df, _, _ = m2_statistic(X, mask, theta, a, b)
        m2_null, df_null = m2_baseline_statistic(X, mask)
        cfi, tli = cfi_tli_from_m2(m2, df, m2_null, df_null)

        if not np.isnan(cfi):
            assert 0 <= cfi <= 1
        if not np.isnan(tli):
            assert tli <= 1

    def test_model_fit_summary_includes_new_stats(self):
        """model_fit_summary includes new fit and residual stats."""
        X, mask, theta, a, b = generate_test_data(n_persons=400, n_items=10)
        se = np.abs(np.random.randn(len(theta)) * 0.3 + 0.5)

        summary = model_fit_summary(
            X=X,
            mask_obs=mask,
            theta=theta,
            se=se,
            a=a,
            b=b,
            c=None,
            loglik=-1000.0,
            model="rasch",
        )

        for key in ("cfi", "tli", "srmsr", "q3_mean", "q3_max_abs"):
            assert key in summary


class TestLikelihoodRatioTest:
    """Tests for likelihood ratio test."""

    def test_positive_chi_square(self):
        """Chi-square is non-negative."""
        chi_sq, p = likelihood_ratio_test(-1000, -1100, df=10)
        assert chi_sq >= 0

    def test_p_value_bounded(self):
        """P-value is between 0 and 1."""
        chi_sq, p = likelihood_ratio_test(-1000, -1050, df=5)
        assert 0 <= p <= 1


class TestModelFitSummary:
    """Tests for model_fit_summary function."""

    def test_returns_dict(self):
        """Returns dictionary with expected keys."""
        X, mask, theta, a, b = generate_test_data()
        se = np.abs(np.random.randn(len(theta)) * 0.3 + 0.5)

        summary = model_fit_summary(X, mask, theta, se, a, b, None, -1000, "rasch")

        assert isinstance(summary, dict)
        assert "n_persons" in summary
        assert "n_items" in summary
        assert "reliability" in summary
        assert "separation_index" in summary


class TestItemFitTable:
    """Tests for item_fit_table function."""

    def test_returns_dataframe(self):
        """Returns DataFrame with expected columns."""
        pytest.importorskip("pandas")
        X, mask, theta, a, b = generate_test_data()

        df = item_fit_table(X, mask, theta, a, b)

        assert "item" in df.columns
        assert "a" in df.columns
        assert "b" in df.columns
        assert "infit_ms" in df.columns
        assert "outfit_ms" in df.columns
        assert "rpb" in df.columns
        assert len(df) == X.shape[1]
