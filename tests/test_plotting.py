"""
Tests for plotting functions.

Note: These tests verify that plotting functions run without error
and return the expected types. Visual correctness is not verified.
"""

import numpy as np
import pytest


# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")


def generate_test_data(n_persons=100, n_items=10, seed=42):
    """Generate simulated IRT data for testing."""
    np.random.seed(seed)
    from scipy.special import expit

    theta = np.random.randn(n_persons)
    a = np.random.rand(n_items) + 0.5
    b = np.random.randn(n_items)

    p = expit(a[None, :] * (theta[:, None] - b[None, :]))
    X = (np.random.rand(n_persons, n_items) < p).astype(float)
    mask = np.ones_like(X, dtype=bool)

    return X, mask, theta, a, b


class TestPlotICC:
    """Tests for plot_icc function."""

    def test_single_item(self):
        """Plots single item ICC."""
        from irt.plotting import plot_icc

        fig, ax = plot_icc(a=1.0, b=0.5)
        assert fig is not None
        assert ax is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_multiple_items(self):
        """Plots multiple item ICCs."""
        from irt.plotting import plot_icc

        fig, ax = plot_icc(
            a=[1.0, 1.5, 2.0],
            b=[-1, 0, 1],
            item_labels=["Easy", "Medium", "Hard"],
        )
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_custom_range(self):
        """Plots with custom theta range."""
        from irt.plotting import plot_icc

        fig, ax = plot_icc(a=1.0, b=0.0, theta_range=(-6, 6))
        xlim = ax.get_xlim()
        assert xlim[0] <= -6
        assert xlim[1] >= 6
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotICCEmpirical:
    """Tests for plot_icc_with_empirical function."""

    def test_basic_plot(self):
        """Plots ICC with empirical points."""
        from irt.plotting import plot_icc_with_empirical

        X, mask, theta, a, b = generate_test_data()
        fig, ax = plot_icc_with_empirical(X, mask, theta, a, b, item_idx=0)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotAllICCs:
    """Tests for plot_all_iccs function."""

    def test_grid_layout(self):
        """Plots all ICCs in grid."""
        from irt.plotting import plot_all_iccs

        a = np.array([1.0, 1.5, 2.0, 0.8])
        b = np.array([-1, 0, 1, 0.5])

        fig, axes = plot_all_iccs(a, b)
        assert fig is not None
        assert len(axes) >= len(a)
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotIIF:
    """Tests for plot_iif function."""

    def test_single_item(self):
        """Plots single item information function."""
        from irt.plotting import plot_iif

        fig, ax = plot_iif(a=1.5, b=0.0)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_multiple_items(self):
        """Plots multiple IIFs."""
        from irt.plotting import plot_iif

        fig, ax = plot_iif(a=[0.5, 1.0, 2.0], b=[0, 0, 0])
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotTIF:
    """Tests for plot_tif function."""

    def test_basic_plot(self):
        """Plots test information function."""
        from irt.plotting import plot_tif

        a = np.array([1.0, 1.5, 2.0, 0.8, 1.2])
        b = np.array([-1, -0.5, 0, 0.5, 1])

        fig, ax = plot_tif(a, b)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_with_se(self):
        """Plots TIF with SE curve."""
        from irt.plotting import plot_tif

        a = np.array([1.0, 1.5, 2.0])
        b = np.array([-1, 0, 1])

        fig, ax = plot_tif(a, b, show_se=True)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotWrightMap:
    """Tests for plot_wright_map function."""

    def test_basic_plot(self):
        """Plots Wright map."""
        from irt.plotting import plot_wright_map

        theta = np.random.randn(100)
        b = np.random.randn(20)

        fig, axes = plot_wright_map(theta, b)
        assert fig is not None
        assert len(axes) == 2
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotItemFit:
    """Tests for plot_item_fit function."""

    def test_basic_plot(self):
        """Plots item fit scatter."""
        from irt.plotting import plot_item_fit

        infit_ms = np.random.rand(20) * 0.5 + 0.75
        outfit_ms = np.random.rand(20) * 0.5 + 0.75
        b = np.random.randn(20)

        fig, ax = plot_item_fit(infit_ms, outfit_ms, b)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotItemFitBars:
    """Tests for plot_item_fit_bars function."""

    def test_basic_plot(self):
        """Plots item fit bar charts."""
        from irt.plotting import plot_item_fit_bars

        infit_ms = np.random.rand(10) * 0.5 + 0.75
        outfit_ms = np.random.rand(10) * 0.5 + 0.75

        fig, axes = plot_item_fit_bars(infit_ms, outfit_ms)
        assert fig is not None
        assert len(axes) == 2
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotPersonFit:
    """Tests for plot_person_fit function."""

    def test_basic_plot(self):
        """Plots person fit by ability."""
        from irt.plotting import plot_person_fit

        infit_ms = np.random.rand(100) * 0.5 + 0.75
        outfit_ms = np.random.rand(100) * 0.5 + 0.75
        theta = np.random.randn(100)

        fig, ax = plot_person_fit(infit_ms, outfit_ms, theta)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotAbilityDistribution:
    """Tests for plot_ability_distribution function."""

    def test_basic_plot(self):
        """Plots ability distribution."""
        from irt.plotting import plot_ability_distribution

        theta = np.random.randn(100)
        se = np.abs(np.random.randn(100) * 0.2 + 0.4)

        fig, ax = plot_ability_distribution(theta, se)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotSEByTheta:
    """Tests for plot_se_by_theta function."""

    def test_basic_plot(self):
        """Plots SE by ability."""
        from irt.plotting import plot_se_by_theta

        theta = np.random.randn(100)
        se = np.abs(np.random.randn(100) * 0.2 + 0.4)

        fig, ax = plot_se_by_theta(theta, se)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_with_theoretical(self):
        """Plots SE with theoretical curve."""
        from irt.plotting import plot_se_by_theta

        theta = np.random.randn(100)
        se = np.abs(np.random.randn(100) * 0.2 + 0.4)
        a = np.ones(10)
        b = np.random.randn(10)

        fig, ax = plot_se_by_theta(theta, se, a=a, b=b)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestPlotResiduals:
    """Tests for plot_residuals_by_theta function."""

    def test_basic_plot(self):
        """Plots residuals by ability."""
        from irt.plotting import plot_residuals_by_theta

        X, mask, theta, a, b = generate_test_data()
        fig, ax = plot_residuals_by_theta(X, mask, theta, a, b)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_single_item(self):
        """Plots residuals for single item."""
        from irt.plotting import plot_residuals_by_theta

        X, mask, theta, a, b = generate_test_data()
        fig, ax = plot_residuals_by_theta(X, mask, theta, a, b, item_idx=0)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestFitResultPlotMethods:
    """Tests for FitResult plotting methods."""

    @pytest.fixture
    def fit_result(self):
        """Create a fitted model for testing."""
        from irt import fit
        X, mask, theta, a, b = generate_test_data()
        return fit(X, model="rasch")

    def test_plot_icc(self, fit_result):
        """FitResult.plot_icc works."""
        fig, ax = fit_result.plot_icc(items=[0, 1, 2])
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_plot_icc_empirical(self, fit_result):
        """FitResult.plot_icc_empirical works."""
        fig, ax = fit_result.plot_icc_empirical(item_idx=0)
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_plot_tif(self, fit_result):
        """FitResult.plot_tif works."""
        fig, ax = fit_result.plot_tif()
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_plot_wright_map(self, fit_result):
        """FitResult.plot_wright_map works."""
        fig, axes = fit_result.plot_wright_map()
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_plot_item_fit(self, fit_result):
        """FitResult.plot_item_fit works."""
        fig, ax = fit_result.plot_item_fit()
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)

    def test_plot_ability_distribution(self, fit_result):
        """FitResult.plot_ability_distribution works."""
        fig, ax = fit_result.plot_ability_distribution()
        assert fig is not None
        plt = pytest.importorskip("matplotlib.pyplot")
        plt.close(fig)


class TestFitResultDiagnosticMethods:
    """Tests for FitResult diagnostic methods."""

    @pytest.fixture
    def fit_result(self):
        """Create a fitted model for testing."""
        from irt import fit
        X, mask, theta, a, b = generate_test_data()
        return fit(X, model="rasch")

    def test_item_fit(self, fit_result):
        """FitResult.item_fit returns DataFrame."""
        pytest.importorskip("pandas")
        df = fit_result.item_fit()
        assert "infit_ms" in df.columns
        assert len(df) == fit_result.X.shape[1]

    def test_person_fit(self, fit_result):
        """FitResult.person_fit returns DataFrame."""
        pytest.importorskip("pandas")
        df = fit_result.person_fit()
        assert "infit_ms" in df.columns
        assert len(df) == fit_result.X.shape[0]

    def test_model_fit(self, fit_result):
        """FitResult.model_fit returns dict."""
        summary = fit_result.model_fit()
        assert isinstance(summary, dict)
        assert "reliability" in summary
        assert "n_persons" in summary
