"""Integration tests for polytomous plotting."""

import numpy as np
import pytest

from irt import fit
from irt.plotting import plot_ccc


class TestPolytomousPlotting:
    def test_plot_ccc_does_not_raise(self):
        np.random.seed(42)
        X = np.random.randint(0, 4, size=(50, 5)).astype(float)
        result = fit(X, model="pcm")
        fig, ax = plot_ccc(
            result.params, "pcm", 0, 4,
            theta_range=(-3, 3)
        )
        assert fig is not None
        assert ax is not None

    def test_plot_icc_polytomous(self):
        np.random.seed(42)
        X = np.random.randint(0, 4, size=(80, 6)).astype(float)
        result = fit(X, model="gpcm")
        fig, ax = result.plot_icc(items=[0])
        assert fig is not None
        assert ax is not None
