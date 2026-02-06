"""Unit tests for polytomous data handling."""

import numpy as np
import pytest

from irt.data import as_polytomous_matrix, compute_item_category_proportions


class TestAsPolytomousMatrix:
    def test_infers_categories(self):
        X = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=float)
        X_np, mask, items, persons, n_cat = as_polytomous_matrix(X)
        np.testing.assert_array_equal(n_cat, [3, 3, 3])
        assert X_np.shape == (3, 3)

    def test_explicit_categories(self):
        X = np.array([[0, 1], [1, 0]], dtype=float)
        _, _, _, _, n_cat = as_polytomous_matrix(X, n_categories=4)
        np.testing.assert_array_equal(n_cat, [4, 4])

    def test_rejects_invalid(self):
        X = np.array([[0, 1, 5], [1, 0, 2]], dtype=float)
        with pytest.raises(ValueError, match="invalid"):
            as_polytomous_matrix(X, n_categories=4)

    def test_dataframe(self):
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame([[0, 1], [1, 0]], columns=["A", "B"])
        X_np, _, items, persons, n_cat = as_polytomous_matrix(X)
        assert items == ["A", "B"]
        assert len(persons) == 2

    def test_too_few_items_raises(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            as_polytomous_matrix(np.array([[0], [1]]))  # 1 item, 2 persons


class TestComputeItemCategoryProportions:
    def test_correct_proportions(self):
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        mask = np.ones_like(X, dtype=bool)
        n_cat = np.array([3, 3])
        props = compute_item_category_proportions(X, mask, n_cat)
        np.testing.assert_allclose(props[:, 0], [1/3, 1/3])
        np.testing.assert_allclose(props[:, 1], [1/3, 1/3])
        np.testing.assert_allclose(props[:, 2], [1/3, 1/3])
