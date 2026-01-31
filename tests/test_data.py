"""
Tests for data handling functions.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose

from irt.data import (
    as_binary_matrix,
    filter_people_min_items,
    filter_items_min_people,
    compute_item_pvalues,
    compute_person_scores,
    identify_extreme_scores,
)


class TestAsBinaryMatrix:
    """Tests for as_binary_matrix function."""

    def test_numpy_array(self):
        """Converts numpy array correctly."""
        X = np.array([[1, 0, 1], [0, 1, 0]])
        X_np, mask, items, persons = as_binary_matrix(X)
        
        assert X_np.dtype == np.float64
        assert X_np.shape == (2, 3)
        assert_array_equal(X_np, X)
        assert_array_equal(mask, np.ones((2, 3), dtype=bool))
        assert items == ['item_0', 'item_1', 'item_2']
        assert persons == ['person_0', 'person_1']

    def test_handles_nan(self):
        """NaN values are preserved and masked correctly."""
        X = np.array([[1, np.nan, 0], [0, 1, np.nan]])
        X_np, mask, _, _ = as_binary_matrix(X)
        
        assert np.isnan(X_np[0, 1])
        assert np.isnan(X_np[1, 2])
        assert mask[0, 0] == True
        assert mask[0, 1] == False
        assert mask[1, 2] == False

    def test_invalid_values_raises(self):
        """Invalid values raise ValueError."""
        X = np.array([[1, 0, 2], [0, 1, 0]])  # 2 is invalid
        with pytest.raises(ValueError, match="invalid values"):
            as_binary_matrix(X)

    def test_too_few_items_raises(self):
        """Fewer than 2 items raises ValueError."""
        X = np.array([[1], [0]])
        with pytest.raises(ValueError, match="at least 2 items"):
            as_binary_matrix(X)

    def test_too_few_persons_raises(self):
        """Fewer than 2 persons raises ValueError."""
        X = np.array([[1, 0, 1]])
        with pytest.raises(ValueError, match="at least 2 persons"):
            as_binary_matrix(X)

    def test_empty_raises(self):
        """Empty array raises ValueError."""
        X = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError, match="empty"):
            as_binary_matrix(X)


class TestFilterPeopleMinItems:
    """Tests for filter_people_min_items function."""

    def test_filters_correctly(self):
        """Persons with too few items are filtered."""
        X = np.array([
            [1, np.nan, np.nan],  # 1 item - filter
            [0, 1, 1],            # 3 items - keep
            [np.nan, np.nan, 1],  # 1 item - filter
        ])
        mask = ~np.isnan(X)
        names = ['A', 'B', 'C']
        
        X_f, mask_f, names_f, keep = filter_people_min_items(
            X, mask, names, min_items=2
        )
        
        assert X_f.shape == (1, 3)
        assert names_f == ['B']
        assert_array_equal(keep, [False, True, False])

    def test_preserves_alignment(self):
        """Filtering preserves alignment between X, mask, and names."""
        X = np.array([
            [1, 0],    # keep
            [np.nan, np.nan],  # filter (0 items)
            [1, 1],    # keep
        ])
        mask = ~np.isnan(X)
        names = ['first', 'second', 'third']
        
        X_f, mask_f, names_f, _ = filter_people_min_items(
            X, mask, names, min_items=1
        )
        
        assert names_f == ['first', 'third']
        assert X_f.shape[0] == 2
        assert mask_f.shape[0] == 2


class TestFilterItemsMinPeople:
    """Tests for filter_items_min_people function."""

    def test_filters_correctly(self):
        """Items with too few people are filtered."""
        X = np.array([
            [1, np.nan],
            [0, np.nan],
            [1, 1],
        ])
        mask = ~np.isnan(X)
        names = ['Q1', 'Q2']
        
        X_f, mask_f, names_f, keep = filter_items_min_people(
            X, mask, names, min_people=2
        )
        
        assert X_f.shape == (3, 1)
        assert names_f == ['Q1']
        assert_array_equal(keep, [True, False])


class TestComputeItemPvalues:
    """Tests for compute_item_pvalues function."""

    def test_basic_computation(self):
        """P-values are computed correctly."""
        X = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ])
        mask = np.ones_like(X, dtype=bool)
        
        pvals = compute_item_pvalues(X, mask)
        expected = np.array([2/3, 1/3, 2/3])
        assert_allclose(pvals, expected)

    def test_with_missing(self):
        """P-values ignore missing data."""
        X = np.array([
            [1, np.nan],
            [1, 1],
            [0, 0],
        ])
        mask = ~np.isnan(X)
        
        pvals = compute_item_pvalues(X, mask)
        assert_allclose(pvals[0], 2/3)
        assert_allclose(pvals[1], 1/2)

    def test_clips_extreme(self):
        """P-values are clipped to [0.01, 0.99]."""
        X = np.array([
            [1, 0],
            [1, 0],
            [1, 0],
        ])
        mask = np.ones_like(X, dtype=bool)
        
        pvals = compute_item_pvalues(X, mask)
        assert pvals[0] == 0.99  # All correct clipped
        assert pvals[1] == 0.01  # All wrong clipped


class TestComputePersonScores:
    """Tests for compute_person_scores function."""

    def test_basic_computation(self):
        """Scores are computed correctly."""
        X = np.array([
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ])
        mask = np.ones_like(X, dtype=bool)
        
        scores, max_scores = compute_person_scores(X, mask)
        assert_array_equal(scores, [3, 1, 0])
        assert_array_equal(max_scores, [3, 3, 3])

    def test_with_missing(self):
        """Scores account for missing data."""
        X = np.array([
            [1, np.nan, 1],
            [0, 1, np.nan],
        ])
        mask = ~np.isnan(X)
        
        scores, max_scores = compute_person_scores(X, mask)
        assert_allclose(scores, [2, 1])
        assert_array_equal(max_scores, [2, 2])


class TestIdentifyExtremeScores:
    """Tests for identify_extreme_scores function."""

    def test_identifies_perfect(self):
        """Perfect scores (all correct) are identified."""
        X = np.array([
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ])
        mask = np.ones_like(X, dtype=bool)
        
        perfect, zero = identify_extreme_scores(X, mask)
        assert_array_equal(perfect, [True, False, False])
        assert_array_equal(zero, [False, False, True])

    def test_with_missing(self):
        """Extreme scores with missing data are identified."""
        X = np.array([
            [1, 1, np.nan],  # Perfect on observed
            [0, np.nan, 0],  # Zero on observed
        ])
        mask = ~np.isnan(X)
        
        perfect, zero = identify_extreme_scores(X, mask)
        assert perfect[0] == True
        assert zero[1] == True
