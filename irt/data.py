"""
Data handling and validation for IRT models.

This module provides functions for:
- Converting various input formats to standardized numpy arrays
- Validating binary response data
- Filtering persons/items with insufficient data

All data processing functions preserve alignment between data,
observation masks, and name labels.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from .item_priors import prior_domain, validate_prior_params

if TYPE_CHECKING:
    import pandas as pd


def as_binary_matrix(
    X: np.ndarray | "pd.DataFrame",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """
    Convert input data to a standardized binary response matrix.

    Parameters
    ----------
    X : array-like or DataFrame
        Response matrix of shape (N, J) with values in {0, 1, NaN}.
        - If numpy array: converted to float64, names auto-generated
        - If DataFrame: column names become item_names, index becomes person_names

    Returns
    -------
    X_np : np.ndarray
        Response matrix as float64, shape (N, J). Missing values are NaN.
    mask_obs : np.ndarray
        Boolean array indicating observed (non-missing) responses, shape (N, J).
    item_names : list[str]
        Names for each item (column).
    person_names : list[str]
        Names for each person (row).

    Raises
    ------
    ValueError
        If X contains values other than 0, 1, or NaN.
        If X has fewer than 2 items or 2 persons.
        If X is empty.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 0, np.nan], [0, 1, 1]])
    >>> X_np, mask, items, persons = as_binary_matrix(X)
    >>> X_np
    array([[ 1.,  0., nan],
           [ 0.,  1.,  1.]])
    >>> mask
    array([[ True,  True, False],
           [ True,  True,  True]])
    >>> items
    ['item_0', 'item_1', 'item_2']
    """
    # Check for pandas DataFrame
    try:
        import pandas as pd
        is_dataframe = isinstance(X, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    if is_dataframe:
        # Extract names from DataFrame
        item_names = [str(c) for c in X.columns.tolist()]
        person_names = [str(i) for i in X.index.tolist()]
        X_np = X.values.astype(np.float64)
    else:
        # Convert to numpy array
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim != 2:
            raise ValueError(
                f"X must be a 2D array, got {X_np.ndim} dimensions."
            )
        n_persons, n_items = X_np.shape
        item_names = [f"item_{j}" for j in range(n_items)]
        person_names = [f"person_{i}" for i in range(n_persons)]

    # Validate dimensions
    if X_np.size == 0:
        raise ValueError("X is empty. Provide a non-empty response matrix.")

    n_persons, n_items = X_np.shape
    if n_items < 2:
        raise ValueError(
            f"X must have at least 2 items (columns), got {n_items}."
        )
    if n_persons < 2:
        raise ValueError(
            f"X must have at least 2 persons (rows), got {n_persons}."
        )

    # Compute observation mask (True where not NaN)
    mask_obs = ~np.isnan(X_np)

    # Validate values: must be 0, 1, or NaN
    observed_values = X_np[mask_obs]
    valid_values = np.isin(observed_values, [0.0, 1.0])
    if not np.all(valid_values):
        invalid_vals = np.unique(observed_values[~valid_values])
        raise ValueError(
            f"X contains invalid values. Expected 0, 1, or NaN, "
            f"but found: {invalid_vals[:5]}{'...' if len(invalid_vals) > 5 else ''}"
        )

    return X_np, mask_obs, item_names, person_names


def as_polytomous_matrix(
    X: np.ndarray | "pd.DataFrame",
    n_categories: int | list[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    """
    Convert input data to a standardized polytomous response matrix.

    Parameters
    ----------
    X : array-like or DataFrame
        Response matrix of shape (N, J) with integer values 0, 1, ..., m_j-1
        (or 1, 2, ..., m_j if 1-based). Missing values are NaN.
    n_categories : int, list[int], or None, optional
        Number of categories per item:
        - None: infer from data as max(observed) + 1 per item
        - int: same number for all items
        - list/array: per-item counts, length J

    Returns
    -------
    X_np : np.ndarray
        Response matrix as float64, shape (N, J). Missing values are NaN.
    mask_obs : np.ndarray
        Boolean array indicating observed responses, shape (N, J).
    item_names : list[str]
        Names for each item.
    person_names : list[str]
        Names for each person.
    n_categories : np.ndarray
        Number of categories per item, shape (J,).

    Raises
    ------
    ValueError
        If X contains values outside valid range for each item.
    """
    try:
        import pandas as pd
        is_dataframe = isinstance(X, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    if is_dataframe:
        item_names = [str(c) for c in X.columns.tolist()]
        person_names = [str(i) for i in X.index.tolist()]
        X_np = X.values.astype(np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X_np.ndim} dimensions.")
        n_persons, n_items = X_np.shape
        item_names = [f"item_{j}" for j in range(n_items)]
        person_names = [f"person_{i}" for i in range(n_persons)]

    if X_np.size == 0:
        raise ValueError("X is empty. Provide a non-empty response matrix.")

    n_persons, n_items = X_np.shape
    if n_items < 2:
        raise ValueError(f"X must have at least 2 items (columns), got {n_items}.")
    if n_persons < 2:
        raise ValueError(f"X must have at least 2 persons (rows), got {n_persons}.")

    mask_obs = ~np.isnan(X_np)

    # Infer or validate n_categories
    if n_categories is None:
        n_cat_per_item = np.zeros(n_items, dtype=np.int64)
        for j in range(n_items):
            obs_vals = X_np[mask_obs[:, j], j]
            if len(obs_vals) == 0:
                n_cat_per_item[j] = 2  # default
            else:
                max_val = int(np.nanmax(obs_vals))
                n_cat_per_item[j] = max_val + 1
    elif isinstance(n_categories, (int, np.integer)):
        n_cat_per_item = np.full(n_items, int(n_categories), dtype=np.int64)
    else:
        n_cat_per_item = np.asarray(n_categories, dtype=np.int64)
        if n_cat_per_item.shape != (n_items,):
            raise ValueError(
                f"n_categories must have length {n_items}, got {len(n_cat_per_item)}."
            )

    # Validate values: for each item j, observed values must be in [0, n_cat_per_item[j]-1]
    for j in range(n_items):
        obs_j = mask_obs[:, j]
        if not np.any(obs_j):
            continue
        vals = X_np[obs_j, j]
        valid = (vals >= 0) & (vals < n_cat_per_item[j]) & (vals == np.floor(vals))
        if not np.all(valid):
            invalid = np.unique(vals[~valid])
            raise ValueError(
                f"Item {j} contains invalid values. Expected 0..{n_cat_per_item[j]-1}, "
                f"found: {invalid[:5]}{'...' if len(invalid) > 5 else ''}"
            )

    return X_np, mask_obs, item_names, person_names, n_cat_per_item


def compute_item_category_proportions(
    X: np.ndarray,
    mask_obs: np.ndarray,
    n_categories: np.ndarray,
) -> np.ndarray:
    """
    Compute proportion of responses in each category for each item.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    n_categories : np.ndarray
        Number of categories per item, shape (J,).

    Returns
    -------
    np.ndarray
        Proportions, shape (J, max(n_categories)). Rows sum to 1.
        Unused categories for items with fewer categories are 0.
    """
    J = X.shape[1]
    max_cat = int(np.max(n_categories))
    props = np.zeros((J, max_cat), dtype=np.float64)

    for j in range(J):
        obs = mask_obs[:, j]
        n_obs = obs.sum()
        if n_obs == 0:
            props[j, :] = 1.0 / n_categories[j]  # uniform
            continue
        for c in range(n_categories[j]):
            count = np.sum((X[:, j] == c) & obs)
            props[j, c] = count / n_obs
        # Normalize to sum to 1 (in case of rounding)
        s = props[j, :n_categories[j]].sum()
        if s > 0:
            props[j, :n_categories[j]] /= s

    return props


def filter_people_min_items(
    X: np.ndarray,
    mask_obs: np.ndarray,
    person_names: list[str],
    min_items: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Filter out persons with fewer than min_items observed responses.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Boolean observation mask, shape (N, J).
    person_names : list[str]
        Names for each person.
    min_items : int, default=2
        Minimum number of observed items required to keep a person.

    Returns
    -------
    X_filtered : np.ndarray
        Filtered response matrix, shape (N', J) where N' <= N.
    mask_filtered : np.ndarray
        Filtered observation mask, shape (N', J).
    person_names_filtered : list[str]
        Filtered person names.
    keep_idx : np.ndarray
        Boolean array of shape (N,) indicating which persons were kept.

    Warns
    -----
    UserWarning
        If more than 10% of persons are filtered out.

    Examples
    --------
    >>> X = np.array([[1, np.nan, np.nan], [0, 1, 1], [np.nan, np.nan, np.nan]])
    >>> mask = ~np.isnan(X)
    >>> names = ['A', 'B', 'C']
    >>> X_f, mask_f, names_f, keep = filter_people_min_items(X, mask, names, min_items=2)
    >>> names_f
    ['B']
    """
    n_items_per_person = mask_obs.sum(axis=1)
    keep_idx = n_items_per_person >= min_items

    n_dropped = (~keep_idx).sum()
    n_total = len(keep_idx)

    if n_dropped > 0:
        drop_pct = 100 * n_dropped / n_total
        if drop_pct > 10:
            warnings.warn(
                f"Filtered {n_dropped} persons ({drop_pct:.1f}%) with fewer than "
                f"{min_items} observed items.",
                UserWarning,
                stacklevel=2,
            )

    X_filtered = X[keep_idx]
    mask_filtered = mask_obs[keep_idx]
    person_names_filtered = [person_names[i] for i in range(n_total) if keep_idx[i]]

    return X_filtered, mask_filtered, person_names_filtered, keep_idx


def filter_items_min_people(
    X: np.ndarray,
    mask_obs: np.ndarray,
    item_names: list[str],
    min_people: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Filter out items with fewer than min_people observed responses.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Boolean observation mask, shape (N, J).
    item_names : list[str]
        Names for each item.
    min_people : int, default=2
        Minimum number of observed persons required to keep an item.

    Returns
    -------
    X_filtered : np.ndarray
        Filtered response matrix, shape (N, J') where J' <= J.
    mask_filtered : np.ndarray
        Filtered observation mask, shape (N, J').
    item_names_filtered : list[str]
        Filtered item names.
    keep_idx : np.ndarray
        Boolean array of shape (J,) indicating which items were kept.

    Warns
    -----
    UserWarning
        If more than 10% of items are filtered out.

    Examples
    --------
    >>> X = np.array([[1, np.nan], [0, np.nan], [1, 1]])
    >>> mask = ~np.isnan(X)
    >>> names = ['Q1', 'Q2']
    >>> X_f, mask_f, names_f, keep = filter_items_min_people(X, mask, names, min_people=2)
    >>> names_f
    ['Q1']
    """
    n_people_per_item = mask_obs.sum(axis=0)
    keep_idx = n_people_per_item >= min_people

    n_dropped = (~keep_idx).sum()
    n_total = len(keep_idx)

    if n_dropped > 0:
        drop_pct = 100 * n_dropped / n_total
        if drop_pct > 10:
            warnings.warn(
                f"Filtered {n_dropped} items ({drop_pct:.1f}%) with fewer than "
                f"{min_people} observed persons.",
                UserWarning,
                stacklevel=2,
            )

    X_filtered = X[:, keep_idx]
    mask_filtered = mask_obs[:, keep_idx]
    item_names_filtered = [item_names[j] for j in range(n_total) if keep_idx[j]]

    return X_filtered, mask_filtered, item_names_filtered, keep_idx


def compute_item_pvalues(
    X: np.ndarray,
    mask_obs: np.ndarray,
) -> np.ndarray:
    """
    Compute item p-values (proportion correct) from response data.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Boolean observation mask, shape (N, J).

    Returns
    -------
    np.ndarray
        P-values for each item, shape (J,).

    Notes
    -----
    P-values are clipped to [0.01, 0.99] to avoid extreme initial
    difficulty estimates.

    Examples
    --------
    >>> X = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
    >>> mask = np.ones_like(X, dtype=bool)
    >>> compute_item_pvalues(X, mask)
    array([0.66666667, 0.33333333, 0.66666667])
    """
    # Sum correct responses per item
    X_masked = np.where(mask_obs, X, 0)
    n_correct = X_masked.sum(axis=0)
    n_obs = mask_obs.sum(axis=0)

    # Avoid division by zero
    n_obs_safe = np.maximum(n_obs, 1)
    p_values = n_correct / n_obs_safe

    # Clip to avoid extreme values
    return np.clip(p_values, 0.01, 0.99)


def compute_person_scores(
    X: np.ndarray,
    mask_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute raw scores (sum of correct) for each person.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Boolean observation mask, shape (N, J).

    Returns
    -------
    sum_score : np.ndarray
        Raw sum scores, shape (N,).
    max_score : np.ndarray
        Maximum possible scores (number of items attempted), shape (N,).

    Examples
    --------
    >>> X = np.array([[1, 0, np.nan], [1, 1, 1]])
    >>> mask = ~np.isnan(X)
    >>> scores, max_scores = compute_person_scores(X, mask)
    >>> scores
    array([1., 3.])
    >>> max_scores
    array([2, 3])
    """
    X_masked = np.where(mask_obs, X, 0)
    sum_score = X_masked.sum(axis=1)
    max_score = mask_obs.sum(axis=1)
    return sum_score, max_score


def identify_extreme_scores(
    X: np.ndarray,
    mask_obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify persons with extreme (perfect or zero) scores.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Boolean observation mask, shape (N, J).

    Returns
    -------
    is_perfect : np.ndarray
        Boolean array, True for persons with all correct responses.
    is_zero : np.ndarray
        Boolean array, True for persons with all incorrect responses.

    Notes
    -----
    Extreme scores are problematic for MLE estimation as the likelihood
    is maximized at +/- infinity. JMLE typically handles these with
    nudging or Bayesian priors.

    Examples
    --------
    >>> X = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]])
    >>> mask = np.ones_like(X, dtype=bool)
    >>> perfect, zero = identify_extreme_scores(X, mask)
    >>> perfect
    array([ True, False, False])
    >>> zero
    array([False,  True, False])
    """
    sum_score, max_score = compute_person_scores(X, mask_obs)
    is_perfect = (sum_score == max_score) & (max_score > 0)
    is_zero = (sum_score == 0) & (max_score > 0)
    return is_perfect, is_zero


def validate_start_params(
    start: dict | None,
    n_items: int,
    n_persons: int,
    model: str,
    estimator: str,
) -> dict | None:
    """
    Validate user-provided starting parameter values.

    Parameters
    ----------
    start : dict or None
        Starting values with keys "a", "b", and optionally "c" or "theta".
    n_items : int
        Number of items in the data.
    n_persons : int
        Number of persons in the data.
    model : str
        Model type ("rasch", "2pl", or "3pl").
    estimator : str
        Estimator type ("mml_em" or "jmle").

    Returns
    -------
    dict or None
        Validated starting parameters, or None if not provided.

    Raises
    ------
    ValueError
        If start parameters have incorrect shapes or invalid values.
    """
    if start is None:
        return None

    validated = {}

    if "b" in start:
        b = np.asarray(start["b"], dtype=np.float64)
        if b.shape != (n_items,):
            raise ValueError(
                f"start['b'] must have shape ({n_items},), got {b.shape}"
            )
        if not np.all(np.isfinite(b)):
            raise ValueError("start['b'] contains non-finite values")
        validated["b"] = b

    if "a" in start:
        a = np.asarray(start["a"], dtype=np.float64)
        if a.shape != (n_items,):
            raise ValueError(
                f"start['a'] must have shape ({n_items},), got {a.shape}"
            )
        if not np.all(np.isfinite(a)):
            raise ValueError("start['a'] contains non-finite values")
        if np.any(a <= 0):
            raise ValueError("start['a'] must contain positive values")
        if model == "rasch" and not np.allclose(a, 1.0):
            warnings.warn(
                "Ignoring start['a'] for Rasch model. Discrimination fixed at 1.",
                UserWarning,
                stacklevel=2,
            )
        else:
            validated["a"] = a

    if "c" in start:
        c = np.asarray(start["c"], dtype=np.float64)
        if c.shape != (n_items,):
            raise ValueError(
                f"start['c'] must have shape ({n_items},), got {c.shape}"
            )
        if not np.all(np.isfinite(c)):
            raise ValueError("start['c'] contains non-finite values")
        if np.any((c < 0.0) | (c >= 1.0)):
            raise ValueError("start['c'] must be in [0.0, 1.0)")
        if model != "3pl":
            warnings.warn(
                "Ignoring start['c'] for non-3PL models.",
                UserWarning,
                stacklevel=2,
            )
        else:
            validated["c"] = c

    if "theta" in start:
        if estimator != "jmle":
            warnings.warn(
                "start['theta'] is only used for JMLE estimation. Ignoring.",
                UserWarning,
                stacklevel=2,
            )
        else:
            theta = np.asarray(start["theta"], dtype=np.float64)
            if theta.shape != (n_persons,):
                raise ValueError(
                    f"start['theta'] must have shape ({n_persons},), got {theta.shape}"
                )
            if not np.all(np.isfinite(theta)):
                raise ValueError("start['theta'] contains non-finite values")
            validated["theta"] = theta

    return validated if validated else None


def validate_fixed_params(
    fixed: dict | None,
    n_items: int,
    start: dict | None,
) -> dict | None:
    """
    Validate user-provided fixed parameter masks.

    Parameters
    ----------
    fixed : dict or None
        Boolean arrays indicating which parameters to hold fixed.
    n_items : int
        Number of items in the data.
    start : dict or None
        Starting values (required for fixed parameters).

    Returns
    -------
    dict or None
        Validated fixed parameter masks.

    Raises
    ------
    ValueError
        If fixed masks have incorrect shapes or fixed params lack start values.
    """
    if fixed is None:
        return None

    validated = {}

    for key in ("a", "b", "c"):
        if key in fixed:
            mask = np.asarray(fixed[key], dtype=bool)
            if mask.shape != (n_items,):
                raise ValueError(
                    f"fixed['{key}'] must have shape ({n_items},), got {mask.shape}"
                )
            if np.any(mask):
                # Check that start values exist for fixed parameters
                if start is None or key not in start:
                    raise ValueError(
                        f"Fixed parameters in fixed['{key}'] require corresponding "
                        f"values in start['{key}']"
                    )
            validated[key] = mask

    return validated if validated else None


def parse_item_priors(
    priors: dict | "pd.DataFrame" | None,
    item_names: list[str],
    model: str,
    a_bounds: tuple[float, float] | None = None,
    b_bounds: tuple[float, float] | None = None,
    c_bounds: tuple[float, float] | None = None,
) -> dict[str, list[dict[str, Any] | None]]:
    """
    Parse and validate item parameter priors.

    Parameters
    ----------
    priors : dict, DataFrame, or None
        Prior specifications for item parameters. Supported formats:
        - dict: keyed by "a" and/or "b", with per-item or grouped specs
        - DataFrame: columns ["item", "param", "dist", ...] or ["item", "param", "kind", ...]
    item_names : list[str]
        Item identifiers to map names to indices.
    model : str
        Model name ("rasch", "2pl", or "3pl").
    a_bounds : tuple or None
        Bounds for discrimination (used for domain checks).
    b_bounds : tuple or None
        Bounds for difficulty (used for domain checks).

    Returns
    -------
    dict
        {"a": list|None, "b": list|None, "c": list|None} with length J lists
        containing prior specs or None for each item.
    """
    n_items = len(item_names)
    priors_by_param: dict[str, list[dict[str, Any] | None]] = {
        "a": [None] * n_items,
        "b": [None] * n_items,
        "c": [None] * n_items,
    }

    if priors is None:
        return priors_by_param

    # Check for pandas DataFrame
    try:
        import pandas as pd
        is_dataframe = isinstance(priors, pd.DataFrame)
    except ImportError:
        is_dataframe = False

    if is_dataframe:
        _apply_priors_from_dataframe(
            priors,
            item_names,
            priors_by_param,
        )
    elif isinstance(priors, dict):
        _apply_priors_from_dict(
            priors,
            item_names,
            priors_by_param,
        )
    else:
        raise ValueError(
            "priors must be a dict, pandas DataFrame, or None."
        )

    if model == "rasch" and any(priors_by_param["a"]):
        warnings.warn(
            "Ignoring item priors for discrimination ('a') in Rasch model.",
            UserWarning,
            stacklevel=2,
        )
        priors_by_param["a"] = [None] * n_items

    if model != "3pl" and any(priors_by_param["c"]):
        warnings.warn(
            "Ignoring item priors for guessing ('c') in non-3PL model.",
            UserWarning,
            stacklevel=2,
        )
        priors_by_param["c"] = [None] * n_items

    _validate_priors_against_bounds(priors_by_param["a"], a_bounds, "a")
    _validate_priors_against_bounds(priors_by_param["b"], b_bounds, "b")
    _validate_priors_against_bounds(priors_by_param["c"], c_bounds, "c")

    return priors_by_param


def _apply_priors_from_dict(
    priors: dict[str, Any],
    item_names: list[str],
    priors_by_param: dict[str, list[dict[str, Any] | None]],
) -> None:
    for param in ("a", "b", "c"):
        if param not in priors:
            continue
        param_spec = priors[param]
        _apply_param_spec(param, param_spec, item_names, priors_by_param)


def _apply_priors_from_dataframe(
    df: "pd.DataFrame",
    item_names: list[str],
    priors_by_param: dict[str, list[dict[str, Any] | None]],
) -> None:
    required = {"item", "param"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"priors DataFrame missing columns: {missing}")

    has_kind = "kind" in df.columns
    has_dist = "dist" in df.columns
    if not (has_kind or has_dist):
        raise ValueError("priors DataFrame must include 'dist' or 'kind' column.")

    for row in df.to_dict(orient="records"):
        param = str(row["param"]).lower()
        if param not in ("a", "b", "c"):
            raise ValueError(f"Invalid param '{param}'. Use 'a', 'b', or 'c'.")
        item_idx = _item_to_index(row["item"], item_names)
        kind = str(row.get("dist", row.get("kind"))).lower()
        params = _extract_params(kind, row)
        prior = {"kind": kind, "params": validate_prior_params(kind, params)}
        _set_item_prior(priors_by_param, param, item_idx, prior)


def _apply_param_spec(
    param: str,
    spec: Any,
    item_names: list[str],
    priors_by_param: dict[str, list[dict[str, Any] | None]],
) -> None:
    if spec is None:
        return
    if isinstance(spec, list):
        for entry in spec:
            _apply_param_spec(param, entry, item_names, priors_by_param)
        return

    if isinstance(spec, dict):
        has_dist = "dist" in spec or "kind" in spec
        has_items = "items" in spec or "item" in spec
        if has_dist or has_items:
            items = spec.get("items", spec.get("item", None))
            item_indices = _normalize_items(items, item_names)
            kind_value = spec.get("dist", spec.get("kind"))
            if kind_value is None:
                raise ValueError(f"Prior spec for '{param}' missing 'dist' or 'kind'.")
            kind = str(kind_value).lower()
            params = _extract_params(kind, spec)
            prior = {"kind": kind, "params": validate_prior_params(kind, params)}
            for idx in item_indices:
                _set_item_prior(priors_by_param, param, idx, prior)
            return

        # Treat dict as item -> prior spec mapping
        for item_key, entry in spec.items():
            item_idx = _item_to_index(item_key, item_names)
            _apply_param_spec(param, {**entry, "item": item_idx}, item_names, priors_by_param)
        return

    raise ValueError(
        f"Invalid prior specification for '{param}'. "
        "Expected dict or list of dicts."
    )


def _extract_params(kind: str, entry: dict[str, Any]) -> dict[str, Any]:
    kind = kind.lower()
    params = dict(entry.get("params", {}))

    def take(keys: list[str]) -> Any | None:
        for key in keys:
            if key in params:
                return params[key]
            if key in entry:
                return entry[key]
        return None

    if kind in ("normal", "lognormal"):
        mean = take(["mean", "mu", "p1"])
        sd = take(["sd", "sigma", "p2"])
        if mean is None or sd is None:
            raise ValueError(f"{kind} prior requires mean and sd.")
        return {"mean": mean, "sd": sd}

    if kind == "gamma":
        shape = take(["shape", "alpha", "p1"])
        scale = take(["scale", "beta", "p2"])
        if shape is None or scale is None:
            raise ValueError("gamma prior requires shape and scale.")
        return {"shape": shape, "scale": scale}

    if kind == "beta":
        alpha = take(["alpha", "p1"])
        beta = take(["beta", "p2"])
        if alpha is None or beta is None:
            raise ValueError("beta prior requires alpha and beta.")
        return {"alpha": alpha, "beta": beta}

    raise ValueError(f"Unknown prior kind '{kind}'.")


def _normalize_items(
    items: Any | None,
    item_names: list[str],
) -> list[int]:
    if items is None:
        return list(range(len(item_names)))
    if isinstance(items, (list, tuple, np.ndarray)):
        return [_item_to_index(item, item_names) for item in items]
    return [_item_to_index(items, item_names)]


def _item_to_index(item: Any, item_names: list[str]) -> int:
    name_to_idx = {name: idx for idx, name in enumerate(item_names)}
    if isinstance(item, (int, np.integer)):
        idx = int(item)
        if idx < 0 or idx >= len(item_names):
            raise ValueError(f"Item index {idx} out of range.")
        return idx
    item_str = str(item)
    if item_str in name_to_idx:
        return name_to_idx[item_str]
    try:
        idx = int(item_str)
        if idx < 0 or idx >= len(item_names):
            raise ValueError(f"Item index {idx} out of range.")
        return idx
    except ValueError as exc:
        raise ValueError(f"Unknown item '{item}'.") from exc


def _set_item_prior(
    priors_by_param: dict[str, list[dict[str, Any] | None]],
    param: str,
    item_idx: int,
    prior: dict[str, Any],
) -> None:
    existing = priors_by_param[param][item_idx]
    if existing is not None:
        raise ValueError(
            f"Duplicate prior specified for param '{param}' at item index {item_idx}."
        )
    priors_by_param[param][item_idx] = prior


def _validate_priors_against_bounds(
    priors: list[dict[str, Any] | None],
    bounds: tuple[float, float] | None,
    param: str,
) -> None:
    if bounds is None or not any(priors):
        return
    lo, hi = bounds
    for prior in priors:
        if prior is None:
            continue
        kind = prior["kind"]
        dom_lo, dom_hi, dom_lo_open, dom_hi_open = prior_domain(kind)
        if dom_lo is not None:
            if (lo <= dom_lo and dom_lo_open) or (lo < dom_lo and not dom_lo_open):
                raise ValueError(
                    f"{kind} prior for '{param}' requires bounds above {dom_lo}."
                )
        if dom_hi is not None:
            if (hi >= dom_hi and dom_hi_open) or (hi > dom_hi and not dom_hi_open):
                raise ValueError(
                    f"{kind} prior for '{param}' requires bounds below {dom_hi}."
                )
