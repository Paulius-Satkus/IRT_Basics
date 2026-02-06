"""
Public API for the IRT library.

This module provides the main entry point for fitting IRT models:
- fit(): Fits binary (Rasch, 2PL, 3PL) or polytomous (PCM, RSM, GRM, GPCM, NRM)
  IRT models using MML-EM or JMLE estimation.

Example
-------
>>> import numpy as np
>>> from irt import fit
>>>
>>> # Binary: Rasch model
>>> X = np.random.binomial(1, 0.7, size=(100, 20)).astype(float)
>>> result = fit(X, model="rasch", estimator="mml_em")
>>>
>>> # Polytomous: Partial Credit Model
>>> X_poly = np.random.randint(0, 5, size=(100, 10)).astype(float)
>>> result_poly = fit(X_poly, model="pcm")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import warnings
import numpy as np

from .types import FitResult, ScoreResult, MML_EM_DEFAULTS, JMLE_DEFAULTS

if TYPE_CHECKING:
    import pandas as pd


def fit(
    X: np.ndarray | "pd.DataFrame",
    model: str = "rasch",
    estimator: str = "mml_em",
    technical: dict[str, Any] | None = None,
    start: dict[str, np.ndarray] | None = None,
    fixed: dict[str, np.ndarray] | None = None,
    priors: dict[str, Any] | "pd.DataFrame" | None = None,
    constraints: dict[str, Any] | None = None,
    n_categories: int | list[int] | np.ndarray | None = None,
) -> FitResult:
    """
    Fit an IRT model to response data.

    Parameters
    ----------
    X : array-like or DataFrame
        Response matrix of shape (N, J) where N is the number of persons
        and J is the number of items. For binary models values must be
        in {0, 1, NaN}; for polytomous models values must be integers
        in {0, 1, ..., m_j-1, NaN}. NaN indicates missing responses.

    model : {"rasch", "2pl", "3pl", "pcm", "rsm", "grm", "gpcm", "nrm"}, default="rasch"
        IRT model to fit:
        Binary: rasch, 2pl, 3pl
        Polytomous: pcm, rsm, grm, gpcm, nrm

    estimator : {"mml_em", "jmle"}, default="mml_em"
        Estimation method:
        - "mml_em": Marginal Maximum Likelihood via EM algorithm
        - "jmle": Joint Maximum Likelihood Estimation (Rasch only)

    technical : dict, optional
        Technical parameters to override defaults. Keys depend on estimator:

        For MML-EM:
            - quadpts (int): Number of quadrature points (default: 61)
            - theta_lo (float): Lower bound of theta grid (default: -4.0)
            - theta_hi (float): Upper bound of theta grid (default: 4.0)
            - prior (tuple): Prior specification as (kind, mean, sd)
              (default: ("normal", 0.0, 1.0))
            - max_iter (int): Maximum EM iterations (default: 200)
            - tol (float): Convergence tolerance (default: 1e-4)
            - a_bounds (tuple): Bounds for discrimination (default: (0.25, 4.0))
            - b_bounds (tuple): Bounds for difficulty (default: (-6.0, 6.0))
            - c_bounds (tuple): Bounds for guessing (default: (1e-6, 0.35))
            - mstep_max_iter (int): Max iterations for M-step (default: 25)

        For JMLE:
            - max_iter (int): Maximum iterations (default: 50)
            - tol (float): Convergence tolerance (default: 1e-4)
            - jmle_method (str): Update method, "newton" (default: "newton")
            - nudge (float): Nudge for extreme scores (default: 0.3)
            - theta_bounds (tuple): Bounds for theta (default: (-6.0, 6.0))
            - b_bounds (tuple): Bounds for difficulty (default: (-6.0, 6.0))

    start : dict, optional
        Starting values for parameters:
        - "a": np.ndarray of shape (J,) - initial discrimination values
        - "b": np.ndarray of shape (J,) - initial difficulty values
        - "c": np.ndarray of shape (J,) - initial guessing values (3PL only)
        - "theta": np.ndarray of shape (N,) - initial theta values (JMLE only)
        If not provided, reasonable defaults are computed from the data.

    fixed : dict, optional
        Boolean arrays indicating which parameters to hold fixed:
        - "a": np.ndarray of shape (J,) - True = fixed, False = estimated
        - "b": np.ndarray of shape (J,) - True = fixed, False = estimated
        - "c": np.ndarray of shape (J,) - True = fixed, False = estimated (3PL only)
        Fixed parameters must have starting values provided in `start`.

    priors : dict or DataFrame, optional
        Item parameter priors. Supports per-item specifications for "a", "b",
        and "c" (3PL only). For 3PL, a gamma prior is used for "c" by default
        unless a custom prior is provided.
        - dict: {"a": {...}, "b": {...}, "c": {...}} with optional item selectors
        - DataFrame: columns ["item", "param", "dist", ...] or ["item", "param", "kind", ...]
        See MANUAL.md for examples.

    constraints : dict, optional
        Constraints on estimation:
        - "center_b" (bool): Center difficulties to have mean 0 (default: True
          for JMLE, False for MML-EM which uses prior for identification)

    n_categories : int, list[int], or None, optional
        For polytomous models: number of categories per item. None = infer from data.

    Returns
    -------
    FitResult
        Object containing estimated parameters, convergence information,
        and methods for scoring and diagnostics.

    Raises
    ------
    ValueError
        If model, estimator, or data are invalid.

    Notes
    -----
    Model parameterization:
        P(X=1|theta) = 1 / (1 + exp(-a * (theta - b)))

    Identification:
        - MML-EM: Identified via the prior on theta (typically N(0,1))
        - JMLE: Identified by centering item difficulties (mean(b) = 0)

    Examples
    --------
    >>> import numpy as np
    >>> from irt import fit
    >>>
    >>> # Create response data (100 persons, 20 items)
    >>> np.random.seed(42)
    >>> X = np.random.binomial(1, 0.6, size=(100, 20)).astype(float)
    >>>
    >>> # Fit Rasch model with MML-EM
    >>> result = fit(X, model="rasch", estimator="mml_em")
    >>> print(f"Converged: {result.converged}")
    >>> print(f"Item difficulties: {result.params['b'][:5]}")
    >>>
    >>> # Score persons
    >>> scores = result.score(method="eap")
    >>> print(f"Ability estimates: {scores.theta[:5]}")
    >>>
    >>> # Fit 2PL model
    >>> result_2pl = fit(X, model="2pl")
    >>> print(f"Discriminations: {result_2pl.params['a'][:5]}")
    """
    # Import here to avoid circular imports
    from .data import (
        as_binary_matrix,
        as_polytomous_matrix,
        filter_people_min_items,
    )
    from .estimators.mml_em import fit_mml_em
    from .estimators.mml_em_poly import fit_mml_em_poly
    from .estimators.jmle import fit_jmle

    POLYTOMOUS_MODELS = ("pcm", "rsm", "grm", "gpcm", "nrm")
    BINARY_MODELS = ("rasch", "2pl", "3pl")

    # Validate model parameter
    model = model.lower()
    if model not in BINARY_MODELS + POLYTOMOUS_MODELS:
        raise ValueError(
            f"Invalid model '{model}'. Must be one of "
            f"{BINARY_MODELS + POLYTOMOUS_MODELS}."
        )

    # Validate estimator parameter
    estimator = estimator.lower()
    if estimator not in ("mml_em", "jmle"):
        raise ValueError(
            f"Invalid estimator '{estimator}'. Must be 'mml_em' or 'jmle'."
        )

    # JMLE only supports Rasch
    if estimator == "jmle" and model in ("2pl", "3pl"):
        raise ValueError(
            "JMLE estimator only supports Rasch model. "
            "Use estimator='mml_em' for 2PL/3PL models."
        )

    # Polytomous models only support MML-EM
    if model in POLYTOMOUS_MODELS and estimator == "jmle":
        raise ValueError(
            f"Polytomous model '{model}' only supports estimator='mml_em'."
        )

    is_polytomous = model in POLYTOMOUS_MODELS

    if is_polytomous:
        X_np, mask_obs, item_names, person_names, n_cat = as_polytomous_matrix(
            X, n_categories=n_categories
        )
    else:
        X_np, mask_obs, item_names, person_names = as_binary_matrix(X)

    # Filter persons with too few responses
    min_items = 2
    X_np, mask_obs, person_names, _ = filter_people_min_items(
        X_np, mask_obs, person_names, min_items=min_items
    )

    if X_np.shape[0] == 0:
        raise ValueError(
            f"No persons remaining after filtering. All persons had fewer than "
            f"{min_items} observed responses."
        )

    # Warn if any items are very sparse
    min_people = 2
    n_people_per_item = mask_obs.sum(axis=0)
    n_sparse_items = int(np.sum(n_people_per_item < min_people))
    if n_sparse_items > 0:
        warnings.warn(
            f"{n_sparse_items} item(s) have fewer than {min_people} observed "
            "responses. Consider filtering items with "
            "irt.data.filter_items_min_people() before fitting.",
            UserWarning,
            stacklevel=2,
        )

    # Merge technical parameters with defaults
    if estimator == "mml_em":
        tech = {**MML_EM_DEFAULTS}
    else:
        tech = {**JMLE_DEFAULTS}

    if technical is not None:
        # Validate technical keys
        valid_keys = set(tech.keys())
        invalid_keys = set(technical.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid technical parameters: {invalid_keys}. "
                f"Valid keys for {estimator}: {valid_keys}"
            )
        tech.update(technical)

    # Set default constraints
    if constraints is None:
        constraints = {}

    # Dispatch to appropriate estimator
    if is_polytomous:
        result = fit_mml_em_poly(
            X=X_np,
            mask_obs=mask_obs,
            n_categories=n_cat,
            model=model,
            technical=tech,
            start=start,
            item_names=item_names,
            person_names=person_names,
        )
    elif estimator == "mml_em":
        result = fit_mml_em(
            X=X_np,
            mask_obs=mask_obs,
            model=model,
            technical=tech,
            start=start,
            fixed=fixed,
            priors=priors,
            constraints=constraints,
            item_names=item_names,
            person_names=person_names,
        )
    else:  # jmle
        result = fit_jmle(
            X=X_np,
            mask_obs=mask_obs,
            technical=tech,
            start=start,
            fixed=fixed,
            priors=priors,
            constraints=constraints,
            item_names=item_names,
            person_names=person_names,
        )

    return result


__all__ = ["fit", "FitResult", "ScoreResult"]
