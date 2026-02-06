"""
Type definitions for IRT model fitting results.

This module defines the core data structures used throughout the IRT library:
- FitResult: Contains all information from a fitted IRT model
- ScoreResult: Contains person ability estimates and standard errors

Model Parameterization
----------------------
The 2PL model uses the parameterization: P(X=1|theta) = 1 / (1 + exp(-a*(theta-b)))
where:
    - theta: person ability parameter
    - a: item discrimination parameter (a > 0)
    - b: item difficulty parameter

The Rasch model is a special case with a = 1 for all items.

The 3PL model adds a guessing parameter c:
    P(X=1|theta) = c + (1 - c) / (1 + exp(-a*(theta-b)))

Identification
--------------
- MML-EM: Identified via the prior distribution on theta (typically N(0,1))
- JMLE: Identified by centering item difficulties (mean(b) = 0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import warnings
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ScoreResult:
    """
    Result of person ability scoring.

    Attributes
    ----------
    theta : np.ndarray
        Person ability estimates, shape (N,).
    se : np.ndarray or None
        Standard errors of ability estimates, shape (N,).
        None if standard errors are not available for the chosen method.
    method : str
        Scoring method used: "eap", "map", or "mle".
    person_names : list[str]
        Names/identifiers for each person.

    Examples
    --------
    >>> result = fit_result.score(method="eap")
    >>> print(result.theta[:5])  # First 5 ability estimates
    >>> print(result.se[:5])     # Their standard errors
    """

    theta: np.ndarray
    se: np.ndarray | None
    method: str
    person_names: list[str]

    def __repr__(self) -> str:
        n_persons = len(self.theta)
        se_info = "available" if self.se is not None else "not available"
        return (
            f"ScoreResult(method='{self.method}', n_persons={n_persons}, "
            f"se={se_info})"
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert scoring results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'person', 'theta', and optionally 'se'.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            ) from e

        data = {"person": self.person_names, "theta": self.theta}
        if self.se is not None:
            data["se"] = self.se
        return pd.DataFrame(data)


@dataclass
class FitResult:
    """
    Result of IRT model fitting.

    This class contains all information from a fitted IRT model, including
    parameter estimates, convergence diagnostics, and methods for scoring
    new or existing response data.

    Attributes
    ----------
    model : str
        Model type: "rasch", "2pl", or "3pl".
    estimator : str
        Estimation method: "mml_em" or "jmle".
    params : dict
        Estimated parameters with keys:
        - "a": np.ndarray of shape (J,) - discrimination parameters
        - "b": np.ndarray of shape (J,) - difficulty parameters
        - "c": np.ndarray of shape (J,) - guessing parameters (3PL only)
        For Rasch, a = np.ones(J).
    converged : bool
        Whether the estimation algorithm converged.
    n_iter : int
        Number of iterations performed.
    loglik : float or None
        Final log-likelihood value. None for JMLE (marginal loglik not computed).
    history : dict
        Convergence history with keys:
        - "loglik": list of log-likelihood values per iteration
        - "max_param_change": list of maximum parameter changes per iteration
    theta_grid : np.ndarray or None
        Quadrature points used for MML-EM, shape (K,). None for JMLE.
    log_prior : np.ndarray or None
        Log-prior values at theta_grid points. None for JMLE.
    posterior_weights : np.ndarray or None
        Posterior weights w_nk from final E-step, shape (N, K). None for JMLE.
    item_names : list[str]
        Names/identifiers for each item.
    person_names : list[str]
        Names/identifiers for each person.
    mask_obs : np.ndarray
        Boolean mask indicating observed responses, shape (N, J).
    X : np.ndarray
        Original response data, shape (N, J).
    _score_fn : Callable or None
        Internal scoring function (set during fitting).

    Examples
    --------
    >>> from irt import fit
    >>> result = fit(X, model="rasch", estimator="mml_em")
    >>> print(result.params["b"])  # Item difficulties
    >>> scores = result.score(method="eap")
    >>> print(scores.theta)  # Person abilities
    """

    model: str
    estimator: str
    params: dict[str, np.ndarray]
    converged: bool
    n_iter: int
    loglik: float | None
    history: dict[str, list[float]]
    theta_grid: np.ndarray | None
    log_prior: np.ndarray | None
    posterior_weights: np.ndarray | None
    item_names: list[str]
    person_names: list[str]
    mask_obs: np.ndarray
    X: np.ndarray
    _score_fn: Callable[..., ScoreResult] | None = field(default=None, repr=False)
    n_categories: np.ndarray | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        b_or_a = self.params.get("b", self.params.get("a", np.array([])))
        n_items = b_or_a.shape[0] if hasattr(b_or_a, "shape") else len(b_or_a)
        n_persons = self.X.shape[0]
        conv_str = "converged" if self.converged else "not converged"
        loglik_str = f"{self.loglik:.2f}" if self.loglik is not None else "N/A"
        return (
            f"FitResult(model='{self.model}', estimator='{self.estimator}', "
            f"n_items={n_items}, n_persons={n_persons}, "
            f"n_iter={self.n_iter}, {conv_str}, loglik={loglik_str})"
        )

    def score(
        self,
        X: np.ndarray | None = None,
        method: str = "eap",
        **kwargs: Any,
    ) -> ScoreResult:
        """
        Compute person ability estimates.

        Parameters
        ----------
        X : array-like or None, optional
            Response data to score. If None, scores the original training data.
            Shape (N, J) with values in {0, 1, NaN}.
        method : {"eap", "map", "mle"}, default="eap"
            Scoring method:
            - "eap": Expected A Posteriori (posterior mean)
            - "map": Maximum A Posteriori (posterior mode)
            - "mle": Maximum Likelihood Estimate
        **kwargs : dict
            Additional arguments passed to the scoring function.

        Returns
        -------
        ScoreResult
            Object containing theta estimates, standard errors, and metadata.

        Raises
        ------
        ValueError
            If method is not recognized or incompatible with the estimator.
        RuntimeError
            If scoring function was not properly initialized during fitting.

        Notes
        -----
        For MML-EM fits:
            - "eap" and "map" use the precomputed posterior weights when X=None
            - For new data, posterior weights are recomputed

        For JMLE fits:
            - "eap" is not available (no posterior weights)
            - "map" and "mle" use Newton-Raphson optimization

        Examples
        --------
        >>> result = fit(X, model="rasch")
        >>> eap_scores = result.score(method="eap")
        >>> map_scores = result.score(method="map")
        """
        if self._score_fn is None:
            raise RuntimeError(
                "Scoring function not initialized. This may indicate a bug in "
                "the fitting procedure. Please report this issue."
            )
        return self._score_fn(X=X, method=method, **kwargs)

    def item_report(self) -> "pd.DataFrame":
        """
        Generate a summary report for item parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'item': item names
            - 'a': discrimination parameters (1.0 for Rasch)
            - 'b': difficulty parameters
            - 'n_obs': number of observed responses per item

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for item_report(). "
                "Install it with: pip install pandas"
            ) from e

        n_obs = self.mask_obs.sum(axis=0)
        POLYTOMOUS = ("pcm", "rsm", "grm", "gpcm", "nrm")
        if self.model in POLYTOMOUS:
            data = {"item": self.item_names, "n_obs": n_obs}
            if "a" in self.params:
                data["a"] = self.params["a"]
            if "b" in self.params:
                b = self.params["b"]
                if b.ndim == 1:
                    data["b"] = b
                else:
                    data["b_mean"] = np.nanmean(np.where(np.isfinite(b), b, np.nan), axis=1)
            if "tau" in self.params:
                data["tau"] = [str(self.params["tau"])] * len(self.item_names)  # type: ignore
            return pd.DataFrame(data)
        data = {
            "item": self.item_names,
            "a": self.params["a"],
            "b": self.params["b"],
            "n_obs": n_obs,
        }
        if "c" in self.params:
            data["c"] = self.params["c"]
        return pd.DataFrame(data)

    def person_report(self) -> "pd.DataFrame":
        """
        Generate a summary report for persons including ability estimates.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'person': person names
            - 'n_items': number of items responded to
            - 'sum_score': raw sum of correct responses
            - 'theta': EAP ability estimate (if available)
            - 'se': standard error of theta (if available)

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for person_report(). "
                "Install it with: pip install pandas"
            ) from e

        n_items = self.mask_obs.sum(axis=1)
        # Compute sum scores, treating NaN as 0 for summation
        X_filled = np.where(self.mask_obs, self.X, 0)
        sum_score = X_filled.sum(axis=1)

        data = {
            "person": self.person_names,
            "n_items": n_items,
            "sum_score": sum_score,
        }

        # Add ability estimates if scoring is available
        if self._score_fn is not None:
            try:
                scores = self.score(method="eap")
                data["theta"] = scores.theta
                if scores.se is not None:
                    data["se"] = scores.se
            except (ValueError, RuntimeError):
                warnings.warn(
                    "Ability estimates are not available for this fit, so "
                    "person_report() is returning only raw summaries.",
                    UserWarning,
                    stacklevel=2,
                )
                # EAP not available (e.g., JMLE without posterior)
                pass

        return pd.DataFrame(data)

    # =========================================================================
    # Diagnostic Methods
    # =========================================================================

    def item_fit(self, method: str = "eap") -> "pd.DataFrame":
        """
        Compute comprehensive item fit statistics.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.

        Returns
        -------
        pd.DataFrame
            Item fit table with columns: item, a, b, n_obs, p_value,
            disc_emp, rpb, infit_ms, infit_z, outfit_ms, outfit_z,
            chi_sq, chi_p.

        Examples
        --------
        >>> result = fit(X)
        >>> item_fit = result.item_fit()
        >>> misfitting = item_fit[item_fit['infit_ms'] > 1.5]
        """
        from .diagnostics import item_fit_table, item_fit_table_poly

        try:
            scores = self.score(method=method)
            theta = scores.theta
        except (ValueError, RuntimeError):
            scores = self.score(method="map")
            theta = scores.theta

        POLYTOMOUS = ("pcm", "rsm", "grm", "gpcm", "nrm")
        if self.model in POLYTOMOUS:
            n_cat = getattr(self, "n_categories", None)
            if n_cat is None:
                n_cat = np.max(self.X[self.mask_obs].astype(int)) + 1
                n_cat = np.full(self.X.shape[1], int(n_cat))
            return item_fit_table_poly(
                X=self.X,
                mask_obs=self.mask_obs,
                theta=theta,
                params=self.params,
                model=self.model,
                n_categories=n_cat,
                item_names=self.item_names,
            )
        return item_fit_table(
            X=self.X,
            mask_obs=self.mask_obs,
            theta=theta,
            a=self.params["a"],
            b=self.params["b"],
            c=self.params.get("c"),
            item_names=self.item_names,
        )

    def person_fit(self, method: str = "eap") -> "pd.DataFrame":
        """
        Compute person fit statistics.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.

        Returns
        -------
        pd.DataFrame
            Person fit table with columns: person, theta, se,
            infit_ms, infit_z, outfit_ms, outfit_z.
        """
        from .diagnostics import infit_outfit_persons, infit_outfit_persons_poly

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for person_fit(). "
                "Install it with: pip install pandas"
            ) from e

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            warnings.warn(
                f"Scoring method '{method}' is not available; "
                "falling back to MAP for empirical ICCs.",
                UserWarning,
                stacklevel=2,
            )
            scores = self.score(method="map")

        theta = scores.theta
        se = scores.se if scores.se is not None else np.full_like(theta, np.nan)

        POLYTOMOUS = ("pcm", "rsm", "grm", "gpcm", "nrm")
        if self.model in POLYTOMOUS:
            n_cat = getattr(self, "n_categories", None)
            if n_cat is None:
                n_cat = np.full(self.X.shape[1], int(np.nanmax(self.X[self.mask_obs]) + 1))
            infit_ms, infit_z, outfit_ms, outfit_z = infit_outfit_persons_poly(
                self.X, self.mask_obs, theta, self.params, self.model, n_cat
            )
        else:
            infit_ms, infit_z, outfit_ms, outfit_z = infit_outfit_persons(
                self.X, self.mask_obs, theta,
                self.params["a"], self.params["b"], self.params.get("c")
            )

        return pd.DataFrame({
            "person": self.person_names,
            "theta": theta,
            "se": se,
            "infit_ms": infit_ms,
            "infit_z": infit_z,
            "outfit_ms": outfit_ms,
            "outfit_z": outfit_z,
        })

    def model_fit(self, method: str = "eap") -> dict:
        """
        Compute comprehensive model fit summary.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.

        Returns
        -------
        dict
            Dictionary with fit statistics including:
            - n_persons, n_items, n_observations
            - loglik, aic, bic (if available)
            - reliability, separation_index, n_strata
            - m2, m2_df, m2_p, rmsea (if computable)
            - cfi, tli, srmsr (if computable)
            - q3_mean, q3_max_abs, q3_n_pairs, q3_n_flagged
            - item_chisq_mean, item_chisq_max, n_items_misfit_p05

        Examples
        --------
        >>> result = fit(X)
        >>> fit_stats = result.model_fit()
        >>> print(f"Reliability: {fit_stats['reliability']:.3f}")
        >>> print(f"RMSEA: {fit_stats.get('rmsea', 'N/A')}")
        """
        from .diagnostics import model_fit_summary, model_fit_summary_poly

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        theta = scores.theta
        se = scores.se if scores.se is not None else np.full_like(theta, np.nan)

        POLYTOMOUS = ("pcm", "rsm", "grm", "gpcm", "nrm")
        if self.model in POLYTOMOUS:
            n_cat = getattr(self, "n_categories", None)
            if n_cat is None:
                n_cat = np.full(self.X.shape[1], int(np.nanmax(self.X[self.mask_obs]) + 1))
            return model_fit_summary_poly(
                self.X, self.mask_obs, theta, se,
                self.params, self.model, n_cat, self.loglik
            )

        return model_fit_summary(
            X=self.X,
            mask_obs=self.mask_obs,
            theta=theta,
            se=se,
            a=self.params["a"],
            b=self.params["b"],
            c=self.params.get("c"),
            loglik=self.loglik,
            model=self.model,
        )

    def summary(self, method: str = "eap", as_dataframe: bool = False):
        """
        Return a concise model summary for reporting.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        as_dataframe : bool, default=False
            If True, return a single-row pandas DataFrame (requires pandas).
        """
        stats = self.model_fit(method=method)
        summary = {
            "model": self.model,
            "estimator": self.estimator,
            "n_items": self.n_items,
            "n_persons": self.n_persons,
            **stats,
        }

        if not as_dataframe:
            return summary

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for summary(as_dataframe=True). "
                "Install it with: pip install pandas"
            ) from e

        return pd.DataFrame([summary])

    def coef(
        self,
        irt: bool = True,
        as_dataframe: bool = True,
    ):
        """
        Return item parameter estimates in a report-friendly format.

        Parameters
        ----------
        irt : bool, default=True
            If True, return IRT-parameterization (a, b, c).
            If False, return slope-intercept form (a, d, c) where d = -a*b.
        as_dataframe : bool, default=True
            If True, return a pandas DataFrame (requires pandas).
        """
        a = self.params["a"]
        b = self.params["b"]
        c = self.params.get("c")

        if irt:
            data = {"item": self.item_names, "a": a, "b": b}
            if c is not None:
                data["c"] = c
        else:
            d = -a * b
            data = {"item": self.item_names, "a": a, "d": d}
            if c is not None:
                data["c"] = c

        if not as_dataframe:
            return data

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required for coef(as_dataframe=True). "
                "Install it with: pip install pandas"
            ) from e

        return pd.DataFrame(data)

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot_icc(
        self,
        items: list[int] | None = None,
        theta_range: tuple[float, float] = (-4, 4),
        **kwargs,
    ):
        """
        Plot Item Characteristic Curves.

        Parameters
        ----------
        items : list[int], optional
            Indices of items to plot. If None, plots all items.
        theta_range : tuple, default=(-4, 4)
            Range of theta values.
        **kwargs
            Additional arguments passed to plot_icc.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.

        Examples
        --------
        >>> result = fit(X)
        >>> fig, ax = result.plot_icc(items=[0, 1, 2])
        """
        from .plotting import plot_icc, plot_ccc

        POLYTOMOUS = ("pcm", "rsm", "grm", "gpcm", "nrm")
        if self.model in POLYTOMOUS:
            item_idx = items[0] if items is not None else 0
            n_cat = int(self.n_categories[item_idx]) if self.n_categories is not None else 4
            return plot_ccc(
                self.params, self.model, item_idx, n_cat,
                theta_range=theta_range, **kwargs
            )

        a = self.params["a"]
        b = self.params["b"]
        c = self.params.get("c")

        if items is not None:
            a = a[items]
            b = b[items]
            labels = [self.item_names[i] for i in items]
        else:
            labels = self.item_names

        return plot_icc(a, b, c=c, theta_range=theta_range, item_labels=labels, **kwargs)

    def plot_icc_empirical(
        self,
        item_idx: int,
        method: str = "eap",
        n_groups: int = 10,
        **kwargs,
    ):
        """
        Plot ICC with empirical observed proportions overlaid.

        Parameters
        ----------
        item_idx : int
            Index of item to plot.
        method : str, default="eap"
            Scoring method for ability estimates.
        n_groups : int, default=10
            Number of ability groups for empirical points.
        **kwargs
            Additional arguments passed to plot_icc_with_empirical.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_icc_with_empirical

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        return plot_icc_with_empirical(
            X=self.X,
            mask_obs=self.mask_obs,
            theta=scores.theta,
            a=self.params["a"],
            b=self.params["b"],
            c=self.params.get("c"),
            item_idx=item_idx,
            n_groups=n_groups,
            title=f"{self.item_names[item_idx]}: ICC with Empirical Fit",
            **kwargs,
        )

    def plot_all_iccs(self, **kwargs):
        """
        Plot ICCs for all items in a grid layout.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot_all_iccs.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes array.
        """
        from .plotting import plot_all_iccs

        return plot_all_iccs(
            self.params["a"],
            self.params["b"],
            c=self.params.get("c"),
            item_names=self.item_names,
            **kwargs,
        )

    def plot_iif(self, items: list[int] | None = None, **kwargs):
        """
        Plot Item Information Functions.

        Parameters
        ----------
        items : list[int], optional
            Indices of items to plot. If None, plots all items.
        **kwargs
            Additional arguments passed to plot_iif.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_iif

        a = self.params["a"]
        b = self.params["b"]
        c = self.params.get("c")

        if items is not None:
            a = a[items]
            b = b[items]
            labels = [self.item_names[i] for i in items]
        else:
            labels = self.item_names

        return plot_iif(a, b, c=c, item_labels=labels, **kwargs)

    def plot_tif(self, **kwargs):
        """
        Plot Test Information Function.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to plot_tif.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_tif

        return plot_tif(self.params["a"], self.params["b"], c=self.params.get("c"), **kwargs)

    def plot_wright_map(self, method: str = "eap", **kwargs):
        """
        Plot Wright Map (Item-Person Map).

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        **kwargs
            Additional arguments passed to plot_wright_map.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes tuple.
        """
        from .plotting import plot_wright_map

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        return plot_wright_map(
            theta=scores.theta,
            b=self.params["b"],
            theta_se=scores.se,
            item_names=self.item_names,
            **kwargs,
        )

    def plot_item_fit(self, method: str = "eap", **kwargs):
        """
        Plot item fit statistics (infit vs outfit scatter).

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        **kwargs
            Additional arguments passed to plot_item_fit.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_item_fit
        from .diagnostics import infit_outfit_items

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        infit_ms, _, outfit_ms, _ = infit_outfit_items(
            self.X, self.mask_obs, scores.theta,
            self.params["a"], self.params["b"], self.params.get("c")
        )

        return plot_item_fit(
            infit_ms, outfit_ms, self.params["b"],
            item_names=self.item_names, **kwargs
        )

    def plot_item_fit_bars(self, method: str = "eap", **kwargs):
        """
        Plot item fit statistics as bar charts.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        **kwargs
            Additional arguments passed to plot_item_fit_bars.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes tuple.
        """
        from .plotting import plot_item_fit_bars
        from .diagnostics import infit_outfit_items

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        infit_ms, _, outfit_ms, _ = infit_outfit_items(
            self.X, self.mask_obs, scores.theta,
            self.params["a"], self.params["b"], self.params.get("c")
        )

        return plot_item_fit_bars(
            infit_ms, outfit_ms,
            item_names=self.item_names, **kwargs
        )

    def plot_person_fit(self, method: str = "eap", **kwargs):
        """
        Plot person fit statistics by ability.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        **kwargs
            Additional arguments passed to plot_person_fit.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_person_fit
        from .diagnostics import infit_outfit_persons

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        infit_ms, _, outfit_ms, _ = infit_outfit_persons(
            self.X, self.mask_obs, scores.theta,
            self.params["a"], self.params["b"]
        )

        return plot_person_fit(infit_ms, outfit_ms, scores.theta, **kwargs)

    def plot_ability_distribution(self, method: str = "eap", **kwargs):
        """
        Plot distribution of ability estimates.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        **kwargs
            Additional arguments passed to plot_ability_distribution.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_ability_distribution

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        return plot_ability_distribution(scores.theta, scores.se, **kwargs)

    def plot_se_by_theta(self, method: str = "eap", **kwargs):
        """
        Plot standard errors by ability level.

        Parameters
        ----------
        method : str, default="eap"
            Scoring method for ability estimates.
        **kwargs
            Additional arguments passed to plot_se_by_theta.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        from .plotting import plot_se_by_theta

        try:
            scores = self.score(method=method)
        except (ValueError, RuntimeError):
            scores = self.score(method="map")

        return plot_se_by_theta(
            scores.theta, scores.se,
            a=self.params["a"], b=self.params["b"], c=self.params.get("c"),
            **kwargs
        )


# Default technical parameters for MML-EM
MML_EM_DEFAULTS: dict[str, Any] = {
    "quadpts": 61,
    "theta_lo": -4.0,
    "theta_hi": 4.0,
    "prior": ("normal", 0.0, 1.0),
    "max_iter": 200,
    "tol": 1e-4,
    "a_bounds": (0.25, 4.0),
    "b_bounds": (-6.0, 6.0),
    "c_bounds": (1e-6, 0.35),
    "mstep_max_iter": 25,
}

# Default technical parameters for JMLE
JMLE_DEFAULTS: dict[str, Any] = {
    "max_iter": 50,
    "tol": 1e-4,
    "jmle_method": "newton",
    "nudge": 0.3,
    "theta_bounds": (-6.0, 6.0),
    "b_bounds": (-6.0, 6.0),
}
