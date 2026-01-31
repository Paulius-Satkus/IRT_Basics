"""
Diagnostic statistics for IRT model fit.

This module provides functions for assessing model-data fit:
- Item fit statistics (infit, outfit)
- Person fit statistics
- Point-biserial correlations
- Residual-based diagnostics (Q3, SRMSR)
- Model fit indices (M2, RMSEA, CFI, TLI)

Fit Statistics
--------------
Infit and outfit are standardized residual-based statistics:

- Outfit (unweighted): Mean squared standardized residual
  Sensitive to unexpected responses far from the person's ability

- Infit (weighted): Information-weighted mean squared standardized residual
  Sensitive to unexpected responses near the person's ability

Expected values are 1.0. Values > 1.5 indicate underfit (more noise than
expected), values < 0.5 indicate overfit (less variation than expected,
often due to redundant items or Guttman patterns).

References
----------
- Wright, B. D., & Masters, G. N. (1982). Rating Scale Analysis.
- Linacre, J. M. (2002). What do infit and outfit, mean-square and 
  standardized mean? Rasch Measurement Transactions, 16(2), 878.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import expit

if TYPE_CHECKING:
    import pandas as pd


def compute_residuals(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute raw and standardized residuals.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).

    Returns
    -------
    E : np.ndarray
        Expected responses, shape (N, J). E[X_nj] = P_j(theta_n)
    residual : np.ndarray
        Raw residuals, shape (N, J). r_nj = X_nj - E_nj
    std_residual : np.ndarray
        Standardized residuals, shape (N, J). z_nj = r_nj / sqrt(var)
    """
    N, J = X.shape

    # Compute expected values: P(X=1 | theta, a, b)
    # E[n, j] = 1 / (1 + exp(-a[j] * (theta[n] - b[j])))
    z = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    p_2pl = expit(z)
    if c is None:
        E = p_2pl
    else:
        E = c[np.newaxis, :] + (1.0 - c[np.newaxis, :]) * p_2pl

    # Variance: Var(X) = P * (1 - P)
    V = E * (1 - E)

    # Clip variance for numerical stability
    V = np.clip(V, 1e-12, None)

    # Raw residual
    residual = np.where(mask_obs, X - E, 0.0)

    # Standardized residual
    std_residual = np.where(mask_obs, residual / np.sqrt(V), 0.0)

    return E, residual, std_residual


def _pairwise_corr(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute Pearson correlation for two 1D arrays."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))
    if denom <= 0:
        return np.nan
    return float(np.sum(x_centered * y_centered) / denom)


def q3_residual_correlation(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    min_obs: int = 3,
) -> np.ndarray:
    """
    Compute Yen's Q3 residual correlations between item pairs.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Ability estimates, shape (N,).
    a : np.ndarray
        Item discriminations, shape (J,).
    b : np.ndarray
        Item difficulties, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).
    min_obs : int
        Minimum pairwise observations required to compute correlation.

    Returns
    -------
    q3 : np.ndarray
        Q3 residual correlation matrix, shape (J, J) with np.nan on the diagonal.
    """
    E, residual, _ = compute_residuals(X, mask_obs, theta, a, b, c=c)
    J = X.shape[1]
    q3 = np.full((J, J), np.nan, dtype=float)

    for j in range(J):
        for k in range(j + 1, J):
            obs = mask_obs[:, j] & mask_obs[:, k]
            if obs.sum() < min_obs:
                continue
            r_j = residual[obs, j]
            r_k = residual[obs, k]
            q3_val = _pairwise_corr(r_j, r_k)
            q3[j, k] = q3_val
            q3[k, j] = q3_val

    return q3


def q3_summary(
    q3: np.ndarray,
    threshold: float = 0.2,
) -> dict:
    """
    Summarize Q3 residual correlations.

    Parameters
    ----------
    q3 : np.ndarray
        Q3 matrix, shape (J, J).
    threshold : float
        Absolute threshold for flagging potential local dependence.

    Returns
    -------
    dict
        Summary stats: mean, max_abs, n_pairs, n_flagged.
    """
    if q3.size == 0:
        return {"q3_mean": np.nan, "q3_max_abs": np.nan, "q3_n_pairs": 0, "q3_n_flagged": 0}

    tri = q3[np.tril_indices_from(q3, k=-1)]
    tri = tri[~np.isnan(tri)]
    if tri.size == 0:
        return {"q3_mean": np.nan, "q3_max_abs": np.nan, "q3_n_pairs": 0, "q3_n_flagged": 0}

    return {
        "q3_mean": float(np.mean(tri)),
        "q3_max_abs": float(np.max(np.abs(tri))),
        "q3_n_pairs": int(tri.size),
        "q3_n_flagged": int((np.abs(tri) >= threshold).sum()),
    }


def _model_implied_correlation(
    P: np.ndarray,
    mask_obs: np.ndarray,
    min_obs: int = 3,
) -> np.ndarray:
    """
    Compute model-implied correlation matrix from predicted probabilities.
    """
    N, J = P.shape
    corr = np.full((J, J), np.nan, dtype=float)

    for j in range(J):
        corr[j, j] = 1.0
        for k in range(j + 1, J):
            obs = mask_obs[:, j] & mask_obs[:, k]
            if obs.sum() < min_obs:
                continue
            p_j = P[obs, j]
            p_k = P[obs, k]
            mean_j = p_j.mean()
            mean_k = p_k.mean()
            cov = (p_j * p_k).mean() - mean_j * mean_k
            var_j = (p_j * (1 - p_j)).mean() + p_j.var(ddof=0)
            var_k = (p_k * (1 - p_k)).mean() + p_k.var(ddof=0)
            denom = np.sqrt(var_j * var_k)
            if denom <= 0:
                continue
            corr_val = float(cov / denom)
            corr[j, k] = corr_val
            corr[k, j] = corr_val

    return corr


def _observed_correlation(
    X: np.ndarray,
    mask_obs: np.ndarray,
    min_obs: int = 3,
) -> np.ndarray:
    """
    Compute observed correlation matrix with pairwise complete cases.
    """
    N, J = X.shape
    corr = np.full((J, J), np.nan, dtype=float)

    for j in range(J):
        corr[j, j] = 1.0
        for k in range(j + 1, J):
            obs = mask_obs[:, j] & mask_obs[:, k]
            if obs.sum() < min_obs:
                continue
            x_j = X[obs, j]
            x_k = X[obs, k]
            corr_val = _pairwise_corr(x_j, x_k)
            corr[j, k] = corr_val
            corr[k, j] = corr_val

    return corr


def srmsr_statistic(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    min_obs: int = 3,
) -> float:
    """
    Compute SRMSR (Standardized Root Mean Square Residual) for item correlations.

    Returns
    -------
    float
        SRMSR, or np.nan if insufficient data.
    """
    E, _, _ = compute_residuals(X, mask_obs, theta, a, b, c=c)
    r_obs = _observed_correlation(X, mask_obs, min_obs=min_obs)
    r_model = _model_implied_correlation(E, mask_obs, min_obs=min_obs)

    diffs = r_obs - r_model
    tri = diffs[np.tril_indices_from(diffs, k=-1)]
    tri = tri[~np.isnan(tri)]
    if tri.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(tri ** 2)))


def m2_baseline_statistic(
    X: np.ndarray,
    mask_obs: np.ndarray,
    min_complete: int = 100,
) -> tuple[float, int]:
    """
    Compute baseline (independence) M2 statistic for CFI/TLI.

    The baseline model matches observed univariate proportions and assumes
    item independence.
    """
    N, J = X.shape
    complete = mask_obs.all(axis=1)
    if complete.sum() < min_complete:
        return np.nan, 0

    X_complete = X[complete]
    N_complete = X_complete.shape[0]

    obs_uni = X_complete.mean(axis=0)
    obs_bi = (X_complete.T @ X_complete) / N_complete

    exp_uni = obs_uni
    exp_bi = np.outer(obs_uni, obs_uni)

    resid_uni = obs_uni - exp_uni
    resid_bi = []
    for j in range(J):
        for k in range(j):
            resid_bi.append(obs_bi[j, k] - exp_bi[j, k])
    resid_bi = np.array(resid_bi)
    resid = np.concatenate([resid_uni, resid_bi])

    n_uni = J
    n_bi = J * (J - 1) // 2
    n_stats = n_uni + n_bi

    n_params = J
    df = max(1, n_stats - n_params)

    var_uni = (exp_uni * (1 - exp_uni)) / N_complete
    var_bi = []
    for j in range(J):
        for k in range(j):
            p_jk = exp_bi[j, k]
            var_jk = (p_jk * (1 - p_jk)) / N_complete
            var_bi.append(max(var_jk, 1e-10))
    var_bi = np.array(var_bi)
    var_all = np.concatenate([np.maximum(var_uni, 1e-10), var_bi])

    m2 = np.sum(resid ** 2 / var_all)
    return float(m2), int(df)


def cfi_tli_from_m2(
    m2: float,
    df: int,
    m2_null: float,
    df_null: int,
) -> tuple[float, float]:
    """
    Compute CFI and TLI from model and baseline M2 statistics.
    """
    if df <= 0 or df_null <= 0 or np.isnan(m2) or np.isnan(m2_null):
        return np.nan, np.nan

    denom = m2_null - df_null
    if denom <= 0:
        return np.nan, np.nan

    cfi = 1.0 - max(m2 - df, 0.0) / denom
    tli_denom = (m2_null / df_null) - 1.0
    if tli_denom <= 0:
        tli = np.nan
    else:
        tli = (m2_null / df_null - m2 / df) / tli_denom

    cfi = float(np.clip(cfi, 0.0, 1.0))
    return cfi, float(tli)


def infit_outfit_items(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute infit and outfit statistics for items.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).

    Returns
    -------
    infit_ms : np.ndarray
        Infit mean-square statistics, shape (J,).
    infit_z : np.ndarray
        Infit standardized (z) statistics, shape (J,).
    outfit_ms : np.ndarray
        Outfit mean-square statistics, shape (J,).
    outfit_z : np.ndarray
        Outfit standardized (z) statistics, shape (J,).

    Notes
    -----
    Outfit MS = mean(z^2) for item j across all persons
    Infit MS = sum(var * z^2) / sum(var) for item j

    Z standardization uses Wilson-Hilferty cube root transformation.
    """
    N, J = X.shape

    # Get expected values and residuals
    E, _, std_residual = compute_residuals(X, mask_obs, theta, a, b, c=c)

    # Variance at each cell
    V = E * (1 - E)
    V = np.clip(V, 1e-12, None)

    # Squared standardized residual
    z_sq = std_residual ** 2

    # Item statistics
    infit_ms = np.zeros(J, dtype=np.float64)
    outfit_ms = np.zeros(J, dtype=np.float64)
    infit_z = np.zeros(J, dtype=np.float64)
    outfit_z = np.zeros(J, dtype=np.float64)

    for j in range(J):
        obs_j = mask_obs[:, j]
        n_obs = obs_j.sum()

        if n_obs < 2:
            infit_ms[j] = np.nan
            outfit_ms[j] = np.nan
            infit_z[j] = np.nan
            outfit_z[j] = np.nan
            continue

        z_sq_j = z_sq[obs_j, j]
        V_j = V[obs_j, j]

        # Outfit MS: unweighted mean of squared std residuals
        outfit_ms[j] = z_sq_j.mean()

        # Infit MS: weighted mean
        infit_ms[j] = np.sum(V_j * z_sq_j) / np.sum(V_j)

        # Compute variance of fit statistics for z-standardization
        # Kurtosis excess for Bernoulli
        kurt_excess = (1 - 2 * E[obs_j, j]) ** 2 / V_j - 1

        # Outfit variance
        var_outfit = kurt_excess.sum() / n_obs ** 2

        # Infit variance (more complex)
        sum_V = V_j.sum()
        var_infit = (V_j ** 2 * kurt_excess).sum() / sum_V ** 2

        # Z-standardization using Wilson-Hilferty transformation
        # z = (MS^(1/3) - 1) * 3 / sqrt(var) + sqrt(var) / 3
        # Simplified: z ≈ (MS - 1) / sqrt(var) for small var

        if var_outfit > 1e-12:
            # Wilson-Hilferty
            q_outfit = (var_outfit / 2) ** (1/3)
            outfit_z[j] = (outfit_ms[j] ** (1/3) - 1 + q_outfit ** 3) / q_outfit
        else:
            outfit_z[j] = 0.0

        if var_infit > 1e-12:
            q_infit = (var_infit / 2) ** (1/3)
            infit_z[j] = (infit_ms[j] ** (1/3) - 1 + q_infit ** 3) / q_infit
        else:
            infit_z[j] = 0.0

    return infit_ms, infit_z, outfit_ms, outfit_z


def infit_outfit_persons(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute infit and outfit statistics for persons.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).

    Returns
    -------
    infit_ms : np.ndarray
        Infit mean-square statistics, shape (N,).
    infit_z : np.ndarray
        Infit standardized statistics, shape (N,).
    outfit_ms : np.ndarray
        Outfit mean-square statistics, shape (N,).
    outfit_z : np.ndarray
        Outfit standardized statistics, shape (N,).
    """
    N, J = X.shape

    # Get expected values and residuals
    E, _, std_residual = compute_residuals(X, mask_obs, theta, a, b, c=c)

    # Variance
    V = E * (1 - E)
    V = np.clip(V, 1e-12, None)

    # Squared standardized residual
    z_sq = std_residual ** 2

    # Person statistics
    infit_ms = np.zeros(N, dtype=np.float64)
    outfit_ms = np.zeros(N, dtype=np.float64)
    infit_z = np.zeros(N, dtype=np.float64)
    outfit_z = np.zeros(N, dtype=np.float64)

    for i in range(N):
        obs_i = mask_obs[i]
        n_obs = obs_i.sum()

        if n_obs < 2:
            infit_ms[i] = np.nan
            outfit_ms[i] = np.nan
            infit_z[i] = np.nan
            outfit_z[i] = np.nan
            continue

        z_sq_i = z_sq[i, obs_i]
        V_i = V[i, obs_i]
        E_i = E[i, obs_i]

        # Outfit MS
        outfit_ms[i] = z_sq_i.mean()

        # Infit MS
        infit_ms[i] = np.sum(V_i * z_sq_i) / np.sum(V_i)

        # Kurtosis
        kurt_excess = (1 - 2 * E_i) ** 2 / V_i - 1

        # Variances
        var_outfit = kurt_excess.sum() / n_obs ** 2
        sum_V = V_i.sum()
        var_infit = (V_i ** 2 * kurt_excess).sum() / sum_V ** 2

        # Z-standardization
        if var_outfit > 1e-12:
            q = (var_outfit / 2) ** (1/3)
            outfit_z[i] = (outfit_ms[i] ** (1/3) - 1 + q ** 3) / q
        else:
            outfit_z[i] = 0.0

        if var_infit > 1e-12:
            q = (var_infit / 2) ** (1/3)
            infit_z[i] = (infit_ms[i] ** (1/3) - 1 + q ** 3) / q
        else:
            infit_z[i] = 0.0

    return infit_ms, infit_z, outfit_ms, outfit_z


def infit_outfit(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    item_names: list[str] | None = None,
    person_names: list[str] | None = None,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """
    Compute infit/outfit statistics for items and persons.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    item_names : list[str] or None, optional
        Item names for the DataFrame index.
    person_names : list[str] or None, optional
        Person names for the DataFrame index.

    Returns
    -------
    item_fit : pd.DataFrame
        Item fit statistics with columns:
        - infit_ms: Infit mean square
        - infit_z: Infit z-statistic
        - outfit_ms: Outfit mean square
        - outfit_z: Outfit z-statistic
    person_fit : pd.DataFrame
        Person fit statistics with same columns.

    Raises
    ------
    ImportError
        If pandas is not installed.

    Examples
    --------
    >>> item_fit, person_fit = infit_outfit(X, mask, theta, a, b)
    >>> # Flag misfitting items
    >>> misfitting = item_fit[item_fit['infit_ms'] > 1.5]
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for infit_outfit(). "
            "Install it with: pip install pandas"
        ) from e

    N, J = X.shape

    if item_names is None:
        item_names = [f"item_{j}" for j in range(J)]
    if person_names is None:
        person_names = [f"person_{i}" for i in range(N)]

    # Item fit
    i_infit_ms, i_infit_z, i_outfit_ms, i_outfit_z = infit_outfit_items(
        X, mask_obs, theta, a, b
    )
    item_fit = pd.DataFrame({
        "infit_ms": i_infit_ms,
        "infit_z": i_infit_z,
        "outfit_ms": i_outfit_ms,
        "outfit_z": i_outfit_z,
    }, index=item_names)

    # Person fit
    p_infit_ms, p_infit_z, p_outfit_ms, p_outfit_z = infit_outfit_persons(
        X, mask_obs, theta, a, b
    )
    person_fit = pd.DataFrame({
        "infit_ms": p_infit_ms,
        "infit_z": p_infit_z,
        "outfit_ms": p_outfit_ms,
        "outfit_z": p_outfit_z,
    }, index=person_names)

    return item_fit, person_fit


def point_biserial_items(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """
    Compute point-biserial correlation for each item.

    The point-biserial correlation measures the relationship between
    item responses (0/1) and person abilities. It should be positive
    for well-functioning items.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).

    Returns
    -------
    np.ndarray
        Point-biserial correlations, shape (J,).

    Notes
    -----
    The point-biserial correlation is:
        r_pb = (M1 - M0) / S * sqrt(p * q)

    where:
        M1 = mean theta for persons answering correctly
        M0 = mean theta for persons answering incorrectly
        S = overall standard deviation of theta
        p = proportion correct
        q = 1 - p

    Negative correlations indicate problematic items where low-ability
    persons are more likely to answer correctly than high-ability persons.

    Examples
    --------
    >>> rpb = point_biserial_items(X, mask, theta)
    >>> # Flag items with negative discrimination
    >>> problematic = np.where(rpb < 0)[0]
    """
    N, J = X.shape
    rpb = np.zeros(J, dtype=np.float64)

    # Overall theta SD
    theta_sd = np.std(theta)
    if theta_sd < 1e-12:
        return np.full(J, np.nan)

    for j in range(J):
        obs = mask_obs[:, j]
        n_obs = obs.sum()

        if n_obs < 3:
            rpb[j] = np.nan
            continue

        x_j = X[obs, j]
        theta_obs = theta[obs]

        # Proportion correct
        p = x_j.mean()
        if p < 1e-12 or p > 1 - 1e-12:
            rpb[j] = np.nan
            continue

        # Mean theta for correct and incorrect
        correct = x_j == 1
        incorrect = x_j == 0

        if correct.sum() == 0 or incorrect.sum() == 0:
            rpb[j] = np.nan
            continue

        M1 = theta_obs[correct].mean()
        M0 = theta_obs[incorrect].mean()

        # Point-biserial correlation
        rpb[j] = (M1 - M0) / theta_sd * np.sqrt(p * (1 - p))

    return rpb


def item_discrimination_empirical(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    n_groups: int = 3,
) -> np.ndarray:
    """
    Compute empirical item discrimination using extreme groups.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    n_groups : int, default=3
        Number of groups to divide persons into. Discrimination is
        computed as difference between highest and lowest group.

    Returns
    -------
    np.ndarray
        Empirical discrimination indices, shape (J,).

    Notes
    -----
    This is a non-parametric discrimination measure:
        D = P_high - P_low

    where P_high is the proportion correct in the highest ability group
    and P_low is the proportion correct in the lowest ability group.
    """
    N, J = X.shape
    disc = np.zeros(J, dtype=np.float64)

    # Sort persons by theta
    order = np.argsort(theta)
    group_size = N // n_groups

    if group_size < 2:
        return np.full(J, np.nan)

    low_idx = order[:group_size]
    high_idx = order[-group_size:]

    for j in range(J):
        # Low group
        low_obs = mask_obs[low_idx, j]
        if low_obs.sum() < 1:
            disc[j] = np.nan
            continue
        p_low = X[low_idx[low_obs], j].mean()

        # High group
        high_obs = mask_obs[high_idx, j]
        if high_obs.sum() < 1:
            disc[j] = np.nan
            continue
        p_high = X[high_idx[high_obs], j].mean()

        disc[j] = p_high - p_low

    return disc


def reliability_marginal(
    theta: np.ndarray,
    se: np.ndarray,
) -> float:
    """
    Compute marginal reliability of ability estimates.

    Parameters
    ----------
    theta : np.ndarray
        Ability estimates, shape (N,).
    se : np.ndarray
        Standard errors, shape (N,).

    Returns
    -------
    float
        Marginal reliability coefficient.

    Notes
    -----
    Marginal reliability is:
        rho = 1 - mean(SE^2) / Var(theta)

    This represents the proportion of variance in theta estimates that
    is true score variance (not error variance).

    Equivalent to coefficient alpha / KR-20 in classical test theory
    when the model fits.
    """
    theta_var = np.var(theta)
    if theta_var < 1e-12:
        return 0.0

    # Use only finite SEs
    valid = np.isfinite(se)
    if valid.sum() < 2:
        return np.nan

    mean_se_sq = np.mean(se[valid] ** 2)
    reliability = 1 - mean_se_sq / theta_var
    return float(max(0.0, min(1.0, reliability)))


def separation_index(
    theta: np.ndarray,
    se: np.ndarray,
) -> tuple[float, float]:
    """
    Compute person separation statistics.

    Parameters
    ----------
    theta : np.ndarray
        Ability estimates, shape (N,).
    se : np.ndarray
        Standard errors, shape (N,).

    Returns
    -------
    separation : float
        Separation index (ratio of adjusted SD to RMSE).
    n_strata : float
        Number of statistically distinct strata.

    Notes
    -----
    Separation index G = SD_adj / RMSE where:
        SD_adj = sqrt(Var(theta) - mean(SE^2))
        RMSE = sqrt(mean(SE^2))

    Number of strata = (4 * G + 1) / 3

    Higher values indicate better ability to separate persons.
    G > 2 is considered good (at least 3 strata).
    """
    valid = np.isfinite(se)
    if valid.sum() < 2:
        return np.nan, np.nan

    theta_var = np.var(theta[valid])
    mean_se_sq = np.mean(se[valid] ** 2)

    # Adjusted variance
    adj_var = max(0, theta_var - mean_se_sq)
    sd_adj = np.sqrt(adj_var)

    # RMSE
    rmse = np.sqrt(mean_se_sq)

    if rmse < 1e-12:
        return np.inf, np.inf

    separation = sd_adj / rmse
    n_strata = (4 * separation + 1) / 3

    return float(separation), float(n_strata)


# =============================================================================
# Global Model Fit Statistics
# =============================================================================


def item_chi_square(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    n_groups: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute chi-square fit statistics for each item.

    Persons are grouped by ability, and observed vs expected proportions
    are compared using Pearson chi-square.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    a : np.ndarray
        Item discriminations, shape (J,).
    b : np.ndarray
        Item difficulties, shape (J,).
    n_groups : int, default=10
        Number of ability groups.

    Returns
    -------
    chi_sq : np.ndarray
        Chi-square statistics, shape (J,).
    df : np.ndarray
        Degrees of freedom, shape (J,).
    p_values : np.ndarray
        P-values, shape (J,).

    Notes
    -----
    For each item j:
    - Divide persons into n_groups based on ability
    - In each group g, compute:
      - O_g = observed count correct
      - E_g = expected count = n_g * P(theta_g)
    - Chi-square = sum_g (O_g - E_g)^2 / (n_g * p_g * (1-p_g))

    df = n_groups - 1 (or fewer if groups have no data)
    """
    from scipy import stats as sp_stats

    N, J = X.shape

    # Sort persons by ability
    order = np.argsort(theta)
    theta_sorted = theta[order]

    # Create ability groups
    group_size = N // n_groups
    if group_size < 5:
        # Too few per group, reduce groups
        n_groups = max(2, N // 5)
        group_size = N // n_groups

    chi_sq = np.zeros(J, dtype=np.float64)
    df = np.zeros(J, dtype=np.int32)
    p_values = np.zeros(J, dtype=np.float64)

    for j in range(J):
        chi2_j = 0.0
        n_valid_groups = 0

        for g in range(n_groups):
            start_idx = g * group_size
            if g == n_groups - 1:
                end_idx = N
            else:
                end_idx = (g + 1) * group_size

            group_idx = order[start_idx:end_idx]

            # Get responses and mask for this group
            x_group = X[group_idx, j]
            mask_group = mask_obs[group_idx, j]
            theta_group = theta[group_idx]

            n_obs = mask_group.sum()
            if n_obs < 2:
                continue

            # Observed proportion correct
            o_correct = x_group[mask_group].sum()

            # Expected probability (mean of group)
            theta_mean = theta_group[mask_group].mean()
            z = a[j] * (theta_mean - b[j])
            p_2pl = expit(z)
            if c is None:
                p_expected = p_2pl
            else:
                p_expected = c[j] + (1.0 - c[j]) * p_2pl
            p_expected = np.clip(p_expected, 0.01, 0.99)

            e_correct = n_obs * p_expected

            # Variance under null
            variance = n_obs * p_expected * (1 - p_expected)

            if variance > 0.01:
                chi2_j += (o_correct - e_correct) ** 2 / variance
                n_valid_groups += 1

        if n_valid_groups > 1:
            chi_sq[j] = chi2_j
            df[j] = n_valid_groups - 1
            p_values[j] = 1 - sp_stats.chi2.cdf(chi2_j, df[j])
        else:
            chi_sq[j] = np.nan
            df[j] = 0
            p_values[j] = np.nan

    return chi_sq, df, p_values


def compute_aic_bic(
    loglik: float,
    n_persons: int,
    n_items: int,
    model: str,
) -> tuple[float, float]:
    """
    Compute AIC and BIC for model comparison.

    Parameters
    ----------
    loglik : float
        Log-likelihood of the fitted model.
    n_persons : int
        Number of persons.
    n_items : int
        Number of items.
    model : str
        Model type: "rasch", "2pl", or "3pl".

    Returns
    -------
    aic : float
        Akaike Information Criterion.
    bic : float
        Bayesian Information Criterion.

    Notes
    -----
    Number of parameters:
    - Rasch: J difficulty parameters
    - 2PL: J difficulty + J discrimination = 2J parameters
    - 3PL: J difficulty + J discrimination + J guessing = 3J parameters

    AIC = -2 * loglik + 2 * k
    BIC = -2 * loglik + k * log(n)
    """
    model = model.lower()
    if model == "rasch":
        n_params = n_items  # Only b parameters
    elif model == "2pl":
        n_params = 2 * n_items  # a and b parameters
    elif model == "3pl":
        n_params = 3 * n_items
    else:
        raise ValueError(f"Unknown model '{model}'.")

    n_obs = n_persons  # Could also use total observations

    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + n_params * np.log(n_obs)

    return float(aic), float(bic)


def likelihood_ratio_test(
    loglik_full: float,
    loglik_reduced: float,
    df: int,
) -> tuple[float, float]:
    """
    Perform likelihood ratio test between nested models.

    Parameters
    ----------
    loglik_full : float
        Log-likelihood of full (more complex) model.
    loglik_reduced : float
        Log-likelihood of reduced (simpler) model.
    df : int
        Degrees of freedom (difference in number of parameters).

    Returns
    -------
    chi_sq : float
        Chi-square test statistic.
    p_value : float
        P-value.

    Notes
    -----
    The test statistic is:
        LR = -2 * (loglik_reduced - loglik_full)
           = 2 * (loglik_full - loglik_reduced)

    Under H0 (reduced model is correct), LR ~ chi-square(df)
    """
    from scipy import stats as sp_stats

    chi_sq = 2 * (loglik_full - loglik_reduced)

    if chi_sq < 0:
        # This shouldn't happen if models are properly nested
        chi_sq = 0.0

    p_value = 1 - sp_stats.chi2.cdf(chi_sq, df)

    return float(chi_sq), float(p_value)


def m2_statistic(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
) -> tuple[float, int, float, float]:
    """
    Compute M2 limited-information goodness-of-fit statistic.

    M2 tests whether the model-implied first and second-order marginals
    match the observed marginals.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Ability estimates, shape (N,).
    a : np.ndarray
        Item discriminations, shape (J,).
    b : np.ndarray
        Item difficulties, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).

    Returns
    -------
    m2 : float
        M2 statistic.
    df : int
        Degrees of freedom.
    p_value : float
        P-value.
    rmsea : float
        Root Mean Square Error of Approximation.

    Notes
    -----
    M2 is based on comparing observed vs expected:
    - Univariate margins: P(X_j = 1) for each item
    - Bivariate margins: P(X_j = 1, X_k = 1) for each item pair

    This is a large-sample test that requires complete data
    or proper handling of missing data.
    """
    from scipy import stats as sp_stats

    N, J = X.shape

    # For simplicity, use only complete cases
    complete = mask_obs.all(axis=1)
    if complete.sum() < 100:
        return np.nan, 0, np.nan, np.nan

    X_complete = X[complete]
    theta_complete = theta[complete]
    N_complete = X_complete.shape[0]

    # Compute expected probabilities
    z = a[np.newaxis, :] * (theta_complete[:, np.newaxis] - b[np.newaxis, :])
    p_2pl = expit(z)
    if c is None:
        P = p_2pl
    else:
        P = c[np.newaxis, :] + (1.0 - c[np.newaxis, :]) * p_2pl

    # Observed univariate proportions
    obs_uni = X_complete.mean(axis=0)  # (J,)

    # Expected univariate proportions
    exp_uni = P.mean(axis=0)  # (J,)

    # Observed bivariate proportions (cross-products)
    obs_bi = (X_complete.T @ X_complete) / N_complete  # (J, J)

    # Expected bivariate proportions
    # E[X_j * X_k] = E[P_j * P_k] approximately
    exp_bi = (P.T @ P) / N_complete  # (J, J)

    # Construct residual vector (univariate + lower-triangle bivariate)
    resid_uni = obs_uni - exp_uni
    resid_bi = []
    for j in range(J):
        for k in range(j):
            resid_bi.append(obs_bi[j, k] - exp_bi[j, k])
    resid_bi = np.array(resid_bi)

    # Full residual vector
    resid = np.concatenate([resid_uni, resid_bi])

    # Number of statistics
    n_uni = J
    n_bi = J * (J - 1) // 2
    n_stats = n_uni + n_bi

    # Degrees of freedom = n_stats - n_params
    # Rasch: n_params = J, 2PL: n_params = 2J
    # Assume 2PL for conservative test
    n_params = 2 * J
    df = max(1, n_stats - n_params)

    # Compute M2 (simplified - using diagonal weight matrix)
    # True M2 requires full weight matrix which is complex to compute
    # This is an approximation using variances

    # Variance of univariate proportions
    var_uni = (exp_uni * (1 - exp_uni)) / N_complete

    # Variance of bivariate proportions (approximate)
    var_bi = []
    for j in range(J):
        for k in range(j):
            p_jk = exp_bi[j, k]
            var_jk = (p_jk * (1 - p_jk)) / N_complete
            var_bi.append(max(var_jk, 1e-10))
    var_bi = np.array(var_bi)

    var_all = np.concatenate([np.maximum(var_uni, 1e-10), var_bi])

    # Approximate M2
    m2 = np.sum(resid ** 2 / var_all)

    # P-value
    p_value = 1 - sp_stats.chi2.cdf(m2, df)

    # RMSEA
    if m2 > df:
        rmsea = np.sqrt((m2 - df) / (df * N_complete))
    else:
        rmsea = 0.0

    return float(m2), int(df), float(p_value), float(rmsea)


def model_fit_summary(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    se: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None,
    loglik: float | None,
    model: str,
) -> dict:
    """
    Compute comprehensive model fit summary.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Ability estimates, shape (N,).
    se : np.ndarray
        Standard errors, shape (N,).
    a : np.ndarray
        Item discriminations, shape (J,).
    b : np.ndarray
        Item difficulties, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).
    loglik : float or None
        Log-likelihood (if available).
    model : str
        Model type: "rasch" or "2pl".

    Returns
    -------
    dict
        Dictionary with fit statistics:
        - n_persons, n_items: Sample sizes
        - loglik, aic, bic: Information criteria
        - reliability: Marginal reliability
        - separation, n_strata: Person separation
        - m2, m2_df, m2_p, rmsea: M2 statistics (if computable)
        - cfi, tli, srmsr: Additional fit indices (if computable)
        - q3_mean, q3_max_abs, q3_n_pairs, q3_n_flagged: Q3 summaries
        - item_chi_sq_summary: Summary of item chi-square tests
    """
    N, J = X.shape

    summary = {
        "n_persons": N,
        "n_items": J,
        "n_observations": int(mask_obs.sum()),
        "pct_missing": float(100 * (1 - mask_obs.mean())),
        "model": model,
    }

    # Information criteria
    if loglik is not None:
        summary["loglik"] = loglik
        aic, bic = compute_aic_bic(loglik, N, J, model)
        summary["aic"] = aic
        summary["bic"] = bic

    # Reliability
    rel = reliability_marginal(theta, se)
    summary["reliability"] = rel

    # Separation
    sep, strata = separation_index(theta, se)
    summary["separation_index"] = sep
    summary["n_strata"] = strata

    # Item chi-square
    chi_sq, df, p_vals = item_chi_square(X, mask_obs, theta, a, b, c=c)
    valid_chi = ~np.isnan(chi_sq)
    if valid_chi.sum() > 0:
        summary["item_chisq_mean"] = float(np.nanmean(chi_sq))
        summary["item_chisq_max"] = float(np.nanmax(chi_sq))
        summary["n_items_misfit_p05"] = int((p_vals[valid_chi] < 0.05).sum())

    # M2 statistic
    try:
        m2, m2_df, m2_p, rmsea = m2_statistic(X, mask_obs, theta, a, b, c=c)
        if not np.isnan(m2):
            summary["m2"] = m2
            summary["m2_df"] = m2_df
            summary["m2_p"] = m2_p
            summary["rmsea"] = rmsea
            # Baseline M2 for CFI/TLI
            m2_null, df_null = m2_baseline_statistic(X, mask_obs)
            cfi, tli = cfi_tli_from_m2(m2, m2_df, m2_null, df_null)
            summary["cfi"] = cfi
            summary["tli"] = tli
    except Exception:
        pass  # M2 may fail with sparse data

    # SRMSR
    try:
        summary["srmsr"] = srmsr_statistic(X, mask_obs, theta, a, b, c=c)
    except Exception:
        pass

    # Q3 residual correlations
    try:
        q3 = q3_residual_correlation(X, mask_obs, theta, a, b, c=c)
        summary.update(q3_summary(q3))
    except Exception:
        pass

    return summary


def item_fit_table(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    item_names: list[str] | None = None,
) -> "pd.DataFrame":
    """
    Generate comprehensive item fit table.

    Parameters
    ----------
    X : np.ndarray
        Response matrix, shape (N, J).
    mask_obs : np.ndarray
        Observation mask, shape (N, J).
    theta : np.ndarray
        Ability estimates, shape (N,).
    a : np.ndarray
        Item discriminations, shape (J,).
    b : np.ndarray
        Item difficulties, shape (J,).
    c : np.ndarray or None
        Item guessing parameters for 3PL, shape (J,).
    item_names : list[str], optional
        Item names.

    Returns
    -------
    pd.DataFrame
        Item fit table with columns:
        - item: Item name
        - a: Discrimination
        - b: Difficulty
        - n_obs: Number of observations
        - p_value: Proportion correct
        - rpb: Point-biserial correlation
        - infit_ms, infit_z: Infit statistics
        - outfit_ms, outfit_z: Outfit statistics
        - chi_sq, chi_p: Chi-square test
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for item_fit_table(). "
            "Install it with: pip install pandas"
        ) from e

    J = len(a)
    if item_names is None:
        item_names = [f"item_{j}" for j in range(J)]

    # Basic stats
    n_obs = mask_obs.sum(axis=0)
    X_masked = np.where(mask_obs, X, np.nan)
    p_values = np.nanmean(X_masked, axis=0)

    # Point-biserial
    rpb = point_biserial_items(X, mask_obs, theta)

    # Infit/Outfit
    infit_ms, infit_z, outfit_ms, outfit_z = infit_outfit_items(
        X, mask_obs, theta, a, b, c=c
    )

    # Chi-square
    chi_sq, chi_df, chi_p = item_chi_square(X, mask_obs, theta, a, b, c=c)

    # Empirical discrimination
    disc = item_discrimination_empirical(X, mask_obs, theta)

    data = {
        "item": item_names,
        "a": a,
        "b": b,
        "n_obs": n_obs,
        "p_value": p_values,
        "disc_emp": disc,
        "rpb": rpb,
        "infit_ms": infit_ms,
        "infit_z": infit_z,
        "outfit_ms": outfit_ms,
        "outfit_z": outfit_z,
        "chi_sq": chi_sq,
        "chi_p": chi_p,
    }
    if c is not None:
        data["c"] = c
    return pd.DataFrame(data)
