"""
Plotting functions for IRT model visualization.

This module provides comprehensive visualization tools for IRT analysis:
- Item Characteristic Curves (ICCs)
- Item Information Functions (IIFs)
- Test Information Function (TIF)
- Wright Maps (Item-Person Maps)
- Fit statistic plots
- Residual plots
- Ability distribution plots

All plotting functions use matplotlib and return Figure/Axes objects
for further customization.

Note: matplotlib must be installed to use these functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import warnings
import numpy as np
from scipy.special import expit

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting functions. "
            "Install it with: pip install matplotlib"
        )


# =============================================================================
# Item Characteristic Curves (ICCs)
# =============================================================================


def plot_icc(
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float | None = None,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    item_labels: list[str] | None = None,
    ax: "Axes | None" = None,
    title: str | None = None,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    show_legend: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot Item Characteristic Curves (ICCs).

    Parameters
    ----------
    a : float or array-like
        Item discrimination parameter(s). Scalar for single item, array for multiple.
    b : float or array-like
        Item difficulty parameter(s). Must match shape of a.
    theta_range : tuple, default=(-4, 4)
        Range of theta values to plot.
    n_points : int, default=100
        Number of points for smooth curve.
    item_labels : list[str], optional
        Labels for each item in legend.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    title : str, optional
        Plot title. Default: "Item Characteristic Curves".
    colors : list[str], optional
        Colors for each ICC line.
    linestyles : list[str], optional
        Line styles for each ICC.
    show_legend : bool, default=True
        Whether to show legend.
    figsize : tuple, default=(8, 6)
        Figure size if creating new figure.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.

    Examples
    --------
    >>> # Single item
    >>> fig, ax = plot_icc(a=1.0, b=0.5)
    >>>
    >>> # Multiple items
    >>> fig, ax = plot_icc(a=[1.0, 1.5, 2.0], b=[-1, 0, 1])
    """
    plt = _check_matplotlib()

    # Convert to arrays
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    if len(a) != len(b):
        raise ValueError(f"a and b must have same length, got {len(a)} and {len(b)}")
    if c is not None:
        c = np.atleast_1d(c)
        if len(c) != len(a):
            raise ValueError(f"c must have same length as a, got {len(c)} and {len(a)}")

    n_items = len(a)

    # Create theta grid
    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Default labels
    if item_labels is None:
        if n_items == 1:
            if c is None:
                item_labels = [f"a={a[0]:.2f}, b={b[0]:.2f}"]
            else:
                item_labels = [f"a={a[0]:.2f}, b={b[0]:.2f}, c={c[0]:.2f}"]
        else:
            item_labels = [f"Item {j+1}" for j in range(n_items)]

    # Default colors
    if colors is None:
        colors = plt.cm.tab10.colors[:n_items]

    # Default linestyles
    if linestyles is None:
        linestyles = ['-'] * n_items

    # Plot each ICC
    for j in range(n_items):
        z = a[j] * (theta - b[j])
        p_2pl = expit(z)
        if c is None:
            p = p_2pl
        else:
            p = c[j] + (1.0 - c[j]) * p_2pl
        ax.plot(theta, p, color=colors[j % len(colors)],
                linestyle=linestyles[j % len(linestyles)],
                label=item_labels[j], linewidth=2)

    # Formatting
    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel(r'$P(X=1|\theta)$', fontsize=12)
    ax.set_xlim(theta_range)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Item Characteristic Curves"
    ax.set_title(title, fontsize=14)

    if show_legend and n_items > 1:
        ax.legend(loc='best', fontsize=10)

    fig.tight_layout()
    return fig, ax


def plot_ccc(
    params: dict,
    model: str,
    item_idx: int,
    n_categories: int,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot Category Characteristic Curves (CCCs) for a polytomous item.

    P(X=c|theta) vs theta for each category c.
    """
    from .estimators.mml_em_poly import _prob_item_all_theta

    plt = _check_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    theta_plot = np.linspace(theta_range[0], theta_range[1], n_points)
    p = _prob_item_all_theta(theta_plot, params, model, item_idx, n_categories)

    colors = plt.cm.viridis(np.linspace(0, 1, n_categories))
    for c in range(n_categories):
        ax.plot(theta_plot, p[:, c], label=f"P(X={c})", color=colors[c])

    ax.set_xlabel(r"$\theta$ (Ability)", fontsize=12)
    ax.set_ylabel("P(X = c | θ)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    if title is None:
        title = f"Item {item_idx}: Category Characteristic Curves"
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig, ax


def plot_icc_with_empirical(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    item_idx: int,
    c: np.ndarray | None = None,
    n_groups: int = 10,
    theta_range: tuple[float, float] = (-4, 4),
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot ICC with empirical observed proportions overlaid.

    This is crucial for assessing model fit - the empirical points
    should fall close to the theoretical ICC if the model fits well.

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
    item_idx : int
        Index of item to plot.
    n_groups : int, default=10
        Number of ability groups for empirical proportions.
    theta_range : tuple, default=(-4, 4)
        Range of theta values for theoretical curve.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Theoretical ICC
    theta_grid = np.linspace(theta_range[0], theta_range[1], 100)
    z = a[item_idx] * (theta_grid - b[item_idx])
    p_2pl = expit(z)
    if c is None:
        p_theoretical = p_2pl
    else:
        p_theoretical = c[item_idx] + (1.0 - c[item_idx]) * p_2pl

    ax.plot(theta_grid, p_theoretical, 'b-', linewidth=2,
            label='Model ICC', zorder=2)

    # Empirical proportions by ability group
    obs_mask = mask_obs[:, item_idx]
    theta_obs = theta[obs_mask]
    x_obs = X[obs_mask, item_idx]

    if len(theta_obs) < 2:
        warnings.warn(
            "Not enough observed responses to compute empirical ICC points.",
            UserWarning,
            stacklevel=2,
        )
        ax.text(
            0.5,
            0.1,
            "Not enough observations for empirical points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="darkred",
        )
    else:
        if len(theta_obs) <= n_groups:
            n_groups = max(2, len(theta_obs) // 5)
        if n_groups < 2:
            warnings.warn(
                "Too few observations to form ability groups for empirical ICC.",
                UserWarning,
                stacklevel=2,
            )
            ax.text(
                0.5,
                0.1,
                "Not enough observations for empirical points",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="darkred",
            )
        else:
            # Create ability groups
            percentiles = np.linspace(0, 100, n_groups + 1)
            group_bounds = np.percentile(theta_obs, percentiles)

            group_means = []
            group_props = []
            group_counts = []

            for i in range(n_groups):
                if i == n_groups - 1:
                    mask = (theta_obs >= group_bounds[i]) & (theta_obs <= group_bounds[i+1])
                else:
                    mask = (theta_obs >= group_bounds[i]) & (theta_obs < group_bounds[i+1])

                if mask.sum() > 0:
                    group_means.append(theta_obs[mask].mean())
                    group_props.append(x_obs[mask].mean())
                    group_counts.append(mask.sum())

            group_means = np.array(group_means)
            group_props = np.array(group_props)
            group_counts = np.array(group_counts)

            # Plot empirical points with size proportional to count
            sizes = 50 + 200 * (group_counts / group_counts.max())
            ax.scatter(group_means, group_props, s=sizes, c='red', alpha=0.7,
                      edgecolors='darkred', linewidths=1.5, label='Observed',
                      zorder=3)

    # Formatting
    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel(r'$P(X=1|\theta)$', fontsize=12)
    ax.set_xlim(theta_range)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    if title is None:
        title = f"Item {item_idx + 1}: ICC with Empirical Fit"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax


def plot_all_iccs(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    item_names: list[str] | None = None,
    theta_range: tuple[float, float] = (-4, 4),
    n_cols: int = 4,
    figsize_per_item: tuple[float, float] = (3, 2.5),
) -> tuple["Figure", np.ndarray]:
    """
    Plot ICCs for all items in a grid layout.

    Parameters
    ----------
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    item_names : list[str], optional
        Names for each item.
    theta_range : tuple, default=(-4, 4)
        Range of theta values.
    n_cols : int, default=4
        Number of columns in grid.
    figsize_per_item : tuple, default=(3, 2.5)
        Figure size per subplot.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : np.ndarray
        Array of axes.
    """
    plt = _check_matplotlib()

    J = len(a)
    n_rows = int(np.ceil(J / n_cols))

    figsize = (n_cols * figsize_per_item[0], n_rows * figsize_per_item[1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    theta = np.linspace(theta_range[0], theta_range[1], 100)

    if item_names is None:
        item_names = [f"Item {j+1}" for j in range(J)]

    for j in range(J):
        ax = axes[j]
        z = a[j] * (theta - b[j])
        p_2pl = expit(z)
        if c is None:
            p = p_2pl
        else:
            p = c[j] + (1.0 - c[j]) * p_2pl

        ax.plot(theta, p, 'b-', linewidth=2)
        ax.set_title(item_names[j], fontsize=10)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
        ax.axvline(x=b[j], color='red', linestyle=':', alpha=0.5, linewidth=0.5)
        ax.grid(True, alpha=0.2)

        if j >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'$\theta$', fontsize=9)
        if j % n_cols == 0:
            ax.set_ylabel('P(X=1)', fontsize=9)

    # Hide unused subplots
    for j in range(J, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Item Characteristic Curves", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig, axes


# =============================================================================
# Item Information Functions (IIFs)
# =============================================================================


def plot_iif(
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float | None = None,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    item_labels: list[str] | None = None,
    ax: "Axes | None" = None,
    title: str | None = None,
    show_legend: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot Item Information Functions (IIFs).

    I(theta) = a^2 * P(theta) * (1 - P(theta))

    Parameters
    ----------
    a : float or array-like
        Item discrimination parameter(s).
    b : float or array-like
        Item difficulty parameter(s).
    theta_range : tuple, default=(-4, 4)
        Range of theta values.
    n_points : int, default=100
        Number of points for smooth curve.
    item_labels : list[str], optional
        Labels for each item.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    show_legend : bool, default=True
        Whether to show legend.
    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    n_items = len(a)
    if c is not None:
        c = np.atleast_1d(c)
        if len(c) != len(a):
            raise ValueError(f"c must have same length as a, got {len(c)} and {len(a)}")

    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if item_labels is None:
        if n_items == 1:
            item_labels = [f"a={a[0]:.2f}, b={b[0]:.2f}"]
        else:
            item_labels = [f"Item {j+1}" for j in range(n_items)]

    colors = plt.cm.tab10.colors[:n_items]

    for j in range(n_items):
        z = a[j] * (theta - b[j])
        p_2pl = expit(z)
        if c is None:
            p = p_2pl
            info = a[j] ** 2 * p * (1 - p)
        else:
            p = c[j] + (1.0 - c[j]) * p_2pl
            dp = (1.0 - c[j]) * a[j] * p_2pl * (1 - p_2pl)
            info = (dp ** 2) / (p * (1 - p))
        ax.plot(theta, info, color=colors[j % len(colors)],
                label=item_labels[j], linewidth=2)

    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel('Information', fontsize=12)
    ax.set_xlim(theta_range)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Item Information Functions"
    ax.set_title(title, fontsize=14)

    if show_legend and n_items > 1:
        ax.legend(loc='best', fontsize=10)

    fig.tight_layout()
    return fig, ax


def plot_tif(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    theta_range: tuple[float, float] = (-4, 4),
    n_points: int = 100,
    ax: "Axes | None" = None,
    title: str | None = None,
    show_se: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot Test Information Function (TIF) and optionally Standard Error.

    TIF(theta) = sum_j I_j(theta)
    SE(theta) = 1 / sqrt(TIF(theta))

    Parameters
    ----------
    a : np.ndarray
        Item discrimination parameters, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    theta_range : tuple, default=(-4, 4)
        Range of theta values.
    n_points : int, default=100
        Number of points.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    show_se : bool, default=True
        Whether to show SE on secondary y-axis.
    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    # Compute TIF
    tif = np.zeros_like(theta)
    for j in range(len(a)):
        z = a[j] * (theta - b[j])
        p_2pl = expit(z)
        if c is None:
            p = p_2pl
            info = a[j] ** 2 * p * (1 - p)
        else:
            p = c[j] + (1.0 - c[j]) * p_2pl
            dp = (1.0 - c[j]) * a[j] * p_2pl * (1 - p_2pl)
            info = (dp ** 2) / (p * (1 - p))
        tif += info

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot TIF
    line1 = ax.plot(theta, tif, 'b-', linewidth=2, label='Test Information')
    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel('Information', fontsize=12, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_xlim(theta_range)
    ax.set_ylim(bottom=0)

    # Plot SE on secondary axis
    if show_se:
        ax2 = ax.twinx()
        se = 1 / np.sqrt(np.maximum(tif, 1e-10))
        line2 = ax2.plot(theta, se, 'r--', linewidth=2, label='Standard Error')
        ax2.set_ylabel('Standard Error', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(bottom=0)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
    else:
        ax.legend(loc='best')

    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Test Information Function"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax


# =============================================================================
# Wright Map (Item-Person Map)
# =============================================================================


def plot_wright_map(
    theta: np.ndarray,
    b: np.ndarray,
    theta_se: np.ndarray | None = None,
    item_names: list[str] | None = None,
    person_names: list[str] | None = None,
    theta_range: tuple[float, float] | None = None,
    n_bins: int = 30,
    figsize: tuple[float, float] = (10, 8),
    title: str | None = None,
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """
    Plot Wright Map (Item-Person Map).

    Shows the distribution of person abilities on the left and
    item difficulties on the right, aligned on the same scale.

    Parameters
    ----------
    theta : np.ndarray
        Person ability estimates, shape (N,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    theta_se : np.ndarray, optional
        Standard errors of theta estimates for error bars.
    item_names : list[str], optional
        Names for each item.
    person_names : list[str], optional
        Names for persons (not typically shown).
    theta_range : tuple, optional
        Range for the ability scale. Default: auto from data.
    n_bins : int, default=30
        Number of bins for person histogram.
    figsize : tuple, default=(10, 8)
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : tuple[Axes, Axes]
        Left (persons) and right (items) axes.
    """
    plt = _check_matplotlib()

    J = len(b)
    if item_names is None:
        item_names = [f"Item {j+1}" for j in range(J)]

    # Determine range
    if theta_range is None:
        all_vals = np.concatenate([theta, b])
        margin = 0.5
        theta_range = (all_vals.min() - margin, all_vals.max() + margin)

    fig, (ax_persons, ax_items) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw={'width_ratios': [1, 1.5]}
    )

    # Left panel: Person distribution (rotated histogram)
    hist_range = theta_range
    counts, bins = np.histogram(theta, bins=n_bins, range=hist_range)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Plot horizontal bars
    ax_persons.barh(bin_centers, counts, height=bins[1]-bins[0],
                   color='steelblue', alpha=0.7, edgecolor='darkblue')
    ax_persons.set_xlabel('Number of Persons', fontsize=11)
    ax_persons.set_ylabel(r'Logit Scale ($\theta$ / $b$)', fontsize=11)
    ax_persons.invert_xaxis()  # Histogram extends to the left
    ax_persons.set_ylim(theta_range)

    # Add mean and SD markers
    theta_mean = np.mean(theta)
    theta_sd = np.std(theta)
    ax_persons.axhline(y=theta_mean, color='red', linestyle='-', linewidth=2,
                      label=f'Mean: {theta_mean:.2f}')
    ax_persons.axhline(y=theta_mean + theta_sd, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)
    ax_persons.axhline(y=theta_mean - theta_sd, color='red', linestyle='--',
                      alpha=0.5, linewidth=1)

    ax_persons.set_title('Persons', fontsize=12)
    ax_persons.legend(loc='lower left', fontsize=9)

    # Right panel: Item difficulties
    # Sort items by difficulty for better visualization
    item_order = np.argsort(b)
    b_sorted = b[item_order]
    names_sorted = [item_names[i] for i in item_order]

    # Plot items as points with labels
    y_positions = b_sorted
    x_positions = np.zeros_like(b_sorted)

    ax_items.scatter(x_positions, y_positions, s=100, c='forestgreen',
                    edgecolors='darkgreen', linewidths=1.5, zorder=3)

    # Add item labels
    for i, (y, name) in enumerate(zip(y_positions, names_sorted)):
        ax_items.annotate(name, (0.05, y), fontsize=9, va='center')

    ax_items.set_xlim(-0.5, 2)
    ax_items.set_xlabel('')
    ax_items.set_xticks([])
    ax_items.set_title('Items', fontsize=12)

    # Add difficulty mean
    b_mean = np.mean(b)
    ax_items.axhline(y=b_mean, color='green', linestyle='-', linewidth=2, alpha=0.7)

    # Grid lines for comparison
    ax_items.grid(True, axis='y', alpha=0.3)
    ax_persons.grid(True, axis='y', alpha=0.3)

    if title is None:
        title = "Wright Map (Item-Person Map)"
    fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    return fig, (ax_persons, ax_items)


# =============================================================================
# Fit Statistic Plots
# =============================================================================


def plot_item_fit(
    infit_ms: np.ndarray,
    outfit_ms: np.ndarray,
    b: np.ndarray,
    item_names: list[str] | None = None,
    fit_bounds: tuple[float, float] = (0.5, 1.5),
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot item fit statistics (bubble plot).

    Parameters
    ----------
    infit_ms : np.ndarray
        Infit mean-square values, shape (J,).
    outfit_ms : np.ndarray
        Outfit mean-square values, shape (J,).
    b : np.ndarray
        Item difficulty parameters, shape (J,).
    item_names : list[str], optional
        Names for each item.
    fit_bounds : tuple, default=(0.5, 1.5)
        Acceptable range for fit statistics.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    J = len(b)
    if item_names is None:
        item_names = [f"Item {j+1}" for j in range(J)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Determine colors based on fit
    colors = []
    for inf, outf in zip(infit_ms, outfit_ms):
        if np.isnan(inf) or np.isnan(outf):
            colors.append('gray')
        elif (fit_bounds[0] <= inf <= fit_bounds[1] and
              fit_bounds[0] <= outf <= fit_bounds[1]):
            colors.append('green')
        elif inf > fit_bounds[1] or outf > fit_bounds[1]:
            colors.append('red')
        else:
            colors.append('orange')

    # Plot infit vs outfit
    scatter = ax.scatter(infit_ms, outfit_ms, c=colors, s=100,
                        edgecolors='black', linewidths=0.5, alpha=0.7)

    # Add item labels for misfitting items
    for j in range(J):
        if colors[j] in ['red', 'orange']:
            ax.annotate(item_names[j], (infit_ms[j], outfit_ms[j]),
                       fontsize=8, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')

    # Add reference lines
    ax.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=1.0, color='gray', linestyle='-', alpha=0.5)
    ax.axhline(y=fit_bounds[0], color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=fit_bounds[1], color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=fit_bounds[0], color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=fit_bounds[1], color='gray', linestyle='--', alpha=0.3)

    # Shade acceptable region
    ax.axhspan(fit_bounds[0], fit_bounds[1], alpha=0.1, color='green')
    ax.axvspan(fit_bounds[0], fit_bounds[1], alpha=0.1, color='green')

    ax.set_xlabel('Infit Mean Square', fontsize=12)
    ax.set_ylabel('Outfit Mean Square', fontsize=12)
    ax.set_xlim(0, max(2.5, np.nanmax(infit_ms) * 1.1))
    ax.set_ylim(0, max(2.5, np.nanmax(outfit_ms) * 1.1))
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Good fit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=10, label='Underfit'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Misfit'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    if title is None:
        title = "Item Fit Statistics"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax


def plot_item_fit_bars(
    infit_ms: np.ndarray,
    outfit_ms: np.ndarray,
    item_names: list[str] | None = None,
    fit_bounds: tuple[float, float] = (0.5, 1.5),
    figsize: tuple[float, float] = (12, 6),
    title: str | None = None,
) -> tuple["Figure", tuple["Axes", "Axes"]]:
    """
    Plot item fit statistics as bar charts.

    Parameters
    ----------
    infit_ms : np.ndarray
        Infit mean-square values, shape (J,).
    outfit_ms : np.ndarray
        Outfit mean-square values, shape (J,).
    item_names : list[str], optional
        Names for each item.
    fit_bounds : tuple, default=(0.5, 1.5)
        Acceptable range for fit statistics.
    figsize : tuple, default=(12, 6)
        Figure size.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : tuple[Axes, Axes]
        Infit and outfit axes.
    """
    plt = _check_matplotlib()

    J = len(infit_ms)
    if item_names is None:
        item_names = [f"Item {j+1}" for j in range(J)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(J)

    # Colors based on fit
    def get_colors(vals):
        colors = []
        for v in vals:
            if np.isnan(v):
                colors.append('gray')
            elif fit_bounds[0] <= v <= fit_bounds[1]:
                colors.append('green')
            elif v > fit_bounds[1]:
                colors.append('red')
            else:
                colors.append('orange')
        return colors

    # Infit bars
    colors_infit = get_colors(infit_ms)
    ax1.bar(x, infit_ms, color=colors_infit, edgecolor='black', alpha=0.7)
    ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=fit_bounds[0], color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=fit_bounds[1], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(item_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Infit Mean Square', fontsize=11)
    ax1.set_title('Infit', fontsize=12)
    ax1.set_ylim(0, max(2.0, np.nanmax(infit_ms) * 1.1))
    ax1.grid(True, axis='y', alpha=0.3)

    # Outfit bars
    colors_outfit = get_colors(outfit_ms)
    ax2.bar(x, outfit_ms, color=colors_outfit, edgecolor='black', alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=fit_bounds[0], color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=fit_bounds[1], color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(item_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Outfit Mean Square', fontsize=11)
    ax2.set_title('Outfit', fontsize=12)
    ax2.set_ylim(0, max(2.0, np.nanmax(outfit_ms) * 1.1))
    ax2.grid(True, axis='y', alpha=0.3)

    if title is None:
        title = "Item Fit Statistics"
    fig.suptitle(title, fontsize=14, y=1.02)

    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_person_fit(
    infit_ms: np.ndarray,
    outfit_ms: np.ndarray,
    theta: np.ndarray,
    fit_bounds: tuple[float, float] = (0.5, 1.5),
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot person fit statistics against ability.

    Parameters
    ----------
    infit_ms : np.ndarray
        Infit mean-square values, shape (N,).
    outfit_ms : np.ndarray
        Outfit mean-square values, shape (N,).
    theta : np.ndarray
        Person ability estimates, shape (N,).
    fit_bounds : tuple, default=(0.5, 1.5)
        Acceptable range for fit statistics.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple, default=(10, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Use outfit (more sensitive to outliers)
    # Color by fit status
    colors = []
    for outf in outfit_ms:
        if np.isnan(outf):
            colors.append('gray')
        elif fit_bounds[0] <= outf <= fit_bounds[1]:
            colors.append('green')
        elif outf > fit_bounds[1]:
            colors.append('red')
        else:
            colors.append('orange')

    ax.scatter(theta, outfit_ms, c=colors, s=30, alpha=0.6, edgecolors='none')

    # Reference lines
    ax.axhline(y=1.0, color='black', linestyle='-', alpha=0.5)
    ax.axhline(y=fit_bounds[0], color='gray', linestyle='--', alpha=0.3)
    ax.axhline(y=fit_bounds[1], color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel('Outfit Mean Square', fontsize=12)
    ax.set_ylim(0, min(4.0, np.nanmax(outfit_ms) * 1.1))
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Person Fit by Ability"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax


# =============================================================================
# Ability Distribution Plots
# =============================================================================


def plot_ability_distribution(
    theta: np.ndarray,
    se: np.ndarray | None = None,
    n_bins: int = 30,
    show_normal: bool = True,
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot distribution of ability estimates.

    Parameters
    ----------
    theta : np.ndarray
        Ability estimates, shape (N,).
    se : np.ndarray, optional
        Standard errors for density adjustment.
    n_bins : int, default=30
        Number of histogram bins.
    show_normal : bool, default=True
        Whether to overlay normal distribution.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()
    from scipy import stats

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Histogram
    ax.hist(theta, bins=n_bins, density=True, alpha=0.7,
            color='steelblue', edgecolor='darkblue', label='Observed')

    # Normal overlay
    if show_normal:
        theta_mean = np.mean(theta)
        theta_sd = np.std(theta)
        x = np.linspace(theta.min() - 0.5, theta.max() + 0.5, 100)
        y = stats.norm.pdf(x, theta_mean, theta_sd)
        ax.plot(x, y, 'r-', linewidth=2,
                label=f'Normal(μ={theta_mean:.2f}, σ={theta_sd:.2f})')

    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Ability Distribution"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax


def plot_se_by_theta(
    theta: np.ndarray,
    se: np.ndarray,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    c: np.ndarray | None = None,
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot standard errors by ability level.

    Parameters
    ----------
    theta : np.ndarray
        Ability estimates, shape (N,).
    se : np.ndarray
        Standard errors, shape (N,).
    a : np.ndarray, optional
        Item discriminations for theoretical SE curve.
    b : np.ndarray, optional
        Item difficulties for theoretical SE curve.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Filter out infinite SEs
    valid = np.isfinite(se)
    theta_valid = theta[valid]
    se_valid = se[valid]

    # Scatter plot of empirical SEs
    ax.scatter(theta_valid, se_valid, s=20, alpha=0.5, c='steelblue',
               label='Observed SE')

    # Theoretical SE curve if parameters provided
    if a is not None and b is not None:
        theta_range = np.linspace(theta.min() - 0.5, theta.max() + 0.5, 100)
        tif = np.zeros_like(theta_range)
        for j in range(len(a)):
            z = a[j] * (theta_range - b[j])
            p_2pl = expit(z)
            if c is None:
                p = p_2pl
                info = a[j] ** 2 * p * (1 - p)
            else:
                p = c[j] + (1.0 - c[j]) * p_2pl
                dp = (1.0 - c[j]) * a[j] * p_2pl * (1 - p_2pl)
                info = (dp ** 2) / (p * (1 - p))
            tif += info
        se_theoretical = 1 / np.sqrt(np.maximum(tif, 1e-10))
        ax.plot(theta_range, se_theoretical, 'r-', linewidth=2,
                label='Theoretical SE')

    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel('Standard Error', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    if title is None:
        title = "Standard Error by Ability"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax


# =============================================================================
# Residual Plots
# =============================================================================


def plot_residuals_by_theta(
    X: np.ndarray,
    mask_obs: np.ndarray,
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray | None = None,
    item_idx: int | None = None,
    n_groups: int = 10,
    ax: "Axes | None" = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
) -> tuple["Figure", "Axes"]:
    """
    Plot average residuals by ability groups.

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
    item_idx : int, optional
        Specific item to plot. If None, plots average across items.
    n_groups : int, default=10
        Number of ability groups.
    ax : Axes, optional
        Axes to plot on.
    title : str, optional
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    """
    plt = _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Compute expected values
    N, J = X.shape
    z = a[np.newaxis, :] * (theta[:, np.newaxis] - b[np.newaxis, :])
    p_2pl = expit(z)
    if c is None:
        E = p_2pl
    else:
        E = c[np.newaxis, :] + (1.0 - c[np.newaxis, :]) * p_2pl

    # Raw residuals
    residuals = np.where(mask_obs, X - E, np.nan)

    if item_idx is not None:
        residuals = residuals[:, item_idx:item_idx+1]
        mask_use = mask_obs[:, item_idx:item_idx+1]
    else:
        mask_use = mask_obs

    # Group by ability
    percentiles = np.linspace(0, 100, n_groups + 1)
    group_bounds = np.percentile(theta, percentiles)

    group_means = []
    residual_means = []
    residual_ses = []

    for i in range(n_groups):
        if i == n_groups - 1:
            mask = (theta >= group_bounds[i]) & (theta <= group_bounds[i+1])
        else:
            mask = (theta >= group_bounds[i]) & (theta < group_bounds[i+1])

        if mask.sum() > 0:
            group_means.append(theta[mask].mean())
            # Average residual in this group
            group_resid = residuals[mask]
            resid_flat = group_resid[~np.isnan(group_resid)]
            if len(resid_flat) > 0:
                residual_means.append(resid_flat.mean())
                residual_ses.append(resid_flat.std() / np.sqrt(len(resid_flat)))
            else:
                residual_means.append(np.nan)
                residual_ses.append(np.nan)

    group_means = np.array(group_means)
    residual_means = np.array(residual_means)
    residual_ses = np.array(residual_ses)

    # Plot
    ax.errorbar(group_means, residual_means, yerr=1.96*residual_ses,
                fmt='o-', capsize=3, color='steelblue', markersize=8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

    ax.set_xlabel(r'Ability ($\theta$)', fontsize=12)
    ax.set_ylabel('Mean Residual (Obs - Exp)', fontsize=12)
    ax.grid(True, alpha=0.3)

    if title is None:
        if item_idx is not None:
            title = f"Residuals by Ability - Item {item_idx + 1}"
        else:
            title = "Average Residuals by Ability"
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig, ax
