"""Aitchison / CLR plotting helpers.

Utilities to visualize variance structure and PC1 behaviour across a pseudocount
(`epsilon`) sweep applied before CLR transformation. Functions expect a mapping
from epsilon (float) to CLR DataFrames (rows × features).

This module focuses on concise, testable plotting helpers that return figure
and axis objects for downstream saving or embedding in notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Mapping


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------
def _total_variance(X: np.ndarray) -> float:
    """Population variance per CLR coordinate, summed across features (ddof=0)."""
    return float(np.var(X, axis=0, ddof=0).sum())


def _pc1_variance_ratio(X: np.ndarray) -> float:
    """Return the share of total centered variance explained by the first PC."""
    Xc = X - X.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.nan
    sv_total = np.sum(s ** 2)
    # s contains singular values (sqrt of explained variance per component).
    # The squared singular values sum to the total variance in the centered data.
    return float(s[0] ** 2 / sv_total) if sv_total > 0 else np.nan


def _safe_metrics(X: np.ndarray) -> tuple[float, float]:
    """Return variance metrics for valid numeric arrays; otherwise yield NaNs."""
    if X.size == 0 or not np.isfinite(X).all():
        return np.nan, np.nan
    # Return (total Aitchison variance, PC1 fraction). These are cheap to compute
    # and tolerantly return NaN for degenerate inputs.
    return _total_variance(X), _pc1_variance_ratio(X)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def _compute_variance_profile(clr_data: Mapping[float, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute total Aitchison variance and PC1 variance ratio for each epsilon.

    Parameters
    ----------
    clr_data : Mapping[float, pd.DataFrame]
        Mapping from epsilon to CLR-transformed DataFrames.

    Returns
    -------
    pd.DataFrame
        Float-indexed by eps, columns: total_variance, pc1_variance_ratio.
        Index is guaranteed to be sorted float.
    """
    if clr_data is None or len(clr_data) == 0:
        raise ValueError("clr_data is empty.")

    records = []
    for eps in sorted(clr_data):
        df = clr_data[eps]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"clr_data[{eps}] must be a DataFrame, got {type(df)}.")
        # Work on raw numpy values to avoid pandas overhead in the hot loop.
        # Cast to float to ensure numeric stability for SVD / variance ops.
        total_var, pc1_ratio = _safe_metrics(df.values.astype(float))
        records.append((eps, total_var, pc1_ratio))

    return (
        pd.DataFrame.from_records(records, columns=["eps", "total_variance", "pc1_variance_ratio"])
        .set_index("eps")
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def _nearest_row(profile: pd.DataFrame, eps: float) -> tuple[float, pd.Series]:
    """Snap an epsilon to the nearest row in the plotted profile."""
    if profile.empty:
        raise ValueError("Cannot annotate: variance profile is empty.")
    idx = profile.index.get_indexer([eps], method="nearest")[0]
    return float(profile.index[idx]), profile.iloc[idx]


def _annotate_chosen(ax: plt.Axes, profile: pd.DataFrame, chosen_eps: float) -> None:
    """Annotate the selected epsilon directly on the variance plot."""
    snapped_eps, row = _nearest_row(profile, chosen_eps)
    text = (
        f"ε={snapped_eps:.2g}\n"
        f"var={row.total_variance:.3g}\n"
        f"pc1={row.pc1_variance_ratio:.2%}"
    )
    ax.annotate(
        text,
        xy=(snapped_eps, row.total_variance),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", alpha=0.9),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
_BLUE  = "#185FA5"
_CORAL = "#D85A30"
_GREEN = "#3B6D11"


def _configure_axes(ax1: plt.Axes, ax2: plt.Axes) -> None:
    """Apply shared axis labels and colors for the twin-axis plot."""
    ax1.set_xscale("log")
    ax1.set_xlabel("ε (pseudocount)")
    ax1.set_ylabel("Total Aitchison variance", color=_BLUE)
    ax1.tick_params(axis="y", labelcolor=_BLUE)
    ax2.set_ylabel("Variance explained by PC1", color=_CORAL)
    ax2.tick_params(axis="y", labelcolor=_CORAL)


def _plot_series(
    ax1: plt.Axes,
    ax2: plt.Axes,
    profile: pd.DataFrame,
) -> dict:
    """Plot the two diagnostic series and return handles for a combined legend."""
    l1, = ax1.plot(
        profile.index, profile["total_variance"],
        marker="o", color=_BLUE, label="Total variance",
    )
    l2, = ax2.plot(
        profile.index, profile["pc1_variance_ratio"],
        marker="s", linestyle="--", color=_CORAL, label="PC1 variance ratio",
    )
    return l1, l2


def _plot_chosen_line(ax: plt.Axes, chosen_eps: float) -> plt.Line2D:
    """Draw the selected epsilon marker used by the annotation and legend."""
    return ax.axvline(
        chosen_eps,
        linestyle=":",
        color=_GREEN,
        label=f"Chosen ε ({chosen_eps:.2g})",
    )


# ---------------------------------------------------------------------------
# 1. Main plotting function
# ---------------------------------------------------------------------------
def plot_aitchison(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> dict:
    """
    Plot total Aitchison variance and PC1 variance ratio across epsilon values.

    Parameters
    ----------
    clr_data   : Mapping[float, pd.DataFrame]
        Mapping from epsilon to CLR-transformed DataFrames.
    chosen_eps : if provided, mark with a vertical line and optional annotation
    annotate   : if True and chosen_eps is set, annotate metrics at that point

    Returns
    -------
    dict
        Dictionary with keys ``figure``, ``ax1``, and ``ax2``.
    """
    profile = _compute_variance_profile(clr_data)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    _configure_axes(ax1, ax2)
    l1, l2 = _plot_series(ax1, ax2, profile)

    handles = [l1, l2]
    if chosen_eps is not None:
        # Mark the chosen epsilon on the plot and optionally annotate the local metrics.
        handles.append(_plot_chosen_line(ax1, chosen_eps))
        if annotate:
            _annotate_chosen(ax1, profile, chosen_eps)

    ax1.legend(handles=handles, loc="upper right")
    ax1.set_title("CLR variance structure vs $\\varepsilon$: total variance and PC1 concentration", pad=12)
    fig.tight_layout()
    return {'figure': fig, 'ax1': ax1, 'ax2': ax2}


# ---------------------------------------------------------------------------
# 2. PC1 loadings plot
# ---------------------------------------------------------------------------
def plot_pc1_loadings(
    clr_data: pd.DataFrame,
    chosen_eps: float,
    figsize: tuple[float, float] = (10, 8),
    label_fmt: str = "%.3f",
) -> dict:
    """Plot PC1 loadings for a selected epsilon.

    Parameters
    ----------
    clr_data : pd.DataFrame
        The CLR-transformed DataFrame (T × K).
    chosen_eps : float
        The epsilon value to visualize and report in the plot title.
    figsize : tuple, optional
        Figure size in inches.
    label_fmt : str, optional
        Format string used to label each bar.

    Returns
    -------
    dict
        Contains keys "figure" and "axis" for the created plot.

    Raises
    ------
    ValueError
        If the CLR matrix is degenerate (fewer than 2 rows) or SVD fails.
    """

    # Convert to numeric numpy array for SVD; assert sufficient rows
    X = clr_data.to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        raise ValueError("Need at least 2 rows to compute PC1 reliably")
    # Center columns (features) before SVD
    Xc = X - X.mean(axis=0)
    try:
        _, singular_vals, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"SVD failed: {e}")

    # Fraction of total variance explained by PC1 (squared singular values sum to total)
    sv_sum = np.sum(singular_vals ** 2)
    pc1_variance_ratio = float((singular_vals[0] ** 2) / sv_sum) if sv_sum > 0 else float('nan')

    # PC1 direction (note: sign is arbitrary). Flip sign so the largest-magnitude
    # coefficient is positive to improve consistency across plots.
    pc1 = Vt[0].copy()
    pc1 *= np.sign(pc1[np.argmax(np.abs(pc1))])
    pc1_loadings = pd.Series(pc1, index=clr_data.columns, name="pc1_loading").sort_values()
    # Color positive/negative loadings consistently
    bar_colors = np.where(pc1_loadings.values >= 0, "#D85A30", "#185FA5")
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        y=pc1_loadings.index,
        width=pc1_loadings.values,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar_label(bars, fmt=label_fmt, padding=3, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=1.0, zorder=3)
    ax.set_xlabel("PC1 Loading Magnitude", fontweight="bold")
    ax.set_ylabel("Crime Features", fontweight="bold", labelpad=36)
    ax.tick_params(axis="y", pad=36, length=0)
    ax.set_title(
        f"PC1 Loadings at $\\varepsilon$ = {chosen_eps:.2g}\n"
        f"Explained Variance: {pc1_variance_ratio:.1%}",
        fontweight="bold", pad=18,
    )
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.7, color="#CCCCCC")
    ax.yaxis.grid(True, linestyle=":", alpha=0.3, color="#DDDDDD")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Inform users that PC1 sign is arbitrary and has been normalized for display
    fig.text(
        0.5, -0.02,
        "* PC1 sign is arbitrary; orientation normalized for display.",
        ha="center", fontsize=10, color="gray", style="italic",
    )
    fig.tight_layout()
    return {"figure": fig, "axis": ax}


# ---------------------------------------------------------------------------
# 3. PC1 stability plot
# ---------------------------------------------------------------------------
def plot_pc1_loading_stability(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    top_n: int | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> dict:
    """Plot PC1 loading trajectories across epsilons to assess stability.

    Computes PC1 for each epsilon's CLR matrix and aligns signs to an anchor PC1
    computed at a reference epsilon (either `chosen_eps` or the smallest eps).

    Parameters
    ----------
    clr_data : Mapping[float, pd.DataFrame]
        Mapping from epsilon to CLR DataFrames.
    chosen_eps : float | None
        If provided, used as the reference for sign alignment and marked on plot.
    top_n : int | None
        If provided, only the `top_n` most variable features (by PC1 range) are plotted.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    dict
        Keys: "figure", "axis" for downstream use.
    """
    if clr_data is None or len(clr_data) == 0:
        raise ValueError("clr_data is empty")

    eps_values = sorted(clr_data)
    # Choose a reference epsilon for anchor orientation
    ref_eps = chosen_eps if chosen_eps is not None else eps_values[0]

    # Build anchored PC1 directions: compute SVD at reference to find anchor index
    ref_X = clr_data[ref_eps].values.astype(float)
    ref_Xc = ref_X - ref_X.mean(axis=0)
    _, _, Vt_ref = np.linalg.svd(ref_Xc, full_matrices=False)
    anchor_idx = np.argmax(np.abs(Vt_ref[0]))

    pc1_rows = []
    # Iterate eps values computing PC1 and aligning sign to the reference anchor
    for eps in eps_values:
        X = clr_data[eps].values.astype(float)
        Xc = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        # Align sign so that the feature at anchor_idx keeps consistent orientation
        pc1_rows.append(Vt[0] * np.sign(Vt[0][anchor_idx]))

    df_loadings = pd.DataFrame(
        pc1_rows,
        index=eps_values,
        columns=clr_data[ref_eps].columns,
    )

    # Optionally reduce to the most variable features across epsilons
    if top_n is not None:
        variability = df_loadings.max() - df_loadings.min()
        df_loadings = df_loadings[variability.nlargest(top_n).index]

    # Plot trajectories of PC1 loadings vs epsilon
    fig, ax = plt.subplots(figsize=figsize)
    for feature in df_loadings.columns:
        ax.plot(df_loadings.index, df_loadings[feature], marker="o", markersize=3, label=feature)
    if chosen_eps is not None:
        ax.axvline(chosen_eps, linestyle=":", color="#3B6D11", label=f"Chosen $\\varepsilon$ ({chosen_eps:.2g})")
    ax.set_xscale("log")
    ax.set_xlabel("$\\varepsilon$ (pseudocount)")
    ax.set_ylabel("PC1 loading")
    ax.set_title("PC1 loading stability across $\\varepsilon$")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="x", color="#CCCCCC", linestyle="-", linewidth=0.6, alpha=0.5)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.text(
        0.5, -0.02,
        "* PC1 sign is arbitrary; orientation normalized for display.",
        ha="center", fontsize=10, color="gray", style="italic",
    )
    fig.tight_layout()
    return {"figure": fig, "axis": ax}