"""Plot Aitchison variance diagnostics for CLR-transformed epsilon sweeps.

The module turns a dictionary of CLR matrices into a variance profile and plots
total Aitchison variance alongside the fraction explained by the first principal
component. It is intentionally small and focused so the plot can be reused in
notebooks and scripts without extra orchestration code.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
    return float(s[0] ** 2 / sv_total) if sv_total > 0 else np.nan


def _safe_metrics(X: np.ndarray) -> tuple[float, float]:
    """Return variance metrics for valid numeric arrays; otherwise yield NaNs."""
    if X.size == 0 or not np.isfinite(X).all():
        return np.nan, np.nan
    return _total_variance(X), _pc1_variance_ratio(X)


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_variance_profile(clr_dict: dict[float, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute total Aitchison variance and PC1 variance ratio for each epsilon.

    Parameters
    ----------
    clr_dict : dict mapping eps (float) -> CLR DataFrame

    Returns
    -------
    pd.DataFrame
        Float-indexed by eps, columns: total_variance, pc1_variance_ratio.
        Index is guaranteed to be sorted float.
    """
    if not clr_dict:
        raise ValueError("clr_dict is empty.")

    records = []
    for eps in sorted(clr_dict):
        df = clr_dict[eps]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"clr_dict[{eps}] must be a DataFrame, got {type(df)}.")
        # Work on raw values here because the diagnostics are purely numeric.
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
) -> tuple[plt.Line2D, plt.Line2D]:
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
# Main plotting function
# ---------------------------------------------------------------------------
def plot_aitchison(
    clr_dict: dict[float, pd.DataFrame],
    chosen_eps: float | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] = (10, 5),
) -> tuple[Figure, plt.Axes, plt.Axes]:
    """
    Plot total Aitchison variance and PC1 variance ratio across epsilon values.

    Parameters
    ----------
    clr_dict   : dict mapping eps -> CLR DataFrame
    chosen_eps : if provided, mark with a vertical line and optional annotation
    annotate   : if True and chosen_eps is set, annotate metrics at that point

    Returns
    -------
    tuple[Figure, plt.Axes, plt.Axes]
        The figure and both axes objects so callers can further customize or save.
    """
    profile = compute_variance_profile(clr_dict)

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
    ax1.set_title("CLR variance structure vs ε: total variance and PC1 concentration", pad=12)
    fig.tight_layout()
    return fig, ax1, ax2