"""Rare-event crime extraction and plotting helpers.

Utilities for separating rare-event crimes into binary presence/absence
variables and plotting their monthly occurrence patterns.

Public API
----------
extract_rare_events      - pure data extraction; no side-effects
plot_rare_event_grid     - individual bar chart per crime
plot_rare_event_combined - combined co-occurrence state chart across crimes
extract_rare_event_presence_dict - convenience wrapper (backwards-compat shim)
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _month_widths(index: pd.DatetimeIndex) -> List[int]:
    """Return the number of days in each month for use as bar widths."""
    return [(d + pd.offsets.MonthBegin(1) - d).days for d in index]


def _configure_date_axis(ax: plt.Axes) -> None:
    """Attach AutoDateLocator + ConciseDateFormatter and rotate tick labels."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="x", rotation=45, labelsize=8)


def _fill_runs(ax: plt.Axes, values: pd.Series) -> None:
    """Shade each contiguous run of True values in *values* with fill_between."""
    run_start = None
    for date, present in values.items():
        if present and run_start is None:
            run_start = date
        elif not present and run_start is not None:
            ax.fill_between(
                [run_start, date], 0, 1,
                color="steelblue", alpha=0.85, linewidth=0,
            )
            run_start = None
    if run_start is not None:
        ax.fill_between(
            [run_start, values.index[-1] + pd.offsets.MonthBegin(1)], 0, 1,
            color="steelblue", alpha=0.85, linewidth=0,
        )


def _plot_two_crime_state(
    ax: plt.Axes,
    combined: pd.DataFrame,
    ordered_crimes: List[str],
    colors: List,
) -> List[Patch]:
    """
    Four-state color chart for exactly two crimes.

    Each month bar is colored by which combination of the two crimes is
    present: first only, second only, both, or neither.

    Returns legend handles.
    """
    first, second = ordered_crimes[0], ordered_crimes[1]
    c_first   = colors[0]
    c_second  = colors[1]
    c_both    = "#7B2D8B"
    c_neither = "white"

    state_color = []
    for date in combined.index:
        a = bool(combined.loc[date, first])
        b = bool(combined.loc[date, second])
        if a and b:
            state_color.append(c_both)
        elif a:
            state_color.append(c_first)
        elif b:
            state_color.append(c_second)
        else:
            state_color.append(c_neither)

    ax.bar(
        combined.index, 1,
        width=_month_widths(combined.index),
        color=state_color,
        align="edge",
        linewidth=0,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("State")

    return [
        Patch(color=c_first,  label=first),
        Patch(color=c_second, label=second),
        Patch(color=c_both,   label="Both present"),
        Patch(facecolor=c_neither, edgecolor="lightgray",
              linewidth=0.5, label="Neither"),
    ]


def _plot_heatmap(
    ax: plt.Axes,
    combined: pd.DataFrame,
    ordered_crimes: List[str],
    colors: List,
) -> List[Patch]:
    """
    Per-crime heatmap rows for 3+ crimes.

    Each crime occupies one horizontal row; cells are filled when the crime
    is present and white when absent.

    Returns legend handles.
    """
    n = len(ordered_crimes)
    widths = _month_widths(combined.index)

    for row_idx, crime in enumerate(ordered_crimes):
        cell_colors = [
            colors[row_idx % len(colors)] if combined.loc[date, crime] else "white"
            for date in combined.index
        ]
        ax.bar(
            combined.index, 1,
            width=widths,
            bottom=row_idx,
            color=cell_colors,
            align="edge",
            linewidth=0,
        )

    ax.set_ylim(0, n)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(ordered_crimes, fontsize=7)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylabel("")

    return [
        Patch(color=colors[i % len(colors)], label=crime)
        for i, crime in enumerate(ordered_crimes)
    ] + [
        Patch(facecolor="white", edgecolor="lightgray",
              linewidth=0.5, label="Absent"),
    ]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_rare_events(
    results_df: pd.DataFrame,
    crime_names: List[str],
) -> Dict:
    """
    Separate rare-event crimes from *results_df* into binary presence/absence
    variables, returning them alongside the remaining count-based crimes.

    Rare events are modelled as binary (present/absent) rather than as counts
    because their extreme sparsity makes count-based representations
    uninformative.  The remaining crimes are returned as-is for the caller
    to transform as appropriate (e.g. CLR, log-counts).

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ``fbi_code_desc``, ``year_month``, ``crime_count``.
    crime_names : list[str]
        Ordered list of crime names to treat as rare events.

    Returns
    -------
    dict with keys:
        binary_tables : dict[str, pd.DataFrame]
            One DataFrame per crime, indexed by a shared ``DatetimeIndex``
            (monthly frequency).  Single column ``present`` (bool).
        df_rare       : pd.DataFrame
            Rare-event rows only, with an added boolean ``present`` column.
            Columns: ``fbi_code_desc``, ``year_month``, ``crime_count``, ``present``.
        df_filtered   : pd.DataFrame
            Copy of *results_df* with rare-event rows removed.
        summary       : pd.DataFrame
            Per-crime presence/absence rates and counts.
        ordered_crimes : list[str]
            Validated, ordered subset of *crime_names* that exist in the data.
    """
    available = set(results_df["fbi_code_desc"].unique())
    missing = set(crime_names) - available
    for name in missing:
        warnings.warn(f"'{name}' not found in data — skipping", UserWarning, stacklevel=2)

    ordered_crimes = [c for c in crime_names if c in available]
    if not ordered_crimes:
        raise ValueError("None of the provided crime_names exist in the data.")

    rare_mask = results_df["fbi_code_desc"].isin(set(ordered_crimes))
    rare_df = results_df.loc[
        rare_mask, ["fbi_code_desc", "year_month", "crime_count"]
    ].copy()
    rare_df["year_month"] = pd.to_datetime(rare_df["year_month"])
    rare_df["present"] = rare_df["crime_count"] > 0
    rare_df.sort_values(["fbi_code_desc", "year_month"], inplace=True)

    # Shared DatetimeIndex so every binary table has the same x-axis.
    months_index = pd.date_range(
        start=rare_df["year_month"].min(),
        end=rare_df["year_month"].max(),
        freq="MS",
    )

    grouped = rare_df.groupby("fbi_code_desc", sort=False, observed=True)
    binary_tables: Dict[str, pd.DataFrame] = {}
    for crime in ordered_crimes:
        grp = (
            grouped.get_group(crime)[["year_month", "present"]]
            .set_index("year_month")
            .copy()
        )
        binary_tables[crime] = grp.reindex(months_index).fillna(False)

    df_filtered = results_df.loc[~rare_mask].copy()

    summary_raw = (
        rare_df.groupby("fbi_code_desc", observed=True)["present"]
        .agg(n_months="count", n_present="sum", presence_rate="mean")
    )
    summary = summary_raw.assign(
        n_absent=summary_raw["n_months"] - summary_raw["n_present"],
        absence_rate=1 - summary_raw["presence_rate"],
    ).rename_axis("crime_name")

    return {
        "binary_tables":  binary_tables,
        "df_rare":        rare_df,
        "df_filtered":    df_filtered,
        "summary":        summary,
        "ordered_crimes": ordered_crimes,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rare_event_grid(
    binary_tables: Dict[str, pd.DataFrame],
    ordered_crimes: List[str],
    summary: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 3),
    ncols: int = 2,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Render a grid of binary bar charts — one subplot per crime.

    Parameters
    ----------
    binary_tables : dict[str, pd.DataFrame]
        Output of :func:`extract_rare_events`.
    ordered_crimes : list[str]
        Display order for subplots.
    summary : pd.DataFrame
        Output of :func:`extract_rare_events`; presence/absence stats shown in titles.
    figsize : (width, height_per_row)
        Height is multiplied by the number of rows.
    ncols : int
        Number of columns in the grid.

    Returns
    -------
    fig : plt.Figure
    axes_flat : list[plt.Axes]  (flattened, length == ncols * nrows)
    """
    n = len(ordered_crimes)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize[0], figsize[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, crime in zip(axes_flat, ordered_crimes):
        tbl = binary_tables[crime]
        values = tbl["present"].astype(bool)

        _fill_runs(ax, values)

        row = summary.loc[crime]
        ax.set_title(
            f"{crime}\n"
            f"presence={row['presence_rate']:.1%}  "
            f"n_absent={int(row['n_absent'])}  "
            f"absence={row['absence_rate']:.1%}",
            fontsize=8,
        )
        ax.set_xlim(values.index[0], values.index[-1] + pd.offsets.MonthBegin(1))
        ax.set_ylim(-0.05, 1.15)
        ax.set_yticks([0, 1])
        ax.set_ylabel("Present")
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        _configure_date_axis(ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes_flat.tolist()


def plot_rare_event_combined(
    binary_tables: Dict[str, pd.DataFrame],
    ordered_crimes: List[str],
    figsize: Tuple[int, int] = (14, 4),
) -> Tuple[plt.Figure | None, plt.Axes | None]:
    """
    Render a chart showing presence of all rare crimes over time.

    With two crimes, produces a four-state color bar chart.
    With three or more, produces a per-crime heatmap.

    Parameters
    ----------
    binary_tables : dict[str, pd.DataFrame]
        Output of :func:`extract_rare_events`.
    ordered_crimes : list[str]
        Column order for the chart.
    figsize : (width, height)
        Figure size in inches.

    Returns
    -------
    fig : plt.Figure
    ax  : plt.Axes
        Returns ``(None, None)`` when fewer than 2 crimes are supplied.
    """
    if len(ordered_crimes) < 2:
        warnings.warn(
            "plot_rare_event_combined requires at least 2 crimes — skipping. "
            "Use the grid plot to visualise a single crime.",
            UserWarning,
            stacklevel=2,
        )
        return None, None

    combined = pd.concat(
        [binary_tables[c]["present"].astype(int).rename(c) for c in ordered_crimes],
        axis=1,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=figsize)

    if len(ordered_crimes) == 2:
        legend_handles = _plot_two_crime_state(ax, combined, ordered_crimes, colors)
    else:
        legend_handles = _plot_heatmap(ax, combined, ordered_crimes, colors)

    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=min(len(legend_handles), 4),
        fontsize=8,
        frameon=True,
    )
    ax.set_xlim(combined.index[0], combined.index[-1] + pd.offsets.MonthBegin(1))
    ax.set_title("Rare-Event / Removed Crimes — Monthly Co-occurrence State", pad=40)
    ax.set_xlabel("Year-Month")
    _configure_date_axis(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    return fig, ax


# ---------------------------------------------------------------------------
# Convenience wrapper (backwards-compatible shim)
# ---------------------------------------------------------------------------

def extract_rare_event_presence_dict(
    filled_results: dict,
    crime_names: List[str],
    plot_mode: str = "both",
    figsize: Tuple[int, int] = (14, 4),
    ncols: int = 2,
    combined_height: int = 4,
) -> Dict:
    """
    Backwards-compatible wrapper around the three focused functions above.

    Parameters
    ----------
    filled_results : dict
        Must contain key ``"filled_df"`` holding the source DataFrame.
    crime_names : list[str]
        Ordered list of rare-event crime names.
    plot_mode : {"both", "grid", "combined", "none"}
        Controls which figures are produced.
    figsize : (width, height_per_row)
        Passed to :func:`plot_rare_event_grid`.
    ncols : int
        Grid columns for the grid plot.
    combined_height : int
        Figure height (in inches) for the combined chart.

    Returns
    -------
    dict with keys:
        binary_tables, df_rare, df_filtered, summary,
        fig_grid, axes_grid, fig_combined, ax_combined
    """
    _VALID_MODES = {"both", "grid", "combined", "none"}
    if plot_mode not in _VALID_MODES:
        raise ValueError(f"plot_mode must be one of {_VALID_MODES!r}")

    data = extract_rare_events(filled_results["filled_df"], crime_names)
    binary_tables  = data["binary_tables"]
    ordered_crimes = data["ordered_crimes"]

    fig_grid = axes_grid = fig_combined = ax_combined = None

    if plot_mode in {"both", "grid"}:
        fig_grid, axes_grid = plot_rare_event_grid(
            binary_tables, ordered_crimes, data["summary"],
            figsize=figsize, ncols=ncols,
        )

    if plot_mode in {"both", "combined"} and len(ordered_crimes) >= 2:
        fig_combined, ax_combined = plot_rare_event_combined(
            binary_tables, ordered_crimes,
            figsize=(figsize[0], combined_height),
        )

    return {
        "binary_tables": binary_tables,
        "df_rare":       data["df_rare"],
        "df_filtered":   data["df_filtered"],
        "summary":       data["summary"],
        "fig_grid":      fig_grid,
        "axes_grid":     axes_grid,
        "fig_combined":  fig_combined,
        "ax_combined":   ax_combined,
    }