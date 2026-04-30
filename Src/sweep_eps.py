"""
CLR Pseudocount Sweep and Diagnostics

This module provides a comprehensive pipeline for sweeping pseudo-count epsilon values
across a CLR (Centered Log-Ratio) transformation and computing diagnostic metrics to
guide optimal eps selection.

Key functions:
- sweep_eps_grid: Main orchestrator; loops over eps, computes CLR and rank diagnostics,
  optionally plots and auto-selects.
- _compute_rank_diagnostics: Captures rank stability and entropy metrics across eps.
- _auto_select: Implements a 3-stage eps selection strategy balancing CLR validity
  and rank structure preservation.
- _plot_diagnostics: Visualizes diagnostics and optionally near-zero diagnostics.
"""

from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau


def sweep_eps_grid(
    pivot: pd.DataFrame,
    eps_grid: Iterable[float],
    large_clr_threshold: float = 10.0,
    plot: bool = True,
    auto_select: bool = False,
    verbose: bool = True,
    kendall_threshold: float = 0.98,
    spearman_threshold: float = 0.999,
    near_zero_threshold: float | None = 1e-6,
) -> dict[str, Any]:
    """
    Sweep over epsilon (pseudo-count) values and compute CLR diagnostics.

    Computes CLR transformation and rank stability metrics for each eps in the grid,
    optionally plots diagnostics, and performs automated eps selection if requested.

    Parameters
    ----------
    pivot : pd.DataFrame
        (T, K) count matrix where T is time intervals and K is feature types.
        Must have at least 2 rows for rank correlation computation.
    eps_grid : Iterable[float]
        Pseudo-count values to evaluate. Must be positive and non-duplicating.
    large_clr_threshold : float, default=10.0
        Upper bound for acceptable max |CLR| values; used to flag sensitivity.
    plot : bool, default=True
        If True, generate and display diagnostics plot.
    auto_select : bool, default=False
        If True, apply 3-stage automated eps selection and populate meta["chosen_eps"].
    verbose : bool, default=True
        If True, print diagnostics_df to stdout.
    kendall_threshold : float, default=0.98
        Minimum Kendall tau for rank stability to satisfy constraint C2.
    spearman_threshold : float, default=0.999
        Fallback Spearman rho threshold if kendall_threshold is unachievable.
    near_zero_threshold : float or None, default=1e-6
        If not None, compute per-eps "near-zero" diagnostics using this absolute threshold
        on the per-row probabilities (after adding eps and row-normalizing). Set to None to
        omit these diagnostics entirely.

    Returns
    -------
    dict[str, Any]
        Keys:
        - diagnostics_df: pd.DataFrame indexed by eps with computed metrics
        - clr_dict: dict mapping eps -> CLR DataFrame (for external use)
        - meta: dict containing auto-selection results (always present; minimal when auto_select=False)
        - fig: matplotlib Figure object (None if plot=False)

    Notes
    -----
    - `diagnostics_df` is indexed by the evaluated `eps` values (float) and is sorted
      in ascending order.
    """
    if pivot.shape[0] < 2:
        raise ValueError(
            f"pivot must have at least 2 rows to compute rank diagnostics; got shape {pivot.shape}"
        )

    eps_list = [float(e) for e in eps_grid]
    if any(e <= 0.0 for e in eps_list):
        raise ValueError("eps_grid contains non-positive values; all eps must be > 0")

    # deduplicate while preserving sorted order
    eps_arr = np.array(sorted(set(eps_list)))

    T, K = pivot.shape
    n_zero_cells = int((pivot == 0).sum().sum())
    # baseline mask of exact zeros before smoothing; used to measure contribution
    # of original zeros to CLR variance as eps grows.
    zero_mask_pre = (pivot == 0).values

    diagnostics: list[dict[str, Any]] = []
    clr_dict: dict[float, pd.DataFrame] = {}
    prev_clr = None

    for eps in eps_arr:
        props = pivot + eps
        props = props.div(props.sum(axis=1), axis=0)
        logp = np.log(props)
        clr = logp.sub(logp.mean(axis=1), axis=0)
        abs_clr = np.abs(clr.values)

        clr_var_total = float(np.var(abs_clr))
        clr_var_zeros = float(np.var(abs_clr[zero_mask_pre])) if zero_mask_pre.any() else 0.0

        # optional per-eps near-zero diagnostics
        if near_zero_threshold is not None:
            thresh = float(near_zero_threshold)
            near_mask = (props.values < thresh)
            pct_cells_near_zero = float(near_mask.mean()) * 100.0
            clr_var_near_zero = float(np.var(abs_clr[near_mask])) if near_mask.any() else 0.0
        else:
            pct_cells_near_zero = None
            clr_var_near_zero = None

        rank_diag = _compute_rank_diagnostics(clr, prev_clr)
        prev_clr = clr

        diagnostics.append(
            {
                "eps": eps,
                "max_abs_clr": float(np.nanmax(abs_clr)),
                "mean_max_abs_clr": float(np.nanmean(np.nanmax(abs_clr, axis=1))),
                "pct_rows_large_clr": float((np.nanmax(abs_clr, axis=1) > large_clr_threshold).mean()) * 100.0,
                "rank_stability_spearman": rank_diag["rank_stability_spearman"],
                "rank_stability_kendall": rank_diag["rank_stability_kendall"],
                "rank_unique_ratio": rank_diag["rank_unique_ratio"],
                "rank_entropy": rank_diag["rank_entropy"],
                "zero_contribution_ratio": clr_var_zeros / clr_var_total if clr_var_total > 1e-12 else 0.0,
                "n_zero_cells_pre_smooth": n_zero_cells,
                "T": T,
                "K": K,
            }
        )

        if near_zero_threshold is not None:
            diagnostics[-1]["pct_cells_near_zero"] = pct_cells_near_zero
            diagnostics[-1]["clr_var_near_zero"] = clr_var_near_zero

        clr_dict[eps] = clr

    diagnostics_df = pd.DataFrame(diagnostics).set_index("eps")

    # meta: always present; when auto_select is False it is minimal
    if auto_select:
        meta = _auto_select(diagnostics_df, kendall_threshold, spearman_threshold)
    else:
        meta = {
            "auto_select": False,
            "chosen_eps": None,
            "chosen_reason": None,
            "chosen_msg": None,
            "chosen_row": None,
        }

    fig = None
    if plot:
        fig = _plot_diagnostics(diagnostics_df, large_clr_threshold, kendall_threshold, meta.get("chosen_eps"))
        plt.show()

    if verbose:
        display_df = diagnostics_df.drop(columns=["T", "K", "rank_unique_ratio", "pct_cells_near_zero", "n_zero_cells_pre_smooth"], errors="ignore").reset_index()
        print(display_df.to_string(index=False))

    return {"diagnostics_df": diagnostics_df, "clr_dict": clr_dict, "meta": meta, "fig": fig}


def _compute_rank_diagnostics(clr: pd.DataFrame, prev_clr: pd.DataFrame | None) -> dict[str, float]:
    """
    Compute rank-order stability metrics comparing current CLR to previous eps.
    """
    ranks = clr.rank(axis=0, method="min").values.astype(int)
    n_rows, n_cols = ranks.shape

    if prev_clr is not None:
        prev_ranks = prev_clr.rank(axis=0, method="min").values.astype(int)
        rho, _ = spearmanr(clr.values.ravel(), prev_clr.values.ravel())
        tau, _ = kendalltau(ranks.ravel(), prev_ranks.ravel())
        spearman, kendall = float(rho), float(tau)
    else:
        spearman = kendall = float("nan")

    unique_ratios, entropies = [], []
    for col in range(n_cols):
        counts = np.bincount(ranks[:, col])
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        unique_ratios.append(len(counts) / n_rows)
        entropies.append(float(-np.sum(probs * np.log2(probs + 1e-12))))

    return {
        "rank_stability_spearman": spearman,
        "rank_stability_kendall": kendall,
        "rank_unique_ratio": float(np.mean(unique_ratios)),
        "rank_entropy": float(np.mean(entropies)),
    }


def _plot_diagnostics(
    data_df: pd.DataFrame, large_clr_threshold: float, kendall_threshold: float, chosen_eps: float | None = None
) -> plt.Figure:
    """
    Create diagnostic plots for the eps sweep.

    Renders a 2x3 layout (2 rows, 3 columns) with 6 panels for eps diagnostics.
    Panels without computed data display "not computed".
    """
    T = int(data_df["T"].iloc[0])
    K = int(data_df["K"].iloc[0])
    nz = int(data_df["n_zero_cells_pre_smooth"].iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (axes[0, 0], "max_abs_clr", "#2c3e50", "max |CLR|", "Sensitivity: Max |CLR|"),
        (axes[0, 1], "pct_rows_large_clr", "#e74c3c", "% of Observations", f"Sparsity Impact: % Rows > {large_clr_threshold}"),
        (axes[0, 2], "rank_stability_kendall", "#8e44ad", "Kendall τ", "Rank Stability (Kendall)"),
        (axes[1, 0], "rank_unique_ratio", "#e67e22", "Unique Ratio", "Rank Collapse Detection"),
        (axes[1, 1], "pct_cells_near_zero", "#3498db", "% Cells < thr", "Near-zero Cells (% below threshold)"),
        (axes[1, 2], "clr_var_near_zero", "#16a085", "CLR Var", "CLR Variance from Near-zero Cells"),
    ]

    for ax, col, color, ylabel, title in panels:
        if col in data_df.columns:
            data_df[col].plot(marker="o", color=color, ax=ax)
        else:
            ax.text(0.5, 0.5, f"{col} not computed", ha="center", va="center", alpha=0.6)
        ax.set(xscale="log", title=title, xlabel="ε (Pseudocount)", ylabel=ylabel)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        if chosen_eps is not None:
            ax.axvline(chosen_eps, color="green", linewidth=1.2, linestyle="--", label=f"chosen ε={chosen_eps}")
            ax.legend(fontsize=9)

    # reference lines
    try:
        axes[0, 1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axes[0, 2].axhline(kendall_threshold, color="orange", linewidth=0.8, linestyle=":", label=f"τ={kendall_threshold}")
        axes[0, 2].axhline(1.0, color="black", linewidth=0.8, alpha=0.5, label="τ=1.0")
        axes[0, 2].legend(fontsize=8)
        axes[1, 0].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    except Exception:
        pass

    fig.suptitle(f"ε-Sweep Diagnostics\nData: {T} intervals × {K} types | Pre-smooth Zeros: {nz}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.4)
    return fig


def _auto_select(data_df: pd.DataFrame, kendall_threshold: float, spearman_threshold: float) -> dict[str, Any]:
    """
    Automatic eps selection given diagnostic metrics and thresholds.
    """
    if data_df.empty:
        raise ValueError("diagnostics_df is empty; cannot auto-select eps.")

    data = data_df.sort_index().copy()
    chosen_eps, reason, msg = _select_eps(data, kendall_threshold, spearman_threshold)

    return {
        "chosen_eps": chosen_eps,
        "chosen_reason": reason,
        "chosen_msg": f"{msg} ε={chosen_eps} ** Verify using the four plots. **",
        "chosen_row": data_df.loc[chosen_eps].to_dict(),
    }


def _select_eps(data_df: pd.DataFrame, kendall_threshold: float, spearman_threshold: float) -> tuple[float, str, str]:
    """
    3-stage constraint cascade for eps selection.
    """
    c1 = data_df["pct_rows_large_clr"] == 0.0
    c2 = data_df["rank_stability_kendall"] >= kendall_threshold
    c2_relaxed = data_df["rank_stability_spearman"] >= spearman_threshold

    for mask, reason, detail in [
        (c1 & c2, "optimal", f"C1+C2 satisfied (kendall>={kendall_threshold})"),
        (c1 & c2_relaxed, "suboptimal_kendall_relaxed", f"C1 satisfied; C2 relaxed to spearman>={spearman_threshold}"),
        (c1, "suboptimal_rank_saturated", "C1 satisfied; C2 unachievable - rank stability saturated"),
    ]:
        candidates = data_df[mask]
        if not candidates.empty:
            chosen = float(candidates.index.min())
            dominated = [e for e in candidates.index if e > chosen]
            return chosen, reason, f"{detail}. Optimal ε={chosen}. Pareto-dominated: {dominated}."

    return (
        float(data_df.index.max()),
        "infeasible",
        "C1 violated for all eps: pct_rows_large_clr > 0 across the full grid. Sparsity pathology unresolved - consider a wider eps range.",
    )
