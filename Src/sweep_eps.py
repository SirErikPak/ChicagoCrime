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
- _plot_diagnostics: Visualizes diagnostics across 4 metrics (sensitivity, sparsity,
  rank stability, rank collapse).
"""

from typing import Any, Iterable
import warnings

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
        If False, meta={} and caller must handle eps choice manually.
    verbose : bool, default=True
        If True, print diagnostics_df to stdout.
    kendall_threshold : float, default=0.98
        Minimum Kendall tau for rank stability to satisfy constraint C2.
    spearman_threshold : float, default=0.999
        Fallback Spearman rho threshold if kendall_threshold is unachievable.

    Returns
    -------
    dict[str, Any]
        Keys:
        - diagnostics_df: pd.DataFrame indexed by eps with computed metrics
        - clr_dict: dict mapping eps -> CLR DataFrame (for external use)
        - meta: dict containing auto-selection results (empty if auto_select=False)
        - fig: matplotlib Figure object (None if plot=False)

    Raises
    ------
    ValueError
        If pivot has < 2 rows, eps_grid contains non-positive or duplicate values.
    """

    if pivot.shape[0] < 2:
        raise ValueError(
            f"pivot must have at least 2 rows to compute rank diagnostics; got shape {pivot.shape}"
        )

    eps_list = [float(e) for e in eps_grid]

    # Validate eps grid: no non-positive values
    non_positive = [e for e in eps_list if e <= 0.0]
    if non_positive:
        raise ValueError(
            f"eps_grid contains non-positive values {non_positive}; all eps must be > 0"
        )

    # Detect and warn about duplicates (they will be silently deduplicated)
    seen, duplicates = set(), []
    for e in eps_list:
        if e in seen:
            duplicates.append(e)
        seen.add(e)
    if duplicates:
        warnings.warn(
            f"eps_grid contains duplicate values {duplicates}; duplicates will be ignored",
            UserWarning,
            stacklevel=2,
        )

    eps_arr = np.array(sorted(seen))

    # Pre-compute shape and zero locations in the original pivot.
    # NOTE: zero_mask_pre captures zeros in the ORIGINAL data, not after smoothing.
    # This is intentional: we measure the contribution of original sparsity to CLR variance.
    T, K          = pivot.shape
    n_zero_cells  = int((pivot == 0).sum().sum())
    zero_mask_pre = (pivot == 0).values
    diagnostics, clr_dict, prev_clr = [], {}, None

    # Main sweep loop: for each eps, compute CLR and rank diagnostics
    for eps in eps_arr:
        # Additive smoothing: add pseudo-count eps to all cells
        props   = pivot + eps
        # Normalize rows to proportions
        props   = props.div(props.sum(axis=1), axis=0)
        # Log-transform proportions
        logp    = np.log(props)
        # Center: subtract row-wise mean (CLR transformation)
        clr     = logp.sub(logp.mean(axis=1), axis=0)
        abs_clr = np.abs(clr.values)

        # Compute CLR variance decomposition: how much variation comes from original zeros?
        clr_var_total = float(np.var(abs_clr))
        clr_var_zeros = float(np.var(abs_clr[zero_mask_pre])) if zero_mask_pre.any() else 0.0
        
        # Compute rank stability metrics comparing current eps to previous
        # (NaN for eps[0] since prev_clr is None)
        rank_diag     = _compute_rank_diagnostics(clr, prev_clr)
        prev_clr      = clr

        diagnostics.append({
            "eps":                     eps,
            "max_abs_clr":             float(np.nanmax(abs_clr)),
            "mean_max_abs_clr":        float(np.nanmean(np.nanmax(abs_clr, axis=1))),
            "pct_rows_large_clr":      float((np.nanmax(abs_clr, axis=1) > large_clr_threshold).mean()) * 100.0,
            "rank_stability_spearman": rank_diag["rank_stability_spearman"],
            "rank_stability_kendall":  rank_diag["rank_stability_kendall"],
            "rank_unique_ratio":       rank_diag["rank_unique_ratio"],
            "rank_entropy":            rank_diag["rank_entropy"],
            "zero_contribution_ratio": clr_var_zeros / clr_var_total if clr_var_total > 1e-12 else 0.0,
            "n_zero_cells_pre_smooth": n_zero_cells,
            "T": T, "K": K,
        })
        
        # Store CLR for external use (e.g., downstream analysis)
        clr_dict[eps] = clr 

    diagnostics_df = pd.DataFrame(diagnostics).set_index("eps")
    # Auto-select eps only if requested; otherwise leave meta empty
    meta           = _auto_select(diagnostics_df, kendall_threshold, spearman_threshold) if auto_select else {}

    # Optionally plot diagnostics and show figure
    if plot:
        fig = _plot_diagnostics(diagnostics_df, large_clr_threshold, meta.get("chosen_eps"))
        plt.show()
    else:
        fig = None

    if verbose:
        print(diagnostics_df.to_string())

    return {"diagnostics_df": diagnostics_df, "clr_dict": clr_dict, "meta": meta, "fig": fig}


def _compute_rank_diagnostics(clr: pd.DataFrame, prev_clr: pd.DataFrame | None) -> dict[str, float]:
    """
    Compute rank-order stability metrics comparing current CLR to previous eps.
    
    Measures how much the feature ranks change from eps[i-1] to eps[i]. This helps
    detect when increasing smoothing begins to arbitrarily reorder features, indicating
    loss of signal. Also computes per-column rank entropy and unique-ratio to detect
    rank collapse (e.g., all features tied at the same rank).
    
    Parameters
    ----------
    clr : pd.DataFrame
        (T, K) CLR-transformed matrix for current eps
    prev_clr : pd.DataFrame | None
        (T, K) CLR-transformed matrix from previous eps; None for first eps
    
    Returns
    -------
    dict[str, float]
        - rank_stability_spearman: Spearman correlation of CLR values between eps[i-1] and eps[i]
          (NaN if prev_clr is None)
        - rank_stability_kendall: Kendall tau correlation of feature ranks between eps[i-1] and eps[i]
          (NaN if prev_clr is None)
        - rank_unique_ratio: Mean proportion of unique ranks per column (0 if all tied, 1 if all distinct)
        - rank_entropy: Mean Shannon entropy of rank distribution across columns
    """
    # Compute feature ranks within each column of the CLR matrix
    ranks = clr.rank(axis=0, method="min").values.astype(int)
    n_rows, n_cols = ranks.shape

    # Compare rank structure to previous eps (if available)
    if prev_clr is not None:
        prev_ranks        = prev_clr.rank(axis=0, method="min").values.astype(int)
        # Spearman: correlation of raw CLR values
        rho, _            = spearmanr(clr.values.ravel(), prev_clr.values.ravel())
        # Kendall: correlation of feature orderings
        tau, _            = kendalltau(ranks.ravel(), prev_ranks.ravel())
        spearman, kendall = float(rho), float(tau)
    else:
        # First eps has no previous to compare against
        spearman = kendall = float("nan")

    # Compute rank collapse detection: how many distinct ranks per column?
    # (A value of 1.0 means all ranks are unique; 0.0 means all features tied)
    unique_ratios, entropies = [], []
    for col in range(n_cols):
        # Count occurrence of each rank in this column
        counts = np.bincount(ranks[:, col])
        counts = counts[counts > 0]  # Remove zero-count ranks
        probs  = counts / counts.sum()
        # Ratio of distinct ranks to total rows
        unique_ratios.append(len(counts) / n_rows)
        # Shannon entropy of rank distribution (high entropy = uniform, low = concentrated)
        entropies.append(float(-np.sum(probs * np.log2(probs + 1e-12))))

    return {
        "rank_stability_spearman": spearman,
        "rank_stability_kendall":  kendall,
        "rank_unique_ratio":       float(np.mean(unique_ratios)),
        "rank_entropy":            float(np.mean(entropies)),
    }


def _plot_diagnostics(
    data_df: pd.DataFrame,
    large_clr_threshold: float,
    chosen_eps: float | None = None,
) -> plt.Figure:
    """
    Create a 2x2 diagnostic plot showing eps-sweep results across 4 key metrics.
    
    Subplots:
    - (0,0) Sensitivity: max |CLR| across all observations per eps
    - (0,1) Sparsity Impact: percentage of observations exceeding large_clr_threshold
    - (1,0) Rank Stability: Spearman correlation between successive eps CLR values
    - (1,1) Rank Collapse Detection: ratio of unique feature ranks per column
    
    If chosen_eps is provided, all subplots display a green vertical line marking
    the selected epsilon value.
    """
    # Extract metadata from diagnostics dataframe
    T, K, nz = int(data_df["T"].iloc[0]), int(data_df["K"].iloc[0]), int(data_df["n_zero_cells_pre_smooth"].iloc[0])
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Define the 4 diagnostic plots: axis, column, color, ylabel, title
    for ax, col, color, ylabel, title in [
        (axes[0, 0], "max_abs_clr",            "#2c3e50", "max |CLR|",          "Sensitivity: Max |CLR|"),
        (axes[0, 1], "pct_rows_large_clr",      "#e74c3c", "% of Observations", f"Sparsity Impact: % Rows > {large_clr_threshold}"),
        (axes[1, 0], "rank_stability_spearman", "#8e44ad", "Spearman ρ",        "Rank Stability (Spearman)"),
        (axes[1, 1], "rank_unique_ratio",       "#e67e22", "Unique Ratio",       "Rank Collapse Detection"),
    ]:
        # Plot metric vs log(eps)
        data_df[col].plot(marker="o", color=color, ax=ax)
        ax.set(xscale="log", title=title, xlabel="ε (Pseudocount)", ylabel=ylabel)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # If eps was chosen, mark it with a green dashed line
        if chosen_eps is not None:
            ax.axvline(chosen_eps, color="green", linewidth=1.2, linestyle="--", label=f"chosen ε={chosen_eps}")
            ax.legend(fontsize=9)

    # Add reference lines to subplots
    axes[0, 1].axhline(0,    color="black",  linewidth=0.8, alpha=0.5)  # 0% on sparsity plot
    axes[1, 0].axhline(0.95, color="orange", linewidth=0.8, linestyle=":", label="ρ=0.95")  # Reference correlation
    axes[1, 0].axhline(1.0,  color="black",  linewidth=0.8, alpha=0.5,    label="ρ=1.0")   # Perfect correlation
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].axhline(1.0,  color="black",  linewidth=0.8, alpha=0.5)  # Perfect rank uniqueness

    # Add informative title with dataset summary
    fig.suptitle(
        f"ε-Sweep Diagnostics\nData: {T} intervals × {K} types | Pre-smooth Zeros: {nz}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.85, wspace=0.3, hspace=0.55)
    
    return fig

def _auto_select(data_df: pd.DataFrame, kendall_threshold: float, 
                 spearman_threshold: float) -> dict[str, Any]:
    """
    Automatic eps selection given diagnostic metrics and thresholds.
    
    Implements a 3-stage constraint satisfaction strategy:
    1. Minimize eps (least smoothing) subject to:
       - Constraint C1: CLR validity (pct_rows_large_clr == 0)
       - Constraint C2: Rank stability (kendall_tau >= kendall_threshold)
    2. If C1+C2 infeasible, relax C2 to Spearman >= spearman_threshold
    3. If both fail, report infeasibility and return maximum eps in grid
    
    Returns the first (smallest) eps satisfying the chosen constraint set;
    all larger eps satisfying the same constraints are Pareto-dominated
    (valid but strictly worse on smoothing distortion).
    """
    if data_df.empty:
        raise ValueError("diagnostics_df is empty; cannot auto-select eps.")

    # Clean up any NaN rows (e.g., first row with undefined rank correlation)
    data       = data_df.sort_index().copy()
    bad_rows = data[["pct_rows_large_clr", "max_abs_clr"]].isna().any(axis=1)
    if bad_rows.any():
        data = data[~bad_rows]

    if data.empty:
        raise ValueError("No valid rows remain after dropping NaNs; cannot auto-select eps.")

    # Run the 3-stage constraint cascade
    chosen_eps, reason, msg = _select_eps(data, kendall_threshold, spearman_threshold)
    
    # Construct metadata dict for the caller
    return {
        "chosen_eps":    chosen_eps,
        "chosen_reason": reason,
        "chosen_msg":    (
            f"{msg} "
            f"Values of ε={chosen_eps} are considered 'Pareto-dominated' "
            f"because they provide no additional stability benefit, but "
            f"continue to dampen the signal (smoothing). ** Verify using the four plots. "

        ),
        "chosen_row":    data_df.loc[chosen_eps].to_dict(),
    }

def _select_eps(data_df: pd.DataFrame, kendall_threshold: float, 
                spearman_threshold: float) -> tuple[float, str, str]:
    """
    3-stage constraint cascade for eps selection.
    
    Objective: Minimize ε (least smoothing/distortion)
    Subject to constraints, with escalating relaxation:
    
    Stage 1 (Optimal):    C1 ∧ C2    → pct_rows_large_clr == 0 AND kendall >= kendall_threshold
    Stage 2 (Suboptimal): C1 ∧ C2'   → pct_rows_large_clr == 0 AND spearman >= spearman_threshold
    Stage 3 (Degraded):   C1         → pct_rows_large_clr == 0 only
    Stage 4 (Infeasible): None       → Return rightmost eps; manual intervention needed
    
    Returns
    -------
    tuple[float, str, str]
        (chosen_eps, constraint_stage_name, diagnostic_message)
    """
    # Stage 1: Optimal case - both CLR validity and rank stability achieved
    c1         = data_df["pct_rows_large_clr"] == 0.0
    c2         = data_df["rank_stability_kendall"]  >= kendall_threshold
    c2_relaxed = data_df["rank_stability_spearman"] >= spearman_threshold

    # Try each constraint combination in order of preference
    for mask, reason, detail in [
        (c1 & c2,         "optimal",                    f"C1+C2 satisfied (kendall>={kendall_threshold})"),
        (c1 & c2_relaxed, "suboptimal_kendall_relaxed",  f"C1 satisfied; C2 relaxed to spearman>={spearman_threshold}"),
        (c1,              "suboptimal_rank_saturated",   "C1 satisfied; C2 unachievable - rank stability saturated"),
    ]:
        candidates = data_df[mask]
        if not candidates.empty:
            # Choose smallest eps satisfying this constraint set (least smoothing)
            chosen    = float(candidates.index.min())
            # All larger eps satisfying the same constraints are Pareto-dominated
            dominated = [e for e in candidates.index if e > chosen]
            return chosen, reason, f"{detail}. Optimal ε={chosen}. Pareto-dominated: {dominated}."

    # Stage 4: No feasible solution - all eps violate C1 (CLR validity)
    return (
        float(data_df.index.max()),
        "infeasible",
        "C1 violated for all eps: pct_rows_large_clr > 0 across the full grid. "
        "Sparsity pathology unresolved - consider a wider eps range.",
    )