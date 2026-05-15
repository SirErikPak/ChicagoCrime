import numpy as np
import pandas as pd
from typing import  Any, Tuple, Iterable
import clr_config as cfg
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# ----------------------------------------------------------------------------
# Configuration parameters for clr_eps_grid_search.py
# ----------------------------------------------------------------------------
_N_PER_DECADE               = cfg.config_grid["_N_PER_DECADE"]
_INCLUDE_FIXED              = cfg.config_grid["_INCLUDE_FIXED"]
_MIN_MULTIPLIER_CANDIDATES  = cfg.config_grid["_MIN_MULTIPLIER_CANDIDATES"]
_Q_LOW                      = cfg.config_grid["_Q_LOW"]
_FLOOR                      = cfg.config_grid["_FLOOR"]
_MIN_STEP                   = cfg.config_grid["_MIN_STEP"]
# Pivot table keys from detection_config.py
_DATE_KEY                   = cfg.config_agg["_DATE_KEY"]
_GROUP_KEY                  = cfg.config_agg["_GROUP_KEY"]
_COUNTER_KEY                = cfg.config_agg["_COUNTER_KEY"]

# ---------------------------------------------------------------------------
# Helper function 1-A: Pivot Table for building the eps grid
# ---------------------------------------------------------------------------
def _pivot(data_df: pd.DataFrame, index: str = _DATE_KEY,
           column: str = _GROUP_KEY,
           values: str = _COUNTER_KEY) -> pd.DataFrame:

    # -------------------------------------------------
    # Pivot long-format data into a TXK matrix
    # Converts (date, group, value) rows into a wide
    # matrix required for CLR transformation.

    # Note: pivot() is safe here because the validation
    # pipeline guarantees no duplicate (index, column)
    # pairs. If duplicates were possible, pivot_table()
    # would be required.
    # -------------------------------------------------
    pivot_df = (
        data_df
        .pivot(index=index, columns=column, values=values)
        .sort_index()
    )

    return pivot_df


# ---------------------------------------------------------------------------
# Helper function 1-B: Extract positive values for anchor generation 
#                      (build_eps_grid)
# ---------------------------------------------------------------------------
def _extract_positive(pivot: pd.DataFrame) -> np.ndarray:
    """
    Extract strictly positive values from a pivoted TXK matrix.

    This helper isolates all values greater than zero from the wide
    pivot matrix. It is used to ensure log‑safety prior to CLR or
    log‑ratio transformations, which require strictly positive inputs.

    Parameters
    ----------
    pivot : pd.DataFrame
        Wide TXK matrix produced by the pivot step. Must contain only
        numeric values.

    Returns
    -------
    np.ndarray
        A 1‑D array containing all strictly positive entries from the
        pivot matrix, flattened.
    """
    # -------------------------------------------------
    # Extract strictly positive values
    # CLR/log transforms require values > 0.
    # This flattens the TXK matrix and filters out
    # zeros and negatives to guarantee log‑safety.
    # -------------------------------------------------
    vals = pivot.values
    return vals[vals > 0]


# ---------------------------------------------------------------------------
# 1. Main function: Build the adaptive epsilon grid
# ---------------------------------------------------------------------------
def build_eps_grid(
    data_df: pd.DataFrame,
    n_per_decade: int = _N_PER_DECADE,
    floor: float = _FLOOR,
    include_fixed: Iterable[float] = _INCLUDE_FIXED,
    q_low: float = _Q_LOW,
    min_step: float = _MIN_STEP,
    multipliers: Iterable[float] = _MIN_MULTIPLIER_CANDIDATES,
) -> dict:
    """
    Construct a multi-zone epsilon grid for CLR zero-replacement sweeps.

    This function builds a log-spaced epsilon grid tailored to the empirical
    distribution of positive values in the compositional dataset. The grid is
    divided into three conceptual zones:

        - Zone 1 (Sub-Min / Anchor Bridge):
          Very dense sampling between the global floor and the smallest
          observed positive value. Captures the sensitive region where
          epsilon interacts directly with structural zeros.

        - Zone 2 (Data Transition Zone):
          Dense sampling around the lower quantile of the positive data
          distribution. Tracks the transition from zero-dominated to
          data-dominated CLR behavior.

        - Zone 3 (Plateau Monitoring Zone):
          Coarser sampling at larger epsilons to detect flattening or
          over-smoothing in CLR geometry.

    The final grid merges:
        - zone-specific log-spaced sequences,
        - scaled data-driven multipliers,
        - applies adaptive thinning (smaller step in sensitive regions),
        - and then re-adds the fixed anchors unconditionally so they are
          guaranteed to appear in the final grid.

    Values are returned at full numerical precision. Rounding is the
    responsibility of presentation code (display, plotting, write-up).

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw long-format dataset containing compositional counts.
    n_per_decade : int
        Density of log-spaced samples per decade of epsilon.
    floor : float
        Global minimum epsilon floor (e.g., 1e-12).
    include_fixed : Iterable[float]
        Fixed epsilon anchors guaranteed to appear in the final grid.
    q_low : float
        Lower quantile (0-1) used to define the data-driven scale region.
    min_step : float
        Minimum relative step used during thinning above d_scale.
        Below d_scale, the gate is tightened to min_step / 2.
    multipliers : Iterable[float]
        Multipliers applied to the data scale to generate additional anchors.

    Returns
    -------
    dict
        {
            'eps_values': np.ndarray of final epsilon grid (full precision),
            'pivot_data': pivoted compositional matrix,
            'meta': {
                'data_min':   smallest positive value, or None if no positives,
                'data_scale': lower-quantile scale, or None if no positives,
                'grid_size':  number of epsilons in final grid,
            }
        }
    """
    # Materialize iterables once so we can take min/max and iterate freely.
    fixed_anchors = sorted(set(float(v) for v in include_fixed))
    mult_list = [float(v) for v in multipliers]

    pivot_data = _pivot(data_df)
    pos = _extract_positive(pivot_data)

    # Degenerate case: no positive values to drive zone construction.
    if pos.size == 0:
        eps_values = np.array(fixed_anchors, dtype=float)
        return {
            'eps_values': eps_values,
            'pivot_data': pivot_data,
            'meta': {
                'data_min':   None,
                'data_scale': None,
                'grid_size':  len(eps_values),
            },
        }

    # -----------------------------------------------
    # 1-A: Statistical anchors: data-driven points
    # ------------------------------------------------
    d_min = float(np.nanmin(pos))
    d_scale = float(np.percentile(pos, q_low * 100))
    search_floor = min(d_min, fixed_anchors[0] if fixed_anchors else d_min, floor)

    # -------------------------------------------------
    # 1-B: Zone construction: build dense log grids
    # -------------------------------------------------
    decades_to_cover = np.log10(d_min) - np.log10(search_floor)
    num_z1 = max(int(decades_to_cover * n_per_decade), 20)
    # Zone 1 covers from the global floor up to the smallest positive value
    z1 = np.logspace(np.log10(search_floor), np.log10(d_min), num=num_z1)

    # Zone 2 focuses on the transition region around d_scale
    z2 = np.logspace(np.log10(d_min), np.log10(d_scale * 10), num=max(n_per_decade, 20))

    # Zone 3 monitors the plateau region up to the largest fixed anchor or 10.0, whichever is larger
    z3_upper = max(max(fixed_anchors), 10.0) if fixed_anchors else 10.0
    z3 = np.logspace(np.log10(d_scale * 10), np.log10(z3_upper), num=10)

    # -------------------------------------------------
    # 1-C: Merge non-anchor sources and thin adaptively
    # -------------------------------------------------
    multiplier_anchors = d_scale * np.array(mult_list)
    combined = np.unique(np.concatenate([z1, z2, z3, multiplier_anchors]))

    # -------------------------------------------------
    # 1-D: Adaptive thinning: use a smaller step threshold 
    #      below d_scale to preserve resolution in the 
    #      sensitive region
    # --------------------------------------------------
    thinned = [combined[0]]
    for val in combined[1:]:
        dynamic_step = (min_step / 2.0) if val < d_scale else min_step
        if (val - thinned[-1]) / thinned[-1] >= dynamic_step:
            thinned.append(val)

    # -------------------------------------------------
    # 1-E: Re-add fixed anchors unconditionally to ensure 
    #      they are included in the final grid, even if 
    #      they violate the thinning step
    # -------------------------------------------------
    final = np.unique(np.concatenate([np.array(thinned), np.array(fixed_anchors)]))

    return {
        'eps_values': final,
        'pivot_data': pivot_data,
        'meta': {
            'data_min':   d_min,
            'data_scale': d_scale,
            'grid_size':  len(final),
        },
    }


# ---------------------------------------------------------------------------
# 2. Main function: Sweep over epsilon grid and compute CLR diagnostics
# ---------------------------------------------------------------------------
def sweep_epsilon_grid(
    pivot: pd.DataFrame,
    eps_grid: Iterable[float],
    large_clr_threshold: float = 10.0,
    plot: bool = True,
    auto_select: bool = False,
    verbose: bool = False,
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
    verbose : bool, default=False
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
    # -------------------------------------------------
    # 2-A: Input validation and preprocessing
    # -------------------------------------------------
    if pivot.shape[0] < 2:
        raise ValueError(
            f"pivot must have at least 2 rows to compute rank diagnostics; got shape {pivot.shape}"
        )

    # ------------------------------------------------
    # 2-B: Validate eps_grid values: must be positive and non-duplicating
    # -------------------------------------------------
    eps_list = [float(e) for e in eps_grid]
    if any(e <= 0.0 for e in eps_list):
        raise ValueError("eps_grid contains non-positive values; all eps must be > 0")

    # -------------------------------------------------
    # 2-C: Sort and deduplicate eps_grid to ensure consistent processing
    # -------------------------------------------------
    eps_arr = np.array(sorted(set(eps_list)))

    # --------------------------------------------------
    # 2-D: Initialize diagnostics storage and compute dimensions
    # ---------------------------------------------------
    T, K = pivot.shape
    n_zero_cells = int((pivot == 0).sum().sum())
    # baseline bool_mask of exact zeros before smoothing; used to measure contribution
    # of original zeros to CLR variance as eps grows.
    zero_bool_mask_pre = (pivot == 0).values

    # ------------------------------------------------
    # 2-E: Sweep over eps_grid and compute CLR + diagnostics
    # ------------------------------------------------
    diagnostics: list[dict[str, Any]] = []
    clr_dict: dict[float, pd.DataFrame] = {}
    prev_clr = None

    # ------------------------------------------------
    # 2-F: For each eps, compute CLR and diagnostics
    # ------------------------------------------------
    for eps in eps_arr:
        props = pivot + eps
        props = props.div(props.sum(axis=1), axis=0)
        logp = np.log(props)
        clr = logp.sub(logp.mean(axis=1), axis=0)
        abs_clr = np.abs(clr.values)

        # CLR variance contribution from original zeros vs total variance
        clr_var_total = float(np.var(abs_clr))
        clr_var_zeros = float(np.var(abs_clr[zero_bool_mask_pre])) if zero_bool_mask_pre.any() else 0.0

        # optional per-eps near-zero diagnostics
        if near_zero_threshold is not None:
            thresh = float(near_zero_threshold)
            near_bool_mask = (props.values < thresh)
            pct_cells_near_zero = float(near_bool_mask.mean()) * 100.0
            clr_var_near_zero = float(np.var(abs_clr[near_bool_mask])) if near_bool_mask.any() else 0.0
        else:
            pct_cells_near_zero = None
            clr_var_near_zero = None

        # Compute rank diagnostics compared to previous eps (None for the first iteration)
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
        # Store the CLR DataFrame for potential external use (e.g., plotting, further analysis)
        if near_zero_threshold is not None:
            diagnostics[-1]["pct_cells_near_zero"] = pct_cells_near_zero
            diagnostics[-1]["clr_var_near_zero"] = clr_var_near_zero

        # Store CLR in the dictionary for potential external use (e.g., plotting, further analysis)
        clr_dict[eps] = clr

    # Convert diagnostics list to DataFrame indexed by eps for easier analysis and plotting
    diagnostics_df = pd.DataFrame(diagnostics).set_index("eps")

    # -------------------------------------------------
    # 2-G: Optional automated eps selection based on diagnostics
    # -------------------------------------------------
    if auto_select:
        chosen_eps, chosen_reason, chosen_status = select_eps(
            diagnostics_df, kendall_threshold, spearman_threshold
        )
        # -------------------------------------------------
        # Helper Function 2-H: Stage-specific criteria function to evaluate
        # --------------------------------------------------
        def _passes_criteria(row):
            # All stages require pct_rows_large_clr == 0
            if row.get("pct_rows_large_clr", 0) > 0:
                return False
            
            k = row.get("rank_stability_kendall")
            s = row.get("rank_stability_spearman")
            if pd.isna(k) or pd.isna(s):
                return False
            
            if chosen_status == "optimal":
                # Stage 1 strict criteria
                if row.get("pct_cells_near_zero", 0) > 1e-12:
                    return False
                if k < kendall_threshold or s < spearman_threshold:
                    return False
            elif chosen_status == "near_optimal":
                # Stage 2 soft plateau criteria
                min_zero = diagnostics_df['pct_cells_near_zero'].min()
                max_k = diagnostics_df['rank_stability_kendall'].max()
                max_s = diagnostics_df['rank_stability_spearman'].max()
                if row.get("pct_cells_near_zero", 0) > min_zero + 0.005:
                    return False
                if k < max_k - 0.010 or s < max_s - 0.005:
                    return False
            else:
                # Stage 3/4: use Stage 1 criteria as a baseline "passes core checks" indicator
                if row.get("pct_cells_near_zero", 0) > 1e-12:
                    return False
                if k < kendall_threshold or s < spearman_threshold:
                    return False
            
            return True

        # -------------------------------------------------
        # 2-H: Evaluate neighboring epsilons to determine 
        #      if the chosen epsilon is at an edge or isolated
        # --------------------------------------------------
        pass_mask = diagnostics_df.apply(_passes_criteria, axis=1)
        sorted_eps = list(diagnostics_df.index)
        idx = sorted_eps.index(chosen_eps)
        # Identify neighbors and their pass/fail status
        neighbors = {
            "low":  sorted_eps[idx - 1] if idx > 0 else None,
            "high": sorted_eps[idx + 1] if idx < len(sorted_eps) - 1 else None
        }
        # Determine pass/fail/not_in_grid status for neighbors
        stats = {k: ("pass" if (v and pass_mask.loc[v]) else "fail" if v else "not_in_grid") 
                 for k, v in neighbors.items()}

        # ------------------------------------------------
        # 2-I: Classify the chosen epsilon's position in 
        #      the grid based on neighbor statuses
        # ------------------------------------------------
        if stats["low"] in ("fail", "not_in_grid") and stats["high"] == "pass":
            pos = "lower_edge"
        elif stats["high"] in ("fail", "not_in_grid") and stats["low"] == "pass":
            pos = "upper_edge"
        elif stats["low"] == "pass" and stats["high"] == "pass":
            pos = "interior"
        else:
            pos = "isolated"

        # -------------------------------------------------
        # 2-J: Generate a caveat message if the chosen 
        #      epsilon is at an edge
        # --------------------------------------------------
        caveat = None
        if pos == "lower_edge":
            caveat = f"Selection ε={chosen_eps:g} is at the LOWER EDGE of stability. Neighbors below failed."
        elif pos == "isolated":
            caveat = f"Selection ε={chosen_eps:g} is ISOLATED. Nearby grid points are unstable."

        # -------------------------------------------------
        # 2-K: Compile meta information about the selection 
        #      for reporting and plotting
        # -------------------------------------------------
        status_map = {
            'optimal': '🟢 Optimal stability', 'near_optimal': '🟡 Near-optimal',
            'elbow': '🔵 Elbow-based', 'fallback': '🟠 Fallback (Caution)'
        }

        # Meta Data Structure:
        meta = {
            "auto_select": True,
            "chosen_eps": chosen_eps,
            "chosen_tag": chosen_status,
            "chosen_status": status_map.get(chosen_status, chosen_status),
            "grid_position": pos,
            "boundary_caveat": caveat,
            "grid_spacing_below_log10": float(np.log10(chosen_eps) - np.log10(neighbors["low"])) if neighbors["low"] else None,
            "chosen_row": diagnostics_df.loc[chosen_eps].to_dict(),
            "pass_mask": pass_mask.to_dict()
        }
    else:
        meta = {"auto_select": False, "chosen_eps": None, "boundary_caveat": None}

    # -------------------------------------------------
    # 2-L: Finalization (Plotting & Verbose)
    # -------------------------------------------------
    fig = _plot_diagnostics(diagnostics_df, large_clr_threshold, kendall_threshold, meta["chosen_eps"]) if plot else None
    if plot: plt.show()

    if verbose:
        cols_to_drop = ["T", "K", "rank_unique_ratio", "n_zero_cells_pre_smooth"]
        # 1. Define the custom formatters for specific columns
        formatters = {
            'eps': '{:.16f}'.format,
            'max_abs_clr': '{:.6f}'.format,
            'mean_max_abs_clr': '{:.6f}'.format,
            # Add any other float columns you want at 6 decimals
        }

        # 2. Print the table
        print(
            diagnostics_df.drop(columns=cols_to_drop, errors="ignore")
            .reset_index()
            .to_string(index=False, formatters=formatters)
        )

    return {"diagnostics_df": diagnostics_df, "clr_dict": clr_dict, "meta": meta, "fig": fig}  


# ---------------------------------------------------------------------------
# Helper Plot Function 2-A: Automated epsilon selection based on diagnostics
# ---------------------------------------------------------------------------
def _plot_diagnostics(
    data_df: pd.DataFrame,
    large_clr_threshold: float,
    kendall_threshold: float,
    chosen_eps: float | None = None,
) -> plt.Figure:
    """
    Plot ε-sweep diagnostics across six panels on a log ε-axis.

    Each panel visualizes one diagnostic metric. If a chosen ε is provided,
    it is marked with a dashed vertical line. Legends are placed below each
    panel to avoid overlap.
    """
    # -------------------------------------------------
    # 2-A-1: Theme: Set a clean, minimalist style with a muted color palette
    # -------------------------------------------------
    plt.style.use("default")
    plt.rcParams.update({
        "axes.edgecolor":    "#444444",
        "axes.labelcolor":   "#222222",
        "axes.titlesize":    13,
        "axes.titleweight":  "600",
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
    })
    # -------------------------------------------------
    # 2-A-2: Header values: extract T, K, and n_zero_cells 
    #        from the first row of the diagnostics DataFrame
    # -------------------------------------------------
    T  = int(data_df["T"].iloc[0])
    K  = int(data_df["K"].iloc[0])
    nz = int(data_df["n_zero_cells_pre_smooth"].iloc[0])

    # -------------------------------------------------
    # 2-A-3: Panel config: Define colors, labels, and 
    #        titles for each diagnostic panel
    # -------------------------------------------------
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#1B9E77"]

    # Create a 2x3 grid of subplots for the six diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Define the panels with their corresponding DataFrame column, color, y-label, and title
    panels = [
        (axes[0, 0], "max_abs_clr",            colors[0], "Max |CLR|",      "Sensitivity: Max |CLR|"),
        (axes[0, 1], "pct_rows_large_clr",     colors[1], "% Observations", f"Sparsity Impact: % Rows > {large_clr_threshold}"),
        (axes[0, 2], "rank_stability_kendall", colors[2], "Kendall τ",      "Rank Stability (Kendall)"),
        (axes[1, 0], "rank_unique_ratio",      colors[3], "Unique Ratio",   "Rank Collapse Detection"),
        (axes[1, 1], "pct_cells_near_zero",    colors[4], "% Cells < thr",  "Near-zero Cells"),
        (axes[1, 2], "clr_var_near_zero",      colors[5], "CLR Var",        "CLR Variance from Near-zero Cells"),
    ]
    # ------------------------------------------------
    # 2-A-4: Prepare the label for the chosen ε line if applicable
    # ------------------------------------------------
    chosen_label = fr"chosen $\epsilon$ = {chosen_eps:.6f}" if chosen_eps is not None else None

    # -------------------------------------------------
    # 2-A-5: Draw each panel: plot the metric if present, 
    #        set log x-axis, titles, labels, and mark chosen ε
    # -------------------------------------------------
    for ax, col, color, ylabel, title in panels:

        # Plot metric if present
        if col in data_df.columns:
            ax.plot(
                data_df.index, data_df[col],
                marker="o", markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.6,
                color=color, linewidth=1.8, label=col,
            )

        # Shared formatting
        ax.set_xscale("log")
        ax.set_title(title, pad=10)
        ax.set_xlabel("ε (Pseudocount)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="both", labelsize=9, colors="#555555")

        # Mark chosen ε with a vertical dashed line if provided
        if chosen_eps is not None:
            ax.axvline(chosen_eps, color="#2E7D32", lw=1.6, ls="--", label=chosen_label)

    # Kendall threshold reference line
    axes[0, 2].axhline(
        kendall_threshold, color="#F58518", lw=1.2, ls=":",
        label=fr"$\tau = {kendall_threshold:.3f}$",
    )

    # ------------------------------------------------
    # 2-A-6: Legends below each panel: collect handles 
    #        and labels, then place a single legend 
    #        below each subplot
    # ------------------------------------------------
    for ax in axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.15),
                frameon=False,
                fontsize=9,
                ncol=min(len(handles), 3),
                handlelength=1.8,
                columnspacing=1.4,
                borderaxespad=0,
            )
    # -------------------------------------------------
    # 2-A-7: Title and spacing: Add a comprehensive 
    #        suptitle and adjust subplot spacing to 
    #        prevent overlap
    # -------------------------------------------------
    fig.suptitle(
        fr"$\boldsymbol{{\epsilon}}$ - Sweep Diagnostics  |  Data: {T} × {K}  |  Pre-smooth Zeros: {nz:,}",
        fontsize=16, fontweight="600", y=0.995,
    )

    # subplots_adjust parameters are tuned to balance space for the suptitle, x/y labels, 
    # and legends below each panel while maximizing the plot area for the data.
    fig.subplots_adjust(
        top=0.91,
        bottom=0.14,
        left=0.06,
        right=0.98,
        hspace=0.5,
        wspace=0.15,
    )

    return fig


# ---------------------------------------------------------------------------
# Helper function 2-B: Compute rank‑order stability diagnostics
#                      (sweep_epsilon_grid)
# ---------------------------------------------------------------------------
def _compute_rank_diagnostics(clr: pd.DataFrame, prev_clr: pd.DataFrame | None) -> dict[str, float]:
    """
    Compute rank‑order stability diagnostics for CLR‑transformed data.

    Compares the current CLR matrix to the previous epsilon iteration
    (if provided) and summarizes how stable the rank structure is across
    iterations. Metrics include:

      - Spearman rank correlation between flattened CLR matrices
      - Kendall rank correlation between column‑wise rank orders
      - Rank‑uniqueness ratio (fraction of distinct ranks per column)
      - Rank‑entropy (Shannon entropy of rank distributions)

    These diagnostics help determine whether the epsilon grid search has
    reached a region where rank structure stabilizes.

    Parameters
    ----------
    clr : pd.DataFrame
        Current CLR‑transformed TXK matrix.

    prev_clr : pd.DataFrame or None
        CLR matrix from the previous epsilon value. If None, stability
        metrics are returned as NaN.

    Returns
    -------
    dict[str, float]
        {
            "rank_stability_spearman": Spearman correlation of flattened CLR values,
            "rank_stability_kendall": Kendall correlation of rank orders,
            "rank_unique_ratio": mean fraction of unique ranks per column,
            "rank_entropy": mean entropy of rank distributions
        }
    """
    # -------------------------------------------------
    # A — Compute integer rank matrix for current CLR
    #     Ranks are computed column‑wise using "min" to
    #     ensure deterministic handling of ties.
    # -------------------------------------------------
    ranks = clr.rank(axis=0, method="min").values.astype(int)
    n_rows, n_cols = ranks.shape

    # -------------------------------------------------
    # B — If a previous CLR exists, compute rank‑stability
    #     metrics (Spearman on raw values, Kendall on ranks).
    # -------------------------------------------------
    if prev_clr is not None:
        prev_ranks = prev_clr.rank(axis=0, method="min").values.astype(int)
        rho, _ = spearmanr(clr.values.ravel(), prev_clr.values.ravel())
        tau, _ = kendalltau(ranks.ravel(), prev_ranks.ravel())
        spearman, kendall = float(rho), float(tau)
    else:
        spearman = kendall = float("nan")

    # -------------------------------------------------
    # C — Compute per‑column rank‑uniqueness ratios and
    #     rank‑entropy values to summarize distributional
    #     structure of ranks within each column.
    # -------------------------------------------------
    unique_ratios, entropies = [], []
    for col in range(n_cols):
        counts = np.bincount(ranks[:, col])
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        unique_ratios.append(len(counts) / n_rows)
        entropies.append(float(-np.sum(probs * np.log2(probs + 1e-12))))

    # -------------------------------------------------
    # D — Aggregate diagnostics into a structured dict
    # -------------------------------------------------
    return {
        "rank_stability_spearman": spearman,
        "rank_stability_kendall": kendall,
        "rank_unique_ratio": float(np.mean(unique_ratios)),
        "rank_entropy": float(np.mean(entropies)),
    }


# ---------------------------------------------------------------------------
# 3: Automated epsilon selection based on diagnostics (sweep_epsilon_grid)
# ---------------------------------------------------------------------------
def select_eps(
    df: pd.DataFrame,
    kendall_thresh: float,
    spearman_thresh: float,
    tol: float = 1e-12,
    slack_zero: float = 0.005,
    slack_kendall: float = 0.010,
    slack_spear: float = 0.005,
    elbow_threshold: float = 0.25,
    fallback_weights: tuple = (1.0, 1.0, 1.0),
) -> Tuple[float, str, str]:
    """
    Select the CLR zero-handling epsilon via cascading-fallback criteria.

    Evaluates an epsilon sweep against three diagnostic metrics and returns
    the smallest epsilon that satisfies progressively weaker stage criteria.
    The cascade order is hard plateau -> soft plateau -> elbow -> composite
    fallback; each stage runs only if the previous stage finds no qualifying
    rows.

    Parameters
    ----------
    df : pd.DataFrame
        Sweep diagnostics indexed by candidate epsilon values (sorted
        ascending internally). Required columns:
            'pct_cells_near_zero'      : fraction of cells imputed as ~zero
                                         (e.g., from sweep_epsilon_grid)
            'rank_stability_kendall'   : Kendall tau vs reference ranking
            'rank_stability_spearman'  : Spearman rho vs reference ranking
            'mean_max_abs_clr'         : distortion magnitude (Stage 3 elbow)

    kendall_thresh : float
        Strict Kendall threshold for Stage 1 hard plateau (e.g., 0.98).

    spearman_thresh : float
        Strict Spearman threshold for Stage 1 hard plateau (e.g., 0.999).

    tol : float, default 1e-12
        Strict tolerance for "no zero artifacts" in Stage 1. A row qualifies
        for the hard plateau only if pct_cells_near_zero <= tol. The default
        is essentially zero; pass a small positive value (e.g., 0.001) to
        make Stage 1 reachable on sweeps with residual near-zero cells.

    slack_zero : float, default 0.005
        Stage 2 additive tolerance for pct_cells_near_zero. A row qualifies
        if its zero rate is within `slack_zero` of the sweep's minimum.
        The default of 0.5 percentage point reflects the small sampling
        noise of a metric computed on T*K cells.

    slack_kendall : float, default 0.010
        Stage 2 additive tolerance for rank_stability_kendall. A row
        qualifies if its Kendall is within `slack_kendall` of the sweep's
        maximum. The default of 1 percentage point matches the typical
        sampling SE of Kendall tau on T~300 samples (~0.01-0.02), so it
        avoids rejecting rows that are statistically indistinguishable
        from the best.

    slack_spear : float, default 0.005
        Stage 2 additive tolerance for rank_stability_spearman. Spearman
        has tighter sampling SE than Kendall, so a smaller slack is used.

    elbow_threshold : float, default 0.25
        Stage 3 trigger: a row's normalized curvature must exceed this
        fraction of the sweep's max curvature to count as an elbow.
        Lower values detect gentler bends; higher values demand sharper
        knees.

    fallback_weights : tuple of 3 floats, default (1.0, 1.0, 1.0)
        Stage 4 weights for (zero_score, kendall, spearman) in the
        composite score. Increase the first weight to penalize zero
        artifacts more aggressively in the fallback.

    Returns
    -------
    eps : float
        The selected epsilon value (drawn from df.index).

    reason : str
        Human-readable description of which stage triggered.

    status : str
        Plain status tag for downstream display logic:
            'optimal'      - Stage 1 hard plateau
            'near_optimal' - Stage 2 soft plateau
            'elbow'        - Stage 3 elbow detection
            'fallback'     - Stage 4 composite fallback

    Notes
    -----
    Cascading philosophy:
        Stage 1 is strict by design and may not fire on noisy real data.
        That is expected. Stage 2 then catches "good enough" cases using
        metric-specific additive slack calibrated to each metric's
        sampling noise. Stage 3 falls back to elbow detection if metrics
        never plateau. Stage 4 is a least-bad-option choice that flags
        a caution status; recurring Stage 4 hits suggest the sweep itself
        is underpowered or poorly designed.

    Stage 2 slack calibration:
        The three slack parameters are intentionally different to reflect
        the underlying sampling noise of each metric:
            - Zero rate is computed on T*K cells (thousands of values)
              and has tiny SE; small slack (0.005) is conservative.
            - Kendall has larger SE on rank correlations from T~300
              samples; slack of 0.010 lets rows within ~1 SE qualify.
            - Spearman has SE about 60-70% of Kendall's at the same N;
              slack of 0.005 reflects this tighter distribution.
        The prior uniform slack design implicitly demanded ~10x stricter
        agreement on Kendall than the metric's sampling noise warranted.

    Stage 3 curvature:
        Computed via two applications of `np.gradient` on the log10(eps)
        axis. This produces a 5-point stencil approximation of d^2/dx^2,
        handles non-uniform spacing correctly, and gives values at the
        endpoints (unlike a centered finite difference). The log
        transform is essential: a finite difference on the raw eps
        axis would produce spurious large "curvature" values driven by
        the non-uniform spacing of a log-spaced sweep, not by the
        underlying function shape.

    Stage 4 composite:
        A normalized sum across the three metrics rather than a
        lexicographic sort. Avoids the failure mode where a row with
        marginally lower zero rate but much worse rank stability beats
        a row with slightly higher zero rate and excellent stability,
        purely on tiebreaker order.

    Examples
    --------
    >>> from sweep import sweep_epsilon_grid
    >>> sweep = sweep_epsilon_grid(pivot, np.logspace(-6, -1, 30))
    >>> eps, reason, status = select_eps(
    ...     sweep['diagnostics_df'],
    ...     kendall_thresh=0.98,
    ...     spearman_thresh=0.999,
    ... )
    >>> print(f"Selected eps={eps:.2e} via {status}: {reason}")
    Selected eps=3.16e-04 via near_optimal: Soft plateau (near-asymptotic)
    """
    df = df.copy().sort_index()

    # ------------------------------------------------------------
    # Stage 1 - Hard Plateau (strict)
    # ------------------------------------------------------------
    hard_mask = (
        (df['pct_cells_near_zero']      <= tol) &
        (df['pct_rows_large_clr']       == 0) &   
        (df['rank_stability_kendall']   >= kendall_thresh) &
        (df['rank_stability_spearman']  >= spearman_thresh)
    )
    hard_plateau = df[hard_mask]
    if not hard_plateau.empty:
        eps = float(hard_plateau.index.min())
        return eps, "Hard plateau (zero artifacts negligible)", "optimal"

    # ------------------------------------------------------------
    # Stage 2 - Soft Plateau (additive tolerance)
    # ------------------------------------------------------------
    min_zero = df['pct_cells_near_zero'].min()
    max_k    = df['rank_stability_kendall'].max()
    max_s    = df['rank_stability_spearman'].max()

    soft_mask = (
        (df['pct_cells_near_zero']      <= min_zero + slack_zero) &
        (df['pct_rows_large_clr']       == 0) & # <--- Keep this strict here too
        (df['rank_stability_kendall']   >= max_k    - slack_kendall) &
        (df['rank_stability_spearman']  >= max_s    - slack_spear)
    )
    # Soft plateau relaxes the rank stability constraints to allow epsilons 
    # that are close to the best observed values, while still requiring no
    #  large CLR artifacts. This captures the "near-asymptotic" region where 
    # metrics have essentially plateaued but may not meet the strict criteria 
    # of Stage 1 due to minor sampling noise or residual near-zero cells.
    soft_plateau = df[soft_mask]
    if not soft_plateau.empty:
        eps = float(soft_plateau.index.min())
        return eps, "Soft plateau (near-asymptotic)", "near_optimal"    

    # ------------------------------------------------------------
    # Stage 3 - Elbow Detection (curvature on log10(eps) axis)
    # ------------------------------------------------------------
    if len(df) >= 3:
        y = df['mean_max_abs_clr'].values
        x = np.log10(df.index.values.astype(float))

        # Second derivative via np.gradient (handles non-uniform spacing,
        # gives values at endpoints, reads as standard calculus)
        dy        = np.gradient(y, x)
        curvature = np.gradient(dy, x)

        max_abs_curv = np.max(np.abs(curvature))
        if max_abs_curv > 1e-12:
            curv_norm = curvature / max_abs_curv
            elbow_candidates = np.where(curv_norm > elbow_threshold)[0]
            if len(elbow_candidates) > 0:
                best_idx = elbow_candidates[np.argmax(curv_norm[elbow_candidates])]
                eps = float(df.index.values[best_idx])
                return eps, "Elbow detected in distortion (log-axis curvature)", "elbow"

    # ------------------------------------------------------------
    # Stage 4 - Fallback (composite normalized score)
    # ------------------------------------------------------------
    # Min-max normalize zero-rate to [0, 1] where 1 is best (lowest zeros).
    # Min-max (rather than divide-by-max) prevents amplifying noise when
    # max_zero is small and keeps zero_norm on the same [0, 1] scale as
    # Kendall/Spearman so all three contribute commensurate signal.
    zero_min  = df['pct_cells_near_zero'].min()
    zero_max  = df['pct_cells_near_zero'].max()
    zero_norm = 1.0 - (df['pct_cells_near_zero'] - zero_min) / (zero_max - zero_min + 1e-12)

    # Kendall and Spearman are already in [-1, 1]; treat negative values as 0
    k_norm = df['rank_stability_kendall'].clip(lower=0)
    s_norm = df['rank_stability_spearman'].clip(lower=0)

    w_zero, w_k, w_s = fallback_weights
    composite = w_zero * zero_norm + w_k * k_norm + w_s * s_norm

    eps = float(composite.idxmax())
    return eps, "Fallback (composite score across metrics)", "fallback"