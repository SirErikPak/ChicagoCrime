import warnings
import numpy as np
import pandas as pd
from typing import Sequence, Dict, Any, Iterable
import clr_config as cfg
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

# ----------------------------------------------------------------------------
# Confgiguration parameters for clr_eps_grid_search.py
# ----------------------------------------------------------------------------
# Confgiguration parameters for clr_eps_grid_search.py
_N_PER_DECADE_N_PER_DECADE  = cfg.config_grid["_N_PER_DECADE"]
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
# Helper function 1-A: Pviot Table for building the eps grid
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
# Helper function 1-C: Validate fixed grid values
# ---------------------------------------------------------------------------
def _validate_fixed(include_fixed: Sequence[float], floor: float) -> np.ndarray:
    """
    Validate and filter a user‑supplied fixed grid of positive values.

    Ensures that all fixed grid points exceed the specified minimum
    `floor` value. This is required for log‑ratio and CLR‑based
    transformations, which operate only on strictly positive values.

    Parameters
    ----------
    include_fixed : Sequence[float]
        User‑provided list or sequence of fixed grid values.

    floor : float
        Minimum allowable value. Only entries strictly greater than
        this threshold are retained.

    Returns
    -------
    np.ndarray
        Array of validated fixed values, all strictly above `floor`.

    Raises
    ------
    ValueError
        If no values exceed the floor threshold.
    """
    # -------------------------------------------------
    # A — Filter fixed grid values above the floor
    #     CLR/log transforms require strictly positive
    #     values. This enforces that all user‑supplied
    #     fixed points exceed the minimum threshold.
    # -------------------------------------------------
    fixed = np.array([f for f in include_fixed if f > floor])

    # -------------------------------------------------
    # B — Ensure at least one valid fixed point exists
    #     Without a positive fixed grid, downstream
    #     transformations cannot proceed safely.
    # -------------------------------------------------
    if fixed.size == 0:
        raise ValueError("include_fixed contains no values above floor.")

    return fixed

# ---------------------------------------------------------------------------
# Helper function 1-D: Data‑driven anchor calculations
# ---------------------------------------------------------------------------
def _min_val(pos: np.ndarray) -> float:
    return float(pos.min())

def _q_low(pos: np.ndarray, q: float) -> float:
    return float(np.quantile(pos, q))

def _geom_mean(pos: np.ndarray) -> float:
    """Calculates geometric mean via log-space to handle varied scales."""
    return float(np.exp(np.log(pos).mean()))

def _median(pos: np.ndarray) -> float:
    return float(np.median(pos))

# ---------------------------------------------------------------------------
# Helper function 1-E: Generate anchors based on data distribution
# ---------------------------------------------------------------------------
def _multiplier_anchors(pos: np.ndarray, q: float, multipliers: np.ndarray) -> np.ndarray:
    """Generates anchors at specific fractions of the minimum and lower quantile."""
    base = np.array([_min_val(pos), _q_low(pos, q)])
    return (base[:, None] * multipliers).flatten()

# ---------------------------------------------------------------------------
# Helper Function 1-F: Generate anchors based on geometric mean
# ---------------------------------------------------------------------------
def _geom_anchors(pos: np.ndarray) -> np.ndarray:
    """Generates anchors at 1% and 10% of the geometric mean."""
    g = _geom_mean(pos)
    return np.array([g * 0.01, g * 0.1])

# ---------------------------------------------------------------------------
# Helper Function 1-G: Generate anchors based on median
# ---------------------------------------------------------------------------
def _median_anchors(pos: np.ndarray) -> np.ndarray:
    """Generates anchors at 0.1% and 1% of the median."""
    m = _median(pos)
    return np.array([m * 1e-3, m * 1e-2])

# ---------------------------------------------------------------------------
# Helper Function 1-H: Aggregate and clip anchors to the floor
# ---------------------------------------------------------------------------
def _build_anchors(
    pos: np.ndarray,
    q_low: float,
    multipliers: np.ndarray,
    floor: float,
) -> np.ndarray:
    """
    Construct a unified set of data‑driven anchor points.

    Aggregates multiple anchor families—multiplier‑based anchors,
    geometric anchors, and median‑based anchors—to create a grid that
    adapts to the empirical distribution of the positive data. The
    resulting anchors are clipped to a minimum `floor` value to ensure
    log‑safety and numerical stability.

    Parameters
    ----------
    pos : np.ndarray
        Array of strictly positive values extracted from the pivot matrix.

    q_low : float
        Lower quantile used by the multiplier‑anchor generator.

    multipliers : np.ndarray
        Multiplicative factors applied to the quantile anchor.

    floor : float
        Minimum allowable anchor value. All anchors are clipped to be
        strictly above this threshold.

    Returns
    -------
    np.ndarray
        A 1‑D array of aggregated anchor points, all >= floor.
    """
    # -------------------------------------------------
    # A — Generate anchor families
    #     Each anchor type captures a different aspect
    #     of the data distribution:
    #       • multiplier anchors → scale‑adaptive
    #       • geometric anchors  → multiplicative structure
    #       • median anchors     → central tendency
    # -------------------------------------------------
    parts = [
        _multiplier_anchors(pos, q_low, multipliers),
        _geom_anchors(pos),
        _median_anchors(pos),
    ]

    # -------------------------------------------------
    # B — Concatenate and enforce minimum floor
    #     Ensures all anchors remain strictly positive
    #     and safe for log‑ratio transformations.
    # -------------------------------------------------
    return np.maximum(np.concatenate(parts), floor)

# ---------------------------------------------------------------------------
# Helper Function 1-I: Warn if anchors are out of range
# ---------------------------------------------------------------------------
def _warn_if_anchors_out_of_range(anchors: np.ndarray, lo: float, hi: float) -> None:
    """
    Warn the user when data‑driven anchors fall entirely outside the
    fixed grid range.

    Anchors outside the interval [lo, hi] have no influence on the final
    grid construction. This warning alerts the user that the data‑driven
    component of the anchor set will be ignored, which may indicate a
    mismatch between the data scale and the chosen fixed range.

    Parameters
    ----------
    anchors : np.ndarray
        Array of data‑driven anchor points.

    lo : float
        Lower bound of the fixed grid range.

    hi : float
        Upper bound of the fixed grid range.

    Returns
    -------
    None
        Emits a UserWarning if all anchors lie outside [lo, hi].
    """
    # -------------------------------------------------
    # A — Check whether any anchors fall within the
    #     fixed grid range. If none do, the data‑driven
    #     anchors will have zero influence on the grid.
    # -------------------------------------------------
    if not np.any((anchors >= lo) & (anchors <= hi)):

        # -------------------------------------------------
        # B — Emit a warning to alert the user that the
        #     anchor set is effectively ignored due to
        #     being outside the fixed range.
        # -------------------------------------------------
        warnings.warn(
            f"All {len(anchors)} data-driven anchors fall outside fixed range "
            f"[{lo:.2e}, {hi:.2e}] and will have no effect on the grid.",
            UserWarning,
            stacklevel=3,
        )

# ---------------------------------------------------------------------------
# Helper Function 1-J: Construct a logarithmic backbone grid based on anchors and fixed range
# ---------------------------------------------------------------------------
def _make_log_grid(anchors: np.ndarray, lo: float, hi: float, n_per_decade: int) -> np.ndarray:
    """
    Construct the logarithmic backbone of the grid.

    Builds a log‑spaced sequence between a data‑driven lower bound and
    the fixed upper bound. The lower bound is chosen as the smallest
    anchor clipped into the valid range [lo, hi]. The number of points
    is proportional to the number of decades spanned, scaled by
    `n_per_decade`, ensuring consistent resolution across magnitudes.

    Parameters
    ----------
    anchors : np.ndarray
        Array of data‑driven anchor points.

    lo : float
        Minimum allowable grid value.

    hi : float
        Maximum allowable grid value.

    n_per_decade : int
        Number of grid points to allocate per decade of log‑range.

    Returns
    -------
    np.ndarray
        A 1‑D log‑spaced array forming the base grid backbone.
    """
    # -------------------------------------------------
    # A - Determine the starting point for the log grid
    #     Use the smallest anchor, but clip it into the
    #     valid range [lo, hi]. If clipping pushes it
    #     above hi, fall back to lo.
    # -------------------------------------------------
    spacing_lo = np.clip(anchors.min(), lo, hi)
    if spacing_lo >= hi:
        spacing_lo = lo

    # -------------------------------------------------
    # B - Compute number of grid points based on the
    #     number of decades spanned. Enforce a minimum
    #     of 5 points for stability and smoothness.
    # -------------------------------------------------
    n_pts = max(5, int(np.ceil(np.log10(hi / spacing_lo) * n_per_decade)))

    # -------------------------------------------------
    # C - Generate the logarithmic backbone
    # -------------------------------------------------
    return np.logspace(np.log10(spacing_lo), np.log10(hi), num=n_pts)

# ---------------------------------------------------------------------------
# Helper Function 1-K: Merge and clip anchors, fixed values, and log grid
# ---------------------------------------------------------------------------
def _merge_and_clip(
    anchors: np.ndarray,
    fixed: np.ndarray,
    log_grid: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """
    Merge all grid source points, remove duplicates, and enforce bounds.

    Combines data‑driven anchors, user‑supplied fixed points, and the
    logarithmic backbone into a single unified grid. Duplicate values are
    removed, the result is sorted, and all points are clipped to the
    inclusive range [lo, hi]. This produces the final candidate grid
    before any optional refinement steps.

    Parameters
    ----------
    anchors : np.ndarray
        Data‑driven anchor points derived from the input distribution.

    fixed : np.ndarray
        User‑specified fixed grid points.

    log_grid : np.ndarray
        Logarithmic backbone generated from the data scale.

    lo : float
        Minimum allowable grid value.

    hi : float
        Maximum allowable grid value.

    Returns
    -------
    np.ndarray
        Sorted, deduplicated, range‑restricted grid values.
    """
    # -------------------------------------------------
    # A — Merge all grid components
    #     Concatenate anchors, fixed points, and the
    #     log backbone into a single array.
    # -------------------------------------------------
    raw = np.unique(np.concatenate([anchors, fixed, log_grid]))

    # -------------------------------------------------
    # B — Enforce the allowable range [lo, hi]
    #     Removes any points outside the final grid
    #     bounds while preserving sorted order.
    # -------------------------------------------------
    return raw[(raw >= lo) & (raw <= hi)]

# ---------------------------------------------------------------------------
# Helper Function 1-L: Thin the grid to enforce a minimum relative step size
# ---------------------------------------------------------------------------
def _thin(grid: np.ndarray, hi: float, min_step: float) -> np.ndarray:
    """
    Enforce a minimum relative spacing between consecutive grid points.

    Iteratively scans a sorted grid and retains only those points that
    exceed a required relative increase (`min_step`) from the previously
    accepted point. This reduces grid density while preserving numerical
    stability for log‑ratio transforms. The upper bound `hi` is always
    retained, even if it violates the step constraint.

    Parameters
    ----------
    grid : np.ndarray
        Sorted array of candidate grid points.

    hi : float
        Maximum allowable grid value. Always preserved.

    min_step : float
        Minimum required fractional increase between retained points
        (e.g., 0.20 means each point must be at least 20% larger than
        the previous one).

    Returns
    -------
    np.ndarray
        Thinned grid satisfying the minimum‑step constraint.
    """
    # -------------------------------------------------
    # A — Initialize with the smallest grid point
    #     This ensures the thinned grid always starts
    #     at the lowest valid value.
    # -------------------------------------------------
    thinned = [grid[0]]

    # -------------------------------------------------
    # B — Iterate through remaining points and enforce
    #     the minimum relative step constraint.
    #     Always keep the upper bound `hi`.
    # -------------------------------------------------
    for val in grid[1:]:
        # Relative increase from last accepted point
        rel_diff = (val - thinned[-1]) / thinned[-1]

        # Detect whether this point is effectively the hi boundary
        at_hi = abs(val - hi) / hi < 1e-9

        # Keep if it meets the spacing requirement or is the hi endpoint
        if rel_diff >= min_step or at_hi:
            thinned.append(val)

    return np.array(thinned)

# ---------------------------------------------------------------------------
# Helper Function 1-M: Assemble the final grid by orchestrating all steps
# ---------------------------------------------------------------------------
def _assemble_and_thin(
    anchors: np.ndarray,
    fixed: np.ndarray,
    n_per_decade: int,
    min_step: float,
) -> np.ndarray:
    """
    Construct the final grid by assembling all components and applying thinning.

    This function orchestrates the full grid‑generation pipeline:
    (A) determine the valid range from fixed points,
    (B) warn if anchors fall outside that range,
    (C) build a logarithmic backbone,
    (D) merge and clip all candidate points,
    (E) thin the merged grid to enforce minimum spacing.
    """

    # -------------------------------------------------
    # A — Determine allowable grid range from fixed points
    #     The fixed grid defines the hard bounds [lo, hi].
    # -------------------------------------------------
    lo, hi = fixed.min(), fixed.max()

    # -------------------------------------------------
    # B — Warn if data-driven anchors fall entirely outside
    #     the fixed range, meaning they will not influence
    #     the final grid.
    # -------------------------------------------------
    _warn_if_anchors_out_of_range(anchors, lo, hi)

    # -------------------------------------------------
    # C — Build the logarithmic backbone based on data scale
    #     This provides smooth spacing across decades.
    # -------------------------------------------------
    log_grid = _make_log_grid(anchors, lo, hi, n_per_decade)

    # -------------------------------------------------
    # D — Merge anchors, fixed points, and log-grid candidates,
    #     then clip to the allowable range.
    # -------------------------------------------------
    merged = _merge_and_clip(anchors, fixed, log_grid, lo, hi)

    # -------------------------------------------------
    # E — If merging yields no valid points (e.g., all anchors
    #     out of range), fall back to the fixed grid. Otherwise,
    #     thin the merged grid to enforce minimum relative spacing.
    # -------------------------------------------------
    if merged.size == 0:
        return fixed

    return _thin(merged, hi, min_step)

# ---------------------------------------------------------------------------
# 1. Main function: Build the adaptive epsilon grid
# ---------------------------------------------------------------------------
def build_eps_grid(
    data_df: pd.DataFrame,
    n_per_decade: int = _N_PER_DECADE_N_PER_DECADE,
    include_fixed: Sequence[float] = _INCLUDE_FIXED,
    min_multiplier_candidates: Sequence[float] = _MIN_MULTIPLIER_CANDIDATES,
    q_low: float = _Q_LOW,
    floor: float = _FLOOR,
    min_step: float = _MIN_STEP,
) -> Dict:
    """
    Build an adaptive logarithmic epsilon grid.

    Combines user‑specified fixed values with data‑driven anchors derived
    from the distribution of the input data. The resulting grid adapts to
    the empirical scale of the dataset while preserving stable fixed
    bounds and enforcing minimum spacing.

    The construction pipeline consists of:
      A. Pivoting the long-format data into a TXK matrix
      B. Validating fixed values and extracting strictly positive entries
      C. Generating data-driven anchors from the positive distribution
      D. Assembling anchors, fixed points, and log-grid candidates
      E. Thinning the merged grid to enforce minimum relative spacing

    Parameters
    ----------
    data_df : pd.DataFrame
        Long-format crime-count data used to derive data-driven anchors.

    n_per_decade : int
        Density of the logarithmic backbone (points per decade).

    include_fixed : Sequence[float]
        User-specified fixed grid values that define the allowable range.

    min_multiplier_candidates : Sequence[float]
        Multipliers applied to the lower quantile to generate anchors.

    q_low : float
        Lower quantile used for multiplier-based anchor generation.

    floor : float
        Minimum allowable value for anchors and fixed points.

    min_step : float
        Minimum required fractional increase between adjacent grid points.

    Returns
    -------
    Dict
        {
            'eps_values': np.ndarray — final thinned epsilon grid,
            'pivot_data': pd.DataFrame — TXK pivoted data matrix
        }
    """
    # -------------------------------------------------
    # A - Pivot long-format data into a TXK matrix
    #     Required for extracting positive values and
    #     computing distribution-aware anchors.
    # -------------------------------------------------
    pivot_data = _pivot(data_df)

    # -------------------------------------------------
    # B - Validate fixed values and extract strictly
    #     positive entries from the pivoted matrix.
    #     If no positive values exist, fall back to
    #     the fixed grid.
    # -------------------------------------------------
    fixed = _validate_fixed(include_fixed, floor)
    pos   = _extract_positive(pivot_data)

    if pos.size == 0:
        return {'eps_values': np.sort(fixed), 'pivot_data': pivot_data}

    # -------------------------------------------------
    # C - Generate data-driven anchors using multiplier,
    #     geometric, and median-based anchor families.
    # -------------------------------------------------
    anchors = _build_anchors(
        pos,
        q_low,
        np.asarray(min_multiplier_candidates),
        floor
    )

    # -------------------------------------------------
    # D - Assemble anchors, fixed points, and log-grid
    #     candidates into a unified grid, then clip to
    #     the allowable range.
    # -------------------------------------------------
    eps_values = _assemble_and_thin(
        anchors,
        fixed,
        n_per_decade,
        min_step
    )

    # -------------------------------------------------
    # E - Return both the final epsilon grid and the
    #     pivoted data used to construct it.
    # -------------------------------------------------
    return {
        'eps_values': eps_values,
        'pivot_data': pivot_data
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
    # baseline bool_mask of exact zeros before smoothing; used to measure contribution
    # of original zeros to CLR variance as eps grows.
    zero_bool_mask_pre = (pivot == 0).values

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
        display_df = diagnostics_df.drop(columns=["T", "K", "rank_unique_ratio",  "n_zero_cells_pre_smooth"], errors="ignore").reset_index()
        print(display_df.to_string(index=False))

    return {"diagnostics_df": diagnostics_df, "clr_dict": clr_dict, "meta": meta, "fig": fig}

# ---------------------------------------------------------------------------
# Helper function 2-A: Compute rank‑order stability diagnostics
# ---------------------------------------------------------------------------
def _compute_rank_diagnostics(clr: pd.DataFrame, prev_clr: pd.DataFrame | None) -> dict[str, float]:
    """
    Compute rank‑order stability diagnostics for CLR‑transformed data.

    Compares the current CLR matrix to the previous epsilon iteration
    (if provided) and summarizes how stable the rank structure is across
    iterations. Metrics include:

      • Spearman rank correlation between flattened CLR matrices
      • Kendall rank correlation between column‑wise rank orders
      • Rank‑uniqueness ratio (fraction of distinct ranks per column)
      • Rank‑entropy (Shannon entropy of rank distributions)

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
# Helper function 2-B: Automated epsilon selection based on diagnostics
# ---------------------------------------------------------------------------
def _plot_diagnostics(
    data_df: pd.DataFrame,
    large_clr_threshold: float,
    kendall_threshold: float,
    chosen_eps: float | None = None
) -> plt.Figure:
    """
    Render a 2X3 diagnostic panel summarizing the ε‑sweep.

    Produces six coordinated diagnostic plots that visualize how CLR
    behavior, sparsity effects, and rank‑stability metrics evolve across
    the epsilon grid. Missing diagnostics are automatically labeled
    “not computed”.

    Panels include:
      • Max |CLR| (sensitivity)
      • % rows with large CLR values (sparsity impact)
      • Kendall τ rank stability
      • Rank‑uniqueness ratio (collapse detection)
      • % cells near zero
      • CLR variance from near‑zero cells

    Parameters
    ----------
    data_df : pd.DataFrame
        Long-format diagnostic table indexed by epsilon.

    large_clr_threshold : float
        Threshold used to flag rows with large CLR magnitudes.

    kendall_threshold : float
        Reference line for acceptable Kendall τ stability.

    chosen_eps : float or None
        Optional vertical marker for the selected epsilon.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the 2X3 diagnostic layout.
    """

    # -------------------------------------------------
    # A — Extract metadata for figure title
    # -------------------------------------------------
    T = int(data_df["T"].iloc[0])
    K = int(data_df["K"].iloc[0])
    nz = int(data_df["n_zero_cells_pre_smooth"].iloc[0])

    # -------------------------------------------------
    # B — Create 2X3 subplot layout and define panel specs
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        (axes[0, 0], "max_abs_clr", "#2c3e50", "max |CLR|", "Sensitivity: Max |CLR|"),
        (axes[0, 1], "pct_rows_large_clr", "#e74c3c", "% of Observations",
         f"Sparsity Impact: % Rows > {large_clr_threshold}"),
        (axes[0, 2], "rank_stability_kendall", "#8e44ad", "Kendall τ",
         "Rank Stability (Kendall)"),
        (axes[1, 0], "rank_unique_ratio", "#e67e22", "Unique Ratio",
         "Rank Collapse Detection"),
        (axes[1, 1], "pct_cells_near_zero", "#3498db", "% Cells < thr",
         "Near-zero Cells (% below threshold)"),
        (axes[1, 2], "clr_var_near_zero", "#16a085", "CLR Var",
         "CLR Variance from Near-zero Cells"),
    ]

    # -------------------------------------------------
    # C — Populate each panel or mark as “not computed”
    # -------------------------------------------------
    for ax, col, color, ylabel, title in panels:
        if col in data_df.columns:
            data_df[col].plot(marker="o", color=color, ax=ax)
        else:
            ax.text(0.5, 0.5, f"{col} not computed",
                    ha="center", va="center", alpha=0.6)

        ax.set(
            xscale="log",
            title=title,
            xlabel="ε (Pseudocount)",
            ylabel=ylabel,
        )
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # Optional vertical marker for chosen epsilon
        if chosen_eps is not None:
            ax.axvline(
                chosen_eps,
                color="green",
                linewidth=1.2,
                linestyle="--",
                label=f"chosen ε={chosen_eps}",
            )
            ax.legend(fontsize=9)

    # -------------------------------------------------
    # D — Add reference lines for interpretability
    # -------------------------------------------------
    try:
        axes[0, 1].axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axes[0, 2].axhline(
            kendall_threshold,
            color="orange",
            linewidth=0.8,
            linestyle=":",
            label=f"τ={kendall_threshold}",
        )
        axes[0, 2].axhline(
            1.0, color="black", linewidth=0.8, alpha=0.5, label="τ=1.0"
        )
        axes[0, 2].legend(fontsize=8)
        axes[1, 0].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    except Exception:
        pass

    # -------------------------------------------------
    # E — Final layout adjustments and return figure
    # -------------------------------------------------
    fig.suptitle(
        f"ε‑Sweep Diagnostics\nData: {T} intervals X {K} types | "
        f"Pre-smooth Zeros: {nz}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.subplots_adjust(top=0.92, wspace=0.3, hspace=0.4)
    return fig

# ---------------------------------------------------------------------------
# Helper function 2-C: Automated epsilon selection based on diagnostics
# ---------------------------------------------------------------------------
def _auto_select(
    data_df: pd.DataFrame,
    kendall_threshold: float,
    spearman_threshold: float
) -> dict[str, Any]:
    """
    Automatically select an epsilon value based on diagnostic stability.

    Uses rank‑stability and sensitivity diagnostics to choose the smallest
    epsilon that satisfies the required thresholds. Delegates the actual
    decision logic to `_select_eps`, then packages the result with a
    human‑readable message and the corresponding diagnostic row.

    Steps:
      A. Validate that diagnostics are available
      B. Sort diagnostics by epsilon
      C. Apply selection logic via `_select_eps`
      D. Construct a user‑facing message summarizing the choice
      E. Return the chosen epsilon and its diagnostic row

    Parameters
    ----------
    data_df : pd.DataFrame
        Diagnostic table indexed by epsilon.

    kendall_threshold : float
        Minimum acceptable Kendall τ stability.

    spearman_threshold : float
        Minimum acceptable Spearman stability.

    Returns
    -------
    dict[str, Any]
        {
            "chosen_eps": float,
            "chosen_reason": str,
            "chosen_msg": str,
            "chosen_row": dict
        }
    """
    # -------------------------------------------------
    # A — Validate that diagnostics exist
    # -------------------------------------------------
    if data_df.empty:
        raise ValueError("diagnostics_df is empty; cannot auto-select eps.")

    # -------------------------------------------------
    # B — Sort diagnostics by epsilon for consistent selection
    # -------------------------------------------------
    data = data_df.sort_index().copy()

    # -------------------------------------------------
    # C — Apply selection logic (delegated to _select_eps)
    # -------------------------------------------------
    chosen_eps, reason, msg = _select_eps(
        data,
        kendall_threshold,
        spearman_threshold
    )

    # -------------------------------------------------
    # D — Build a user-facing message summarizing the choice
    # -------------------------------------------------
    full_msg = f"{msg} ε={chosen_eps} ** Verify using the four plots. **"

    # -------------------------------------------------
    # E — Return selected epsilon and its diagnostic row
    # -------------------------------------------------
    return {
        "chosen_eps": chosen_eps,
        "chosen_reason": reason,
        "chosen_msg": full_msg,
        "chosen_row": data_df.loc[chosen_eps].to_dict(),
    }

# ---------------------------------------------------------------------------
# Helper function 2-D: Automated epsilon selection based on diagnostics
# ---------------------------------------------------------------------------
def _select_eps(
    data_df: pd.DataFrame,
    kendall_threshold: float,
    spearman_threshold: float
) -> tuple[float, str, str]:
    """
    Three‑stage constraint cascade for epsilon selection.

    Applies a hierarchical decision rule to choose the smallest epsilon
    that satisfies increasingly relaxed stability constraints:

      • Stage 1: C1 (no large CLR rows) AND C2 (Kendall ≥ threshold)
      • Stage 2: C1 AND relaxed C2 (Spearman ≥ threshold)
      • Stage 3: C1 only (rank stability saturated; Kendall unattainable)

    If none of the stages produce a feasible epsilon, the selection
    defaults to the largest epsilon in the grid and reports infeasibility.

    Returns the chosen epsilon, a reason label, and a human‑readable
    message describing the decision.
    """
    # -------------------------------------------------
    # A — Define constraint masks
    #     C1: No rows with large CLR values
    #     C2: Kendall τ meets threshold
    #     C2_relaxed: Spearman meets relaxed threshold
    # -------------------------------------------------
    c1 = data_df["pct_rows_large_clr"] == 0.0
    c2 = data_df["rank_stability_kendall"] >= kendall_threshold
    c2_relaxed = data_df["rank_stability_spearman"] >= spearman_threshold

    # -------------------------------------------------
    # B — Evaluate the three-stage cascade:
    #     1. C1 + C2 (optimal)
    #     2. C1 + relaxed C2 (suboptimal but acceptable)
    #     3. C1 only (rank stability saturated)
    # -------------------------------------------------
    for bool_mask, reason, detail in [
        (c1 & c2, "optimal", f"C1+C2 satisfied (kendall>={kendall_threshold})"),
        (c1 & c2_relaxed, "suboptimal_kendall_relaxed",
         f"C1 satisfied; C2 relaxed to spearman>={spearman_threshold}"),
        (c1, "suboptimal_rank_saturated",
         "C1 satisfied; C2 unachievable - rank stability saturated"),
    ]:

        # -------------------------------------------------
        # C — If any eps satisfies the current stage,
        #     choose the smallest feasible epsilon.
        # -------------------------------------------------
        candidates = data_df[bool_mask]
        if not candidates.empty:
            chosen = float(candidates.index.min())
            dominated = [e for e in candidates.index if e > chosen]
            return (
                chosen,
                reason,
                f"{detail}. Optimal ε={chosen}. Pareto-dominated: {dominated}."
            )

    # -------------------------------------------------
    # D — If no constraints are satisfied, return the
    #     largest epsilon and mark the selection infeasible.
    # -------------------------------------------------
    return (
        float(data_df.index.max()),
        "infeasible",
        "C1 violated for all eps: pct_rows_large_clr > 0 across the full grid. "
        "Sparsity pathology unresolved - consider a wider eps range.",
    )