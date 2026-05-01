import warnings
import numpy as np
import pandas as pd
from typing import Sequence, Dict
import epsilon_grid_config as egc
import detection_config as cfg


# Confgiguration parameters for build_eps_grid.py
_N_PER_DECADE_N_PER_DECADE  = egc.config["_N_PER_DECADE"]
_INCLUDE_FIXED              = egc.config["_INCLUDE_FIXED"]
_MIN_MULTIPLIER_CANDIDATES  = egc.config["_MIN_MULTIPLIER_CANDIDATES"]
_Q_LOW                      = egc.config["_Q_LOW"]
_FLOOR                      = egc.config["_FLOOR"]
_MIN_STEP                   = egc.config["_MIN_STEP"]
# Pivot table keys from detection_config.py
_DATE_KEY                   = cfg.config["_DATE_KEY"]
_GROUP_KEY                  = cfg.config["_GROUP_KEY"]
_COUNTER_KEY                = cfg.config["_COUNTER_KEY"]


# ---------------------------------------------------------------------------
# Pviot Table for building the eps grid
# ---------------------------------------------------------------------------
def _pivot(data_df: pd.DataFrame, index: str = _DATE_KEY, 
           column: str = _GROUP_KEY, 
           values: str = _COUNTER_KEY) -> pd.DataFrame:
    """
    Pivots long-format data into a wide matrix with optional caching.
    
    Args:
        data_df (pd.DataFrame): The source dataframe.
        index (str): Column to use as the new index (e.g., Date).
        column (str): Column to use as the new column (e.g., Crime Type).
        values (str): Column to populate the cells (e.g., Counts).
        
    Returns:
        pd.DataFrame: The T x K pivot table.
    """
    # 1. Execution
    # Note: Using .pivot_table() instead of .pivot() is safer if there's 
    # any chance of duplicate (index, column) pairs; it aggregates by default.
    pivot_df = (
        data_df
        .pivot(index=index, columns=column, values=values)
        .sort_index()
    )

    return pivot_df


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _extract_positive(pivot: pd.DataFrame) -> np.ndarray:
    """Filters out zero and negative values to ensure log-safety."""
    vals = pivot.values
    return vals[vals > 0]

def _validate_fixed(include_fixed: Sequence[float], floor: float) -> np.ndarray:
    """Ensures fixed grid values are above the minimum floor."""
    fixed = np.array([f for f in include_fixed if f > floor])
    if fixed.size == 0:
        raise ValueError("include_fixed contains no values above floor.")
    return fixed


# ---------------------------------------------------------------------------
# Statistics
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
# Anchor generation
# ---------------------------------------------------------------------------
def _multiplier_anchors(pos: np.ndarray, q: float, multipliers: np.ndarray) -> np.ndarray:
    """Generates anchors at specific fractions of the minimum and lower quantile."""
    base = np.array([_min_val(pos), _q_low(pos, q)])
    return (base[:, None] * multipliers).flatten()


# ---------------------------------------------------------------------------
# Geometric mean-based anchors
# ---------------------------------------------------------------------------
def _geom_anchors(pos: np.ndarray) -> np.ndarray:
    """Generates anchors at 1% and 10% of the geometric mean."""
    g = _geom_mean(pos)
    return np.array([g * 0.01, g * 0.1])


# ---------------------------------------------------------------------------
# Median-based anchors
# ---------------------------------------------------------------------------
def _median_anchors(pos: np.ndarray) -> np.ndarray:
    """Generates anchors at 0.1% and 1% of the median."""
    m = _median(pos)
    return np.array([m * 1e-3, m * 1e-2])


# ---------------------------------------------------------------------------
# Anchor aggregation and clipping
# ---------------------------------------------------------------------------
def _build_anchors(
    pos: np.ndarray,
    q_low: float,
    multipliers: np.ndarray,
    floor: float,
) -> np.ndarray:
    """
    Aggregates all data-driven anchor points and clips them to the floor.
    These points ensure the grid is sensitive to the specific scale of the input data.
    """
    # Each anchor type captures different aspects of the data distribution, providing a
    parts = [
        _multiplier_anchors(pos, q_low, multipliers),
        _geom_anchors(pos),
        _median_anchors(pos),
    ]
    return np.maximum(np.concatenate(parts), floor)


# ---------------------------------------------------------------------------
# Grid assembly
# ---------------------------------------------------------------------------
def _warn_if_anchors_out_of_range(anchors: np.ndarray, lo: float, hi: float) -> None:
    """Alerts user if data-driven anchors won't impact the final fixed range."""

    # Check if any anchors fall within the fixed range; if not, warn the user
    if not np.any((anchors >= lo) & (anchors <= hi)):
        warnings.warn(
            f"All {len(anchors)} data-driven anchors fall outside fixed range "
            f"[{lo:.2e}, {hi:.2e}] and will have no effect on the grid.",
            UserWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# Logarithmic backbone
# ---------------------------------------------------------------------------
def _make_log_grid(anchors: np.ndarray, lo: float, hi: float, n_per_decade: int) -> np.ndarray:
    """Creates a base logarithmic backbone for the grid."""
    # Ensure our log scale starts at the lowest anchor but stays within range
    spacing_lo = np.clip(anchors.min(), lo, hi)
    if spacing_lo >= hi:
        spacing_lo = lo
    
    # Calculate points based on the number of decades covered
    n_pts = max(5, int(np.ceil(np.log10(hi / spacing_lo) * n_per_decade)))
    return np.logspace(np.log10(spacing_lo), np.log10(hi), num=n_pts)


# ---------------------------------------------------------------------------
# Merging and clipping
# ---------------------------------------------------------------------------
def _merge_and_clip(
    anchors: np.ndarray,
    fixed: np.ndarray,
    log_grid: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Merges all source points, removes duplicates, and enforces range bounds."""
    # Use np.unique to remove duplicates and sort the combined array
    raw = np.unique(np.concatenate([anchors, fixed, log_grid]))
    return raw[(raw >= lo) & (raw <= hi)]


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------
def _thin(grid: np.ndarray, hi: float, min_step: float) -> np.ndarray:
    """
    Reduces grid density by enforcing a minimum relative step between points.
    
    Args:
        grid: Sorted array of potential grid points.
        hi: The maximum allowed value (must be preserved).
        min_step: Minimum % increase required to keep a point (e.g., 0.20 = 20%).
    """
    thinned = [grid[0]]
    for val in grid[1:]:
        # Calculate relative distance from the last accepted point
        rel_diff = (val - thinned[-1]) / thinned[-1]
        
        # Always keep the 'hi' boundary point regardless of step size
        at_hi = abs(val - hi) / hi < 1e-9

        # Keep the point if it meets the minimum step requirement or is the hi boundary
        if rel_diff >= min_step or at_hi:
            thinned.append(val)
    return np.array(thinned)

#  ---------------------------------------------------------------------------
# Assemble the final grid by merging all sources and thinning it to ensure a 
# manageable number of points while preserving coverage of the relevant range
# ---------------------------------------------------------------------------
def _assemble_and_thin(
    anchors: np.ndarray,
    fixed: np.ndarray,
    n_per_decade: int,
    min_step: float,
) -> np.ndarray:
    """Orchestrates the creation, merging, and pruning of the grid."""

    # Determine the range for the grid based on fixed values and warn if anchors won't contribute
    lo, hi = fixed.min(), fixed.max()
    _warn_if_anchors_out_of_range(anchors, lo, hi)
    
    # Create a logarithmic backbone and merge it with anchors and fixed values, then thin the result
    log_grid = _make_log_grid(anchors, lo, hi, n_per_decade)
    # Merge all candidate points and clip to the fixed range before thinning
    merged = _merge_and_clip(anchors, fixed, log_grid, lo, hi)
    
    # If merging results in an empty array (which can happen if all anchors are out of range), fall back to the fixed grid
    if merged.size == 0:
        return fixed
        
    return _thin(merged, hi, min_step)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def build_eps_grid(
    data_df: pd.DataFrame,
    n_per_decade: int = _N_PER_DECADE_N_PER_DECADE,
    include_fixed: Sequence[float] =  _INCLUDE_FIXED,
    min_multiplier_candidates: Sequence[float] =  _MIN_MULTIPLIER_CANDIDATES,
    q_low: float = _Q_LOW,
    floor: float = _FLOOR,
    min_step: float = _MIN_STEP,
) -> Dict:
    """
    Constructs an adaptive logarithmic grid for epsilon values.
    
    The grid combines standard fixed values with anchors derived from the 
    distribution of the input data, ensuring the search space is relevant
    to the specific dataset provided.
    
    Args:
        data_df: DataFrame containing values used to calculate data-driven anchors.
        n_per_decade: Density of the logarithmic backbone.
        include_fixed: Baseline values that define the grid's range.
        min_multiplier_candidates: Fractions of data minimums to use as anchors.
        q_low: The lower quantile used for anchor calculation.
        floor: Absolute minimum value allowed in the grid.
        min_step: Minimum relative increase between adjacent points (prevents over-density).
        
    Returns:
        A sorted 1D array of grid points.
    """
    # _pivot to reshape data_df
    pivot_data = _pivot(data_df)

    # Validate and prepare fixed values, and extract positive data for anchor generation
    fixed = _validate_fixed(include_fixed, floor)
    pos = _extract_positive(pivot_data)

    # Fallback to fixed grid if no positive data exists
    if pos.size == 0:
        return np.sort(fixed)
    
    # Generate anchors based on the data distribution and assemble the final grid
    anchors = _build_anchors(pos, q_low, np.asarray(min_multiplier_candidates), floor)
    return { 'eps_values': _assemble_and_thin(anchors, fixed, n_per_decade, min_step),  
            'pivot_data': pivot_data }