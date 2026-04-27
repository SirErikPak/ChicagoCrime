import warnings
import numpy as np
import pandas as pd
from typing import Sequence
import build_eps_grid as eps

# Confgiguration parameters for build_eps_grid.py
_N_PER_DECADE_N_PER_DECADE  = eps.config["_N_PER_DECADE_N_PER_DECADE"]
_INCLUDE_FIXED              = eps.config["_INCLUDE_FIXED"]
_MIN_MULTIPLIER_CANDIDATES  = eps.config["_MIN_MULTIPLIER_CANDIDATES"]
_Q_LOW                      = eps.config["_Q_LOW"]
_FLOOR                      = eps.config["_FLOOR"]
_MIN_STEP                   = eps.config["_MIN_STEP"]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _extract_positive(pivot: pd.DataFrame) -> np.ndarray:
    vals = pivot.values
    return vals[vals > 0]


def _validate_fixed(include_fixed: Sequence[float], floor: float) -> np.ndarray:
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
    return float(np.exp(np.log(pos).mean()))

def _median(pos: np.ndarray) -> float:
    return float(np.median(pos))


# ---------------------------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------------------------

def _multiplier_anchors(pos: np.ndarray, q: float, multipliers: np.ndarray) -> np.ndarray:
    base = np.array([_min_val(pos), _q_low(pos, q)])
    return (base[:, None] * multipliers).flatten()


def _geom_anchors(pos: np.ndarray) -> np.ndarray:
    g = _geom_mean(pos)
    return np.array([g * 0.01, g * 0.1])


def _median_anchors(pos: np.ndarray) -> np.ndarray:
    m = _median(pos)
    return np.array([m * 1e-3, m * 1e-2])


def _build_anchors(
    pos: np.ndarray,
    q_low: float,
    multipliers: np.ndarray,
    floor: float,
) -> np.ndarray:
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
    if not np.any((anchors >= lo) & (anchors <= hi)):
        warnings.warn(
            f"All {len(anchors)} data-driven anchors fall outside fixed range "
            f"[{lo:.2e}, {hi:.2e}] and will have no effect on the grid.",
            UserWarning,
            stacklevel=3,
        )


def _make_log_grid(anchors: np.ndarray, lo: float, hi: float, n_per_decade: int) -> np.ndarray:
    spacing_lo = np.clip(anchors.min(), lo, hi)
    if spacing_lo >= hi:
        spacing_lo = lo
    n_pts = max(5, int(np.ceil(np.log10(hi / spacing_lo) * n_per_decade)))
    return np.logspace(np.log10(spacing_lo), np.log10(hi), num=n_pts)


def _merge_and_clip(
    anchors: np.ndarray,
    fixed: np.ndarray,
    log_grid: np.ndarray,
    lo: float,
    hi: float,
) -> np.ndarray:
    raw = np.unique(np.concatenate([anchors, fixed, log_grid]))
    return raw[(raw >= lo) & (raw <= hi)]


def _thin(grid: np.ndarray, hi: float, min_step: float) -> np.ndarray:
    thinned = [grid[0]]
    for val in grid[1:]:
        rel_diff = (val - thinned[-1]) / thinned[-1]
        at_hi = abs(val - hi) / hi < 1e-9
        if rel_diff >= min_step or at_hi:
            thinned.append(val)
    return np.array(thinned)


def _assemble_and_thin(
    anchors: np.ndarray,
    fixed: np.ndarray,
    n_per_decade: int,
    min_step: float,
) -> np.ndarray:
    lo, hi = fixed.min(), fixed.max()
    _warn_if_anchors_out_of_range(anchors, lo, hi)
    log_grid = _make_log_grid(anchors, lo, hi, n_per_decade)
    merged = _merge_and_clip(anchors, fixed, log_grid, lo, hi)
    if merged.size == 0:
        return fixed
    return _thin(merged, hi, min_step)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def build_eps_grid(
    pivot: pd.DataFrame,
    n_per_decade: int = _N_PER_DECADE_N_PER_DECADE,
    include_fixed =_INCLUDE_FIXED,
    min_multiplier_candidates =_MIN_MULTIPLIER_CANDIDATES,
    q_low =_Q_LOW,
    floor =_FLOOR,
    min_step =_MIN_STEP,
) -> np.ndarray:
    fixed = _validate_fixed(include_fixed, floor)
    pos = _extract_positive(pivot)

    if pos.size == 0:
        return np.sort(fixed)

    anchors = _build_anchors(pos, q_low, np.asarray(min_multiplier_candidates), floor)
    return _assemble_and_thin(anchors, fixed, n_per_decade, min_step)