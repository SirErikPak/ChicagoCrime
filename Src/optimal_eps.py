import warnings
from typing import Any, Dict, Optional, TypedDict

import pandas as pd


_ZERO_SPARSITY_COLUMNS = ("n_rows_large_clr", "pct_rows_large_clr")


class OptimalEpsResult(TypedDict):
    """Structured result returned by the eps selector."""

    chosen_eps: float
    chosen_reason: str
    chosen_row: Dict[str, Any]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_diagnostics(data_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the diagnostics frame used for selection."""
    required = {"max_abs_clr"}
    missing = required - set(data_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not any(column in data_df.columns for column in _ZERO_SPARSITY_COLUMNS):
        raise ValueError(
            "Missing zero-sparsity column. Expected one of: "
            + ", ".join(_ZERO_SPARSITY_COLUMNS)
        )

    data = data_df.copy()
    data.index = pd.Index(pd.to_numeric(data.index, errors="raise"), name=data.index.name)
    if data.index.has_duplicates:
        raise ValueError("diagnostics_df index must not contain duplicate eps values.")
    return data.sort_index()


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------
def _sparsity_resolved(data_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where sparsity is fully resolved.

    Prefer the count-based diagnostic when it is available, otherwise fall back
    to the percentage-based metric used by newer sweep outputs.
    """
    if "n_rows_large_clr" in data_df.columns:
        return data_df[data_df["n_rows_large_clr"] == 0]
    return data_df[data_df["pct_rows_large_clr"] == 0.0]


def _below_clr_threshold(data_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return data_df[data_df["max_abs_clr"] <= threshold]


def _smallest(data_df: pd.DataFrame) -> float:
    return float(data_df.index.min())


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------
def _result(data_df: pd.DataFrame, eps: float, reason: str) -> dict:
    """Build the public result payload for a selected eps."""
    return {
        "chosen_eps": eps,
        "chosen_reason": reason,
        "chosen_row": data_df.loc[eps].to_dict(),
    }


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------
def _select_zero_sparsity(data_df: pd.DataFrame, clr_threshold: float | None) -> OptimalEpsResult | None:
    candidates = _sparsity_resolved(data_df)
    if candidates.empty:
        return None
    if clr_threshold is not None:
        candidates = _below_clr_threshold(candidates, clr_threshold)
        if candidates.empty:
            warnings.warn(
                f"No eps achieves zero sparsity AND max_abs_clr <= {clr_threshold}. "
                "Falling back to threshold-only selection.",
                UserWarning, stacklevel=3,
            )
            return None
    return _result(data_df, _smallest(candidates), "smallest_eps_with_zero_pct_large_clr")


def _select_clr_threshold(data_df: pd.DataFrame, threshold: float) -> OptimalEpsResult | None:
    candidates = _below_clr_threshold(data_df, threshold)
    if candidates.empty:
        return None
    return _result(data_df, _smallest(candidates), "first_eps_below_clr_threshold")


def _select_fallback(data_df: pd.DataFrame) -> OptimalEpsResult:
    warnings.warn(
        "No eps fully resolves sparsity. Returning largest available eps as last resort. "
        "Consider widening build_eps_grid range.",
        UserWarning, stacklevel=3,
    )
    eps = float(data_df.index.max())
    return _result(data_df, eps, "fallback_largest_eps")


# ---------------------------------------------------------------------------
# Main selection function
# ---------------------------------------------------------------------------
def select_optimal_eps(
    diagnostics_df: pd.DataFrame,
    clr_threshold: float | None = None,
) -> OptimalEpsResult:
    """
    Select the optimal epsilon from a CLR diagnostics sweep.

    Selection priority:
      1. Smallest eps where pct_rows_large_clr == 0, optionally also
         satisfying max_abs_clr <= clr_threshold.
      2. If step 1 fails due to clr_threshold conflict: smallest eps
         below clr_threshold regardless of sparsity.
      3. Last resort: largest eps in the grid, with a warning.

    Parameters
    ----------
    diagnostics_df : pd.DataFrame
        Indexed by eps. Must contain 'pct_rows_large_clr' and 'max_abs_clr'.
    clr_threshold : float, optional
        If provided, the chosen eps must also satisfy max_abs_clr <= threshold.

    Returns
    -------
    OptimalEpsResult
        {
            "chosen_eps":    float,
            "chosen_reason": str,
            "chosen_row":    dict,
        }

    Raises
    ------
    ValueError
        If required columns are missing from diagnostics_df.
    """
    data = _validate_diagnostics(diagnostics_df)

    # Keep the three-stage strategy explicit: first resolve sparsity, then fall
    # back to the CLR threshold, and only then use the largest eps available.
    return (
        _select_zero_sparsity(data, clr_threshold)
        or (clr_threshold is not None and _select_clr_threshold(data, clr_threshold))
        or _select_fallback(data)
    )