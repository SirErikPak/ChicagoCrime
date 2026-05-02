# STL Decomposition Strengths: Quantifies Trend and Seasonality vs Residual Noise
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.seasonal import STL
import config_time_series


# STL Decomposition Strengths: Quantifies Trend and Seasonality vs Residual Noise
def _calc_stl_strength(resid: np.ndarray, component: np.ndarray, tol: float) -> float:
    """
    Compute STL component strength using Cleveland et al. (1990) formula.

    strength = max(0, 1 - Var(R) / Var(R + Component))

    Parameters
    ----------
    resid     : np.ndarray - STL residuals
    component : np.ndarray - seasonal or trend component
    tol       : float      - numerical zero threshold from config

    Returns
    -------
    float in [0, 1]
    """
    var_R  = np.nanvar(resid)
    var_RC = np.nanvar(resid + component)

    if var_RC <= tol:
        return 0.0

    return float(np.clip(1 - (var_R / var_RC), 0.0, 1.0))


def _stl_strengths(crime_df: pd.DataFrame) -> pd.Series:
    """
    Compute STL-based seasonal and trend strength for a single crime series.

    Strength formula (Cleveland et al. 1990):
        strength = max(0, 1 - Var(R) / Var(R + Component))

    Parameters
    ----------
    crime_df : pd.DataFrame
        Must contain 'year_month' and 'crime_count' columns.

    Returns
    -------
    pd.Series with keys:
        seasonal_strength  : float in [0, 1]
        trend_strength     : float in [0, 1]
        stl_warning        : str or None
        is_log_transformed : bool
    """

    # Config
    is_logged  = config_time_series.CONFIG.get("_LOG_TRANSFORM", False)
    min_months = config_time_series.CONFIG["_STL_MIN_MONTHS"]
    tol        = config_time_series.CONFIG.get("_NUMERICAL_TOLERANCE_THRESHOLD", 1e-8)
    period     = config_time_series.CONFIG["_STL_SEASONAL_PERIOD"]

    # Period validation
    if period < 2:
        raise ValueError(
            f"_STL_SEASONAL_PERIOD must be >= 2, got {period}. "
            f"Check config_time_series.CONFIG."
        )

    # Build time series
    ts = (
        crime_df
        .sort_values('year_month', kind='stable')['crime_count']
        .to_numpy(dtype=float)
    )

    if is_logged:
        ts = np.log1p(ts)

    # Pre-flight: length and variance check
    if len(ts) < min_months or np.nanvar(ts) < tol:
        return pd.Series({
            'seasonal_strength':  0.0,
            'trend_strength':     0.0,
            'stl_warning':        f"Insufficient variance or length (n={len(ts)}).",
            'is_log_transformed': is_logged
        })

    # STL decomposition
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            fit = STL(ts, period=period, robust=True).fit()

            # Extract inside context manager to capture all warnings
            seasonal_strength = _calc_stl_strength(fit.resid, fit.seasonal, tol)
            trend_strength    = _calc_stl_strength(fit.resid, fit.trend,    tol)
            all_nan           = np.all(np.isnan(fit.resid))

        if all_nan:
            raise ValueError("STL produced all-NaN residuals.")

        return pd.Series({
            'seasonal_strength':  seasonal_strength,
            'trend_strength':     trend_strength,
            'stl_warning':        "; ".join(str(x.message) for x in w) if w else None,
            'is_log_transformed': is_logged
        })

    except Exception as e:
        return pd.Series({
            'seasonal_strength':  0.0,
            'trend_strength':     0.0,
            'stl_warning':        f"STL Error [{type(e).__name__}]: {str(e)}",
            'is_log_transformed': is_logged
        })