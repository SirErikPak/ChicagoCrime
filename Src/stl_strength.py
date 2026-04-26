# STL Decomposition Strengths: Quantifies Trend and Seasonality vs Residual Noise
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.seasonal import STL
import config_time_series


# STL Decomposition Strengths: Quantifies Trend and Seasonality vs Residual Noise
def _stl_strengths(crime_df: pd.DataFrame) -> pd.Series:
    ts = crime_df.sort_values('year_month')['crime_count'].to_numpy(dtype=float)

    # STL requires sufficient length (>= 2 seasonal periods) and non-trivial variance
    if len(ts) < config_time_series.CONFIG["_STL_MIN_MONTHS"] or np.nanvar(ts) < 1e-8:
        return pd.Series({
            'seasonal_strength': 0.0,
            'trend_strength':    0.0,
            'stl_warning':       "Insufficient variance or length."
        })

    # Strength formula: 1 - Var(Remainder) / Var(Remainder + Component)
    # Returns 0.0 if the combined variance is zero (degenerate component)
    def _calc(var_R, comp):
        var_RC = np.nanvar(fit.resid + comp)
        return float(np.clip(1 - (var_R / var_RC), 0.0, 1.0)) if var_RC > 0 else 0.0

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fit = STL(ts, period=config_time_series.CONFIG["_STL_SEASONAL_PERIOD"], robust=True).fit()

        if np.all(np.isnan(fit.resid)):
            raise ValueError("STL produced all-NaN residuals.")

        var_R = np.nanvar(fit.resid)

        return pd.Series({
            'seasonal_strength': _calc(var_R, fit.seasonal),
            'trend_strength':    _calc(var_R, fit.trend),
            'stl_warning':       str(w[0].message) if w else None
        })

    except Exception as e:
        return pd.Series({
            'seasonal_strength': 0.0,
            'trend_strength':    0.0,
            'stl_warning':       str(e)
        })