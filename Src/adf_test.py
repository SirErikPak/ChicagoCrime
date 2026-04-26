import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller
import config


#  ADF (Augmented Dickey-Fuller) Eligibility Helper Function
def _is_adf_eligible(row) -> bool:
    """
    Comprehensive eligibility check to prevent ADF numerical errors and
    'garbage-in-garbage-out' statistical results.

    Gates applied in order (fail-fast):
        1. months        >= _ELIGIBILITY_MIN_MONTHS      — sufficient time-series length
        2. presence_rate >= _ELIGIBILITY_MIN_PRESENCE    — series not too sparse
        3. mean          >= _ELIGIBILITY_MIN_MEAN        — non-trivial signal (also guards cv DivByZero)
        4. cv            <  _ELIGIBILITY_MAX_CV          — not excessively volatile
        5. n_unique      >= _ELIGIBILITY_MIN_UNIQUE      — not a near-constant series
        6. max/mean      <  _ELIGIBILITY_MAX_SPIKE_RATIO — no single outlier dominating signal

    Args:
        row: pd.Series with columns:
             months, presence_rate, mean, cv, n_unique, max

    Returns:
        bool: True if series is eligible for ADF testing, False otherwise.
    """

    # Gate 1: Structural — Duration
    if row["months"] < config.CONFIG["_ELIGIBILITY_MIN_MONTHS"]:
        return False

    # Gate 2: Structural — Density
    if row["presence_rate"] < config.CONFIG["_ELIGIBILITY_MIN_PRESENCE"]:
        return False

    # Gate 3: Signal Quality — Mean
    # NOTE: Must come before gates 4 and 6.
    # If mean == 0, cv = std/mean = inf and max/mean = inf — both undefined.
    # Passing mean >= MIN_MEAN guarantees mean > 0 for all downstream math.
    if row["mean"] < config.CONFIG["_ELIGIBILITY_MIN_MEAN"]:
        return False

    # Gate 4: Variability — CV (noise ceiling)
    if row["cv"] > config.CONFIG["_ELIGIBILITY_MAX_CV"]:
        return False

    # Gate 5: Variability — Unique values (near-constant series)
    if row["n_unique"] < config.CONFIG["_ELIGIBILITY_MIN_UNIQUE"]:
        return False

    # Gate 6: Outlier / Concentration — Spike Ratio
    # mean > 0 is guaranteed by gate 3, so no ZeroDivisionError possible here.
    max_to_mean = row["max"] / row["mean"]
    if max_to_mean > config.CONFIG["_ELIGIBILITY_MAX_SPIKE_RATIO"]:
        return False

    return True


# Robust Augmented Dickey-Fuller (ADF) Test Wrapper
def _stationarity(
    series: pd.Series,
    alpha=config.CONFIG["_ADF_ALPHA"],
    max_lags=config.CONFIG["_ADF_MAX_LAGS"],
    regression=config.CONFIG["_ADF_REGRESSION"],
    autolag=config.CONFIG["_ADF_AUTO_LAG"],
    min_obs=config.CONFIG["_ADF_MIN_OBS"],
    const_threshold=config.CONFIG["_ADF_CONST_THRESHOLD"],
    max_lag_monthly=config.CONFIG["_ADF_MAX_LAG_MONTHLY"],
    log_transform=config.CONFIG["_LOG_TRANSFORM"],
) -> dict:
    """
    Robust Augmented Dickey-Fuller (ADF) test wrapper.

    Enhancements:
    - Constant-series handling (no misleading p-values)
    - Strong stationarity check via critical values
    - Trend detection heuristic
    - Lag ceiling detection
    - Low-power flag for small samples
    - Optional log1p transform for count data
    """

    # 0. Clean data
    clean = np.asarray(series).flatten()
    clean = clean[~np.isnan(clean)]

    n_raw   = len(series)
    n_clean = len(clean)

    # 1. Insufficient data
    if n_clean < min_obs:
        return {
            "adf_status"            : "insufficient_data",
            "adf_stat"              : None,
            "adf_p_value"           : None,           # <-- stat then p_value
            "adf_n_lags"            : None,
            "adf_n_obs"             : n_clean,
            "adf_n_raw_obs"         : n_raw,
            "adf_crit_vals"         : None,
            "adf_icbest"            : None,
            "adf_is_stationary"     : None,
            "adf_strong_stationarity": None,
            "adf_trend_detected"    : None,
            "adf_hit_lag_ceiling"   : None,
            "adf_low_power"         : True,
            "adf_alpha"             : alpha,
            "adf_unstable"          : False,
            "adf_warnings"          : [f"ADF skipped: observations < {min_obs}"],
            "adf_error"             : None,
        }

    # 2. Constant / near-constant
    n_unique = len(np.unique(clean))
    if n_unique <= 1 or (n_unique > 1 and clean.std() < const_threshold):
        return {
            "adf_status"            : "constant_series",
            "adf_stat"              : None,
            "adf_p_value"           : None,
            "adf_n_lags"            : 0,
            "adf_n_obs"             : n_clean,
            "adf_n_raw_obs"         : n_raw,
            "adf_crit_vals"         : None,
            "adf_icbest"            : None,
            "adf_is_stationary"     : True,            # trivially stationary
            "adf_strong_stationarity": True,
            "adf_trend_detected"    : False,
            "adf_hit_lag_ceiling"   : False,
            "adf_low_power"         : n_clean < 40,
            "adf_alpha"             : alpha,
            "adf_unstable"          : False,
            "adf_warnings"          : ["ADF skipped: constant or near-constant series."],
            "adf_error"             : None,
        }

    # 3. Optional log transform
    if log_transform and np.all(clean >= 0):
        clean = np.log1p(clean)

    # 4. Adaptive lag selection
    if max_lags is None:
        max_lags = int(min(max_lag_monthly, n_clean // 3))
    if max_lags < 1:
        max_lags = 1

    caught_warnings = []

    try:
        # 5. Run ADF
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = adfuller(
                clean,
                autolag=autolag,
                maxlag=max_lags,
                regression=regression,
            )

        caught_warnings = [str(wi.message) for wi in w]

        adf_stat, p_value, n_lags, n_obs, crit_vals, icbest = result

        # 6. Interpretation
        is_stationary = bool(p_value < alpha)

        crit_5 = crit_vals.get('5%')
        strong_stationarity = adf_stat < crit_5 if crit_5 is not None else None

        # 7. Diagnostics
        trend_slope = np.polyfit(np.arange(n_clean), clean, 1)[0]
        has_trend   = abs(trend_slope) > 1e-3

        hit_lag_ceiling = (n_lags == max_lags)
        low_power       = n_clean < 40
        adf_unstable    = bool(caught_warnings or hit_lag_ceiling)

        # 8. Return - stat then p_value,
        #    is_stationary right after p_value,
        #    strong_stationarity right after is_stationary
        return {
            "adf_status"            : "ok",
            "adf_stat"              : float(adf_stat),
            "adf_p_value"           : float(p_value),
            "adf_is_stationary"     : is_stationary,         # next to p_value
            "adf_strong_stationarity": strong_stationarity,  # next to is_stationary
            "adf_n_lags"            : int(n_lags),
            "adf_n_obs"             : int(n_obs),
            "adf_n_raw_obs"         : n_raw,
            "adf_crit_vals"         : dict(crit_vals),
            "adf_icbest"            : float(icbest) if icbest is not None else None,
            "adf_trend_detected"    : bool(has_trend),
            "adf_hit_lag_ceiling"   : hit_lag_ceiling,
            "adf_low_power"         : low_power,
            "adf_alpha"             : alpha,
            "adf_unstable"          : adf_unstable,
            "adf_warnings"          : caught_warnings,
            "adf_error"             : None,
        }

    except Exception as e:
        return {
            "adf_status"            : "error",
            "adf_stat"              : None,
            "adf_p_value"           : None,
            "adf_is_stationary"     : None,
            "adf_strong_stationarity": None,
            "adf_n_lags"            : None,
            "adf_n_obs"             : n_clean,
            "adf_n_raw_obs"         : n_raw,
            "adf_crit_vals"         : None,
            "adf_icbest"            : None,
            "adf_trend_detected"    : None,
            "adf_hit_lag_ceiling"   : None,
            "adf_low_power"         : n_clean < 40,
            "adf_alpha"             : alpha,
            "adf_unstable"          : True,
            "adf_warnings"          : caught_warnings,
            "adf_error"             : str(e),
        }