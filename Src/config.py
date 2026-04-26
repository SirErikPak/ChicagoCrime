import numpy as np

# Configurations and Constants for Time Series Analysis
CONFIG = {
    "_VOLATILITY_BINS": [-np.inf, 25, 40, 60, np.inf],
    "_VOLATILITY_BINS_ROBUST": [-np.inf, 20, 35, 55, np.inf],
    "_VOLATILITY_LABELS": ["Very Stable", "Moderate", "Volatile", "Highly Volatile"],
    "_STL_MIN_MONTHS": 24,
    "_STL_SEASONAL_PERIOD": 12,
    "_ADF_ALPHA": 0.05,
    "_ADF_MAX_LAGS": None,
    "_ADF_REGRESSION": "c",
    "_ADF_AUTO_LAG": "AIC",
    "_ADF_MIN_OBS": 20,
    "_ADF_CONST_THRESHOLD": 1e-8,
    "_ADF_MAX_LAG_MONTHLY": 12,
    "_LOG_TRANSFORM": True,
    "_ELIGIBILITY_MIN_MONTHS": 24,
    "_ELIGIBILITY_MIN_PRESENCE": 30,
    "_ELIGIBILITY_MIN_MEAN": 2,
    "_ELIGIBILITY_MAX_CV": 150,
}