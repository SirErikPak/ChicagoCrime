import numpy as np

"""
Global Constants:
- _VOLATILITY_BINS: Defines the thresholds for categorizing time series volatility based on the coefficient of variation (CV). 
    These bins help to classify crime types into "Very Stable", "Moderate", "Volatile", and "Highly Volatile" categories, 
    which can be useful for analysis and visualization.
- _VOLATILITY_BINS_ROBUST: Similar to _VOLATILITY_BINS but with shifted thresholds to account for the typically lower CV 
    values produced by the robust CV calculation.
- _VOLATILITY_LABELS: The corresponding labels for the volatility bins, providing a human-readable classification of the 
    time series volatility.
- _STL_MIN_MONTHS: The minimum number of months required for performing STL decomposition. This ensures that there are enough 
    data points to capture seasonal patterns and trends effectively.
- _STL_SEASONAL_PERIOD: The period for seasonal decomposition in STL, which is set to 12 for monthly data to capture annual seasonality.
- _ADF_ALPHA: The significance level for the Augmented Dickey-Fuller (ADF) test. A common choice is 0.05, which means that if the 
    p-value is less than 0.05, we reject the null hypothesis of non-stationarity, indicating that the data is stationary. This threshold 
    is crucial for determining the stationarity of the time series, which has implications for modeling and forecasting.
- _ADF_MAX_LAGS: The maximum number of lags to include in the ADF test. Setting this to None allows the test to automatically determine 
    the optimal number of lags based on the data, but it can be set to a specific number to limit the complexity of the model.
- _ADF_REGRESSION: The type of regression to include in the ADF test. "c" stands for constant only, which is a common choice for many time series. 
    Other options include "ct" (constant and trend) and "ctt" (constant and trend with quadratic term), which can be used if the data exhibits a trend.
- _ADF_AUTO_LAG: The method for automatically selecting the number of lags in the ADF test. "AIC" (Akaike Information Criterion) is a common choice 
    that balances model fit and complexity.
- _ADF_MIN_OBS: The minimum number of observations required to perform the ADF test. This ensures that the test has enough data to produce reliable results.
- _ADF_CONST_THRESHOLD: A threshold for determining if a series is effectively constant. If the standard deviation of the series is below this threshold, 
    it may be classified as constant, which has implications for stationarity testing.
- _ADF_MAX_LAG_MONTHLY: The maximum number of lags to consider for monthly data in the ADF test. This can help to capture seasonal effects without 
    overfitting the model.
- _LOG_TRANSFORM: A boolean flag indicating whether to apply a log1p transformation to the data before performing the ADF or STL test. 
    This can help to stabilize variance and make the data more normally distributed, which can improve the performance of the ADF or STLtest.
- _ELIGIBILITY_MIN_MONTHS: The minimum number of months of data required for a crime type to be eligible for analysis. This ensures that there is enough data 
    to capture meaningful patterns and trends.
- _ELIGIBILITY_MIN_PRESENCE: The minimum number of months in which a crime type must be present (non-zero) to be eligible for analysis. This helps to filter 
    out crime types that are too sparse to analyze effectively.
- _ELIGIBILITY_MIN_MEAN: The minimum average monthly count for a crime type to be eligible for analysis. This ensures that the crime type has a sufficient level 
    of activity to be meaningful for analysis.
- _ELIGIBILITY_MAX_CV: The maximum coefficient of variation (CV) allowed for a crime type to be eligible for analysis. This helps to filter out crime types that 
    are too volatile to analyze effectively.
- _ELIGIBILITY_MIN_UNIQUE: The minimum number of unique monthly counts for a crime type to be eligible for analysis. This helps to filter out crime types 
    that are too constant or near-constant to analyze effectively.
- _ELIGIBILITY_MAX_SPIKE_RATIO: acts as an outlier filter. It is designed to catch time series that are dominated by a single, extreme event-often referred 
    to as a "pulse" or a "shock", which can mathematically "blind" the ADF test to the underlying trend of the data. The ratio is calculated as the maximum 
    value in the series divided by the mean. If this ratio exceeds a certain threshold (e.g., 10), it suggests that the series has a significant spike 
    relative to its average level, which can distort the ADF test results and lead to misleading conclusions about stationarity. By setting this threshold, 
    we can flag or exclude such series from ADF testing, ensuring more reliable and meaningful results.
- _NUMERICAL_TOLERANCE_THRESHOLD: A small constant used to prevent division by zero or to determine if a series is effectively constant.
- _EPS: A small constant added to denominators to prevent division by zero in calculations such as the coefficient of variation (CV). 
    This helps to ensure numerical stability when calculating metrics that involve division, especially when the mean of the series is close to zero.
- _SPARSE_THRESHOLD: (Percentage threshold for classifying a series as "Sparse") A threshold for determining if a time series is considered "sparse". 
    If the proportion of zero counts in the series exceeds this threshold, the series may be classified as sparse, which has implications for analysis 
    and modeling. This helps to filter out crime types that have very low activity levels, 
    which may not provide meaningful insights and could potentially skew the analysis. 
- _MIN_OBSERVATIONS: The minimum number of observations required for a time series to be included in the analysis. This ensures that there is enough data 
    to capture meaningful patterns and trends, and helps to filter out crime types that are too sparse or have too few data points to analyze effectively.
- _GROUP_KEY: The column name used for grouping the data, typically representing the crime type or category. This is used in various analyses to aggregate 
    data by crime type.
- _COUNTER_KEY: The column name that contains the count of crimes. This is the primary variable of interest for time series analysis, as it represents the 
    frequency of crimes over time.
- _DATE_KEY: The column name that contains the date information, typically in a "year_month" format. This is used to index the time series data and is 
    crucial for performing time series analysis and decomposition.
"""

CONFIG = {
    # Miscellaneous Global Variables
    "_NUMERICAL_TOLERANCE_THRESHOLD": 1e-8,
    "_LOG_TRANSFORM": True,
    "_EPS": 1e-10,
    "_SPARSE_THRESHOLD_PCT": 90.0, 
    "_SPARSE_THRESHOLD_COUNT": 24,
    "_GROUP_KEY": 'fbi_code_desc',
    "_COUNTER_KEY": 'crime_count',
    "_DATE_KEY": 'year_month',

    # Global variables for stl_strength.py (STL Decomposition Strengths)
    "_VOLATILITY_BINS": [-np.inf, 25, 40, 60, np.inf],
    "_VOLATILITY_BINS_ROBUST": [-np.inf, 20, 35, 55, np.inf],
    "_VOLATILITY_LABELS": ["Very Stable", "Moderate", "Volatile", "Highly Volatile"],
    "_STL_MIN_MONTHS": 24,
    "_STL_SEASONAL_PERIOD": 12,

    # Global variables for adf_test.py (ADF Test Parameters)
    "_ADF_ALPHA": 0.05,
    "_ADF_MAX_LAGS": None,
    "_ADF_REGRESSION": "c",
    "_ADF_AUTO_LAG": "AIC",
    "_ADF_MIN_OBS": 20,
    "_ADF_CONST_THRESHOLD": 1e-8,
    "_ADF_MAX_LAG_MONTHLY": 12,
    # Eligibility criteria for ADF testing
    "_ADF_ELIGIBILITY_MIN_MONTHS": 24,
    "_ADF_ELIGIBILITY_MIN_PRESENCE": 30,
    "_ADF_ELIGIBILITY_MIN_MEAN": 2,
    "_ADF_ELIGIBILITY_MAX_CV": 150,
    "_ADF_ELIGIBILITY_MIN_UNIQUE": 5,
    "_ADF_ELIGIBILITY_MAX_SPIKE_RATIO": 10,
}