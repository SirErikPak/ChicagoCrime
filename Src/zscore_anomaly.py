from matplotlib.pyplot import flag
import pandas as pd
import numpy as np


def zscore_anomaly(
    data: pd.DataFrame = None,
    category_col: str = None,
    time_col: str = None,
    fill_zero: bool = True
) -> pd.DataFrame:
    """
    Computes per-category Z-scores across all time periods.
    
    Each category (e.g., crime type) is standardized using its own
    historical mean and standard deviation across all years.
    
    Args:
        data (pd.DataFrame): Raw crime data.
        category_col (str): Column representing the category (e.g., crime type).
        time_col (str): Column representing the time dimension (e.g., year).
        fill_zero (bool): If True, fill missing Z-scores with 0; 
                          if False, preserve NaN values.
    
    Returns:
        pd.DataFrame: Pivoted matrix with categories as rows and time periods as columns.
    """

    # Decide how to fill missing values
    fill_value = 0 if fill_zero else np.nan

    # Count occurrences for each (time, category) pair
    counts = (
        data.groupby([time_col, category_col], observed=False)
            .size()
            .reset_index(name='count')
            .sort_values(time_col)
    )

    # Group by category to compute historical stats
    category_groups = counts.groupby(category_col, observed=False)['count']

    # Compute per-category mean and std
    category_mean = category_groups.transform('mean')
    category_std = category_groups.transform('std').fillna(1).replace(0, 1)

    # Compute Z-scores
    counts['z_score'] = (counts['count'] - category_mean) / category_std
    counts['z_score'] = counts['z_score'].replace([np.inf, -np.inf], 0).fillna(0)

    # Pivot into a category × time matrix
    z_matrix = (
        counts.pivot(index=category_col, columns=time_col, values='z_score')
              .fillna(fill_value)
    )

    return z_matrix