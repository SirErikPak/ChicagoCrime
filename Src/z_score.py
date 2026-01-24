import pandas as pd
import numpy as np


def calculate_crime_zscores(data: pd.DataFrame, index_category: str='fbi_code_desc',
                            column: str='year', flag=True) -> pd.DataFrame:
    """
    Calculates Z-scores for crime types to normalize frequency across categories.
    
    Args:
        data (pd.DataFrame): Raw crime data.
        index_category (str): Index Column name
        column (str): Column name
        flag: Use False when you want to preserve the truth
        
    Returns:
        pd.DataFrame: A pivoted DataFrame with categories as index and time as columns.
    """
    # Aggregate counts by year and category
    aggregate = (
        data.groupby([column, index_category], observed=False)
        .size()
        .reset_index(name='count')
    )

    # Group for vectorized calculations
    group = aggregate.groupby(index_category, observed=False)['count']
    
    # Calculate mean and std using transform (broadcasts results back to original shape)
    mean = group.transform('mean')
    std = group.transform('std').fillna(1).replace(0, 1)

    # Compute Z-score and handle edge cases
    aggregate['z_score'] = (aggregate['count'] - mean) / std
    aggregate['z_score'] = aggregate['z_score'].replace([np.inf, -np.inf], 0).fillna(0)

    # Pivot for analysis
    if flag:
        z_pivot = aggregate.pivot(index=index_category, columns=column, values="z_score").fillna(0)
    else:
        z_pivot = yearly.pivot(index="fbi_code_desc", columns="year", values="z_score").fillna(np.nan)
    
    return z_pivot