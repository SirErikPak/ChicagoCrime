import pandas as pd
import numpy as np
from itertools import product


# Data Prep Helper Functions ────────────────────────────────────────────────────────────
def prepare_era_data(data_df: pd.DataFrame, era_label: str) -> pd.DataFrame:
    """
    Standardizes temporal columns and assigns era labels.

    Args:
        data_df (pd.DataFrame): Must contain 'year_month' as an integer (YYYYMM format).
        era_label (str): Label to assign to the 'era' column.

    Returns:
        pd.DataFrame: Copy of input with 'year_month' as a datetime and 'era' set.

    Raises:
        ValueError: If the year_month column is missing from the data.
    """
    # input validation added - missing column previously raised a bare KeyError
    if 'year_month' not in data_df.columns:
        raise ValueError("Missing 'year_month' column.")

    data = data_df.copy()
    data['year_month'] = pd.to_datetime(data_df['year_month'].astype(str), format='%Y%m')
    data['era'] = era_label
    return data


# 1. Find Missing Crime Months ────────────────────────────────────────────────
def find_missing_crime_months(data_df: pd.DataFrame, 
                              era_label: str) -> tuple[int, int, pd.DataFrame]:
    """
    Identifies gaps in a crime time-series dataset by comparing actual 
    observations against a complete grid of all possible month/crime pairs.

    This function uses a Cartesian product to define the 'ideal' state of the 
    data and a left-merge indicator to isolate specific missing combinations.

    Args:
        data_df (pd.DataFrame): Long-form dataframe containing crime counts. 
            Must include 'year_month' and 'fbi_code_desc'.
        era_label (str): Label for the specific time period (e.g., 'Pre-COVID') 
            used for grouping and logging.

    Returns:
        tuple: (expected_rows, actual_rows, missing_only)
            - expected_rows (int): Theoretical count (unique_months * unique_crimes).
            - actual_rows (int): The number of observations currently in the data.
            - missing_only (pd.DataFrame): Subset of the grid containing only the 
              missing month/crime combinations.

    Raises:
        ValueError: If 'year_month' or 'fbi_code_desc' are missing from data_df.
    """
    
    # 1. Defensive Validation: Ensure the data 'contract' is honored before processing
    required_cols = {'year_month', 'fbi_code_desc'}
    if not required_cols.issubset(data_df.columns):
        missing = required_cols - set(data_df.columns)
        raise ValueError(f"find_missing_crime_months: Missing required columns: {missing}")

    # 2. Data Preparation: Standardize dates and assign era metadata
    # Note: prepare_era_data is assumed to handle datetime conversion
    d = prepare_era_data(data_df, era_label)

    # 3. Define the Dimensions: Get unique categories to build the 'Ideal Grid'
    all_months = d['year_month'].unique()
    all_crimes = d['fbi_code_desc'].unique()

    # Calculate theoretical vs actual density
    expected_rows = len(all_months) * len(all_crimes)
    actual_rows = len(d)

    # 4. The Cartesian Product: Generate every possible combination of month and crime.
    # This creates the 'Rectangular' baseline that a clean time series requires.
    
    full_grid = pd.DataFrame(
        product(all_months, all_crimes),
        columns=['year_month', 'fbi_code_desc']
    )

    # 5. The Gap Analysis: Merge actual data onto the full grid.
    # We use 'indicator=True' to create the '_merge' column, which flags 
    # rows present in the grid but absent in the data ('left_only').
    missing_df = full_grid.merge(
        d[['year_month', 'fbi_code_desc']],
        on=['year_month', 'fbi_code_desc'],
        how='left',
        indicator=True
    )

    # 6. Filter results: Isolate only the 'holes' identified by the merge
    missing_only = missing_df.loc[
        missing_df['_merge'] == 'left_only', 
        ['year_month', 'fbi_code_desc']
    ].copy()

    return expected_rows, actual_rows, missing_only


# 2. Era Integrity Report ────────────────────────────────────────────────────
def run_era_integrity_report(era_dict: dict[str, pd.DataFrame]
                             , expected_counts: dict[str, int]) -> None:
    """
    Orchestrates a multi-era data integrity audit to detect temporal gaps 
    and row-level density issues.

    Phase 1: Temporal Validation - Confirms every era contains the exact number 
             of expected months.
    Phase 2: Density Audit - Uses Cartesian product gaps to find missing 
             crime-month combinations and infer duplicates.

    Args:
        era_dict (dict): Maps era labels (str) to DataFrames.
        expected_counts (dict): Maps era labels (str) to expected unique month 
        counts (int).

    Raises:
        KeyError: If an era label is missing from the expected_counts map.
        AssertionError: If an era's unique month count is incorrect.
    """
    
    # PHASE 1: TEMPORAL GATEKEEPER
    # Verify ALL eras first. If one is wrong, the entire study is compromised.
    era_months_actual = {}
    
    for label, raw_df in era_dict.items():
        if label not in expected_counts:
            raise KeyError(f"run_era_integrity_report: Missing expected_counts for '{label}'")

        actual_months   = raw_df['year_month'].nunique()
        expected_months = expected_counts[label]

        # Fail-fast: prevents downstream analysis on truncated or overlapping eras
        assert actual_months == expected_months, (
            f"Era month mismatch: '{label}' expected {expected_months}, got {actual_months}. "
            f"Verify date boundary filtering logic."
        )
        era_months_actual[label] = actual_months

    print(f"✅ Era month counts verified: {era_months_actual}")

    # PHASE 2: ROW-LEVEL DENSITY AUDIT
    # Header for the diagnostic dashboard
    print(f"\n{'Era':<15} | {'Exp. Rows':<12} | {'Act. Rows':<12} | {'Missing'}")
    print("-" * 58)
    
    for label, raw_df in era_dict.items():
        # Call the Cartesian product gap finder
        exp_rows, act_rows, missing_df = find_missing_crime_months(raw_df, label)
        
        # Infer duplicates: 
        # (Expected - Missing) = what the row count SHOULD be. 
        # Difference from actual = duplicates.
        actual_missing  = len(missing_df)
        duplicate_count = act_rows - (exp_rows - actual_missing)
        
        if duplicate_count > 0:
            print(f"⚠️ WARNING: {duplicate_count} duplicate rows detected in {label}")

        # Summary line
        print(f"{label:<15} | {exp_rows:<12} | {act_rows:<12} | {actual_missing}")

        # If gaps exist, provide a breakdown by crime category
        if not missing_df.empty:
            print(" └── 🔍 Gaps by crime category:")
            gaps = missing_df.groupby('fbi_code_desc').size().sort_values(ascending=False)
            print(gaps.to_string())
        print()


# 3. Fill Missing Crime Months ────────────────────────────────────────────────
def fill_missing(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Zero-filling for time-series gaps.
    
    This function "rectangularizes" the dataframe by ensuring every month within 
    the era's range contains a row for every crime category. Missing combinations 
    are treated as structural zeros (no crimes reported) rather than missing values.

    Args:
        data_df (pd.DataFrame): Input dataframe. Must contain 'year_month', 
            'fbi_code_desc', 'crime_count', and 'era'.
            'year_month' should be in a format that can be converted to PeriodIndex.

    Returns:
        pd.DataFrame: A zero-filled, sorted dataframe with a continuous 
            monthly timeline for every crime category.

    Raises:
        ValueError: If required columns are missing or if the dataframe 
            contains multiple eras.
    """

    # 1.Defensive Validation
    required_cols = {'year_month', 'fbi_code_desc', 'crime_count', 'era'}
    if not required_cols.issubset(data_df.columns):
        missing = required_cols - set(data_df.columns)
        raise ValueError(f"fill_missing: missing required columns: {missing}")

    if data_df.empty:
        return data_df

    # Verification: Logic expects a single study period (era) per call
    if data_df['era'].nunique() != 1:
        raise ValueError("fill_missing: expects exactly one era per call")

    #2. Data Normalization
    out = data_df.copy()

    # Convert to PeriodIndex for robust monthly interval arithmetic
    out['year_month'] = pd.PeriodIndex(out['year_month'], freq='M')
    
    # Cast to Category for significant memory and GroupBy performance gains
    out['fbi_code_desc'] = out['fbi_code_desc'].astype('category')

    # Capture era label before transformation
    era_label = out['era'].iat[0]

    # 3. Grid Construction
    # Generate a continuous range from min to max date (prevents entire missing months)
    months = pd.period_range(
        out['year_month'].min(),
        out['year_month'].max(),
        freq='M'
    )

    # Extract all categories (crimes) present in this era
    crimes = out['fbi_code_desc'].cat.categories

    # Create the theoretical Cartesian product (The "Ideal Grid")
    full_idx = pd.MultiIndex.from_product(
        [months, crimes],
        names=['year_month', 'fbi_code_desc']
    )

    # 4. Aggregation & Zero-Fill 
    # observed=False ensures that even categories with zero total counts appear.
    # sum() handles potential duplicate rows for the same month/crime category.
    filled = (
        out.groupby(['year_month', 'fbi_code_desc'], observed=False)['crime_count']
        .sum()
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    # 5. Finalization
    filled['era'] = era_label
    
    # int32 is memory-efficient and more than sufficient for monthly crime counts
    filled['crime_count'] = filled['crime_count'].astype(np.int32)
    
    # Convert Period back to Timestamp for compatibility with plotting and stats libraries
    filled['year_month'] = filled['year_month'].dt.to_timestamp()

    # Stable sort ensures deterministic ordering for DTW and lagging operations
    return filled.sort_values(['year_month', 'fbi_code_desc'], kind='stable')