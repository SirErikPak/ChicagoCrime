import pandas as pd
import numpy as np
from typing import Dict
import detection_config as cfg


# Define constants for column names
_DATE_KEY       = cfg.config["_DATE_KEY"]
_GROUP_KEY      = cfg.config["_GROUP_KEY"]
_COUNTER_KEY    = cfg.config["_COUNTER_KEY"]


# --- Helper function to aggregate counts safely (removing duplicates)
AGG_DICT_RESULT = None  # global cache

def _aggregate_counts(
    data_df: pd.DataFrame,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    counter_col: str = _COUNTER_KEY,
    force_refresh: bool = False
) -> Dict:
    """
    Aggregates the DataFrame to count occurrences of each group-month combination.
    Uses a global cache to avoid recomputation.
    """

    global AGG_DICT_RESULT  # MUST be declared before use

    # -------------------------------------------------
    # 0. Return cached result if available
    # -------------------------------------------------
    if AGG_DICT_RESULT is not None and not force_refresh:
        return AGG_DICT_RESULT

    # -------------------------------------------------
    # 1. Aggregate counts
    # -------------------------------------------------
    data = (
        data_df.groupby([group_col, date_col], sort=True, observed=False)
        .size()
        .rename(counter_col)
        .reset_index()
    )

    # -------------------------------------------------
    # 2. Convert date FIRST (NOW data exists)
    # -------------------------------------------------
    data[date_col] = pd.to_datetime(data[date_col], format='%Y%m', errors='coerce')

    # -------------------------------------------------
    # 3. Drop invalid dates safely
    # -------------------------------------------------
    data = data.dropna(subset=[date_col])

    # -------------------------------------------------
    # 4. Ensure datetime type (now safe)
    # -------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])


    # -------------------------------------------------
    # 5. Compute date range (YYYY-MM only)
    # -------------------------------------------------
    start_date = data[date_col].min().strftime('%Y-%m')
    end_date = data[date_col].max().strftime('%Y-%m')

    # -------------------------------------------------
    # 6. Normalize dtypes
    # -------------------------------------------------
    data[group_col] = data[group_col].astype("string").astype("category")

    # -------------------------------------------------
    # 7. Store globally
    # -------------------------------------------------
    AGG_DICT_RESULT = {
        "data": data,
        "start_date": start_date,
        "end_date": end_date
    }

    return AGG_DICT_RESULT


# --- Main function for panel integrity reporting
def run_integrity_report(
    data_df: pd.DataFrame,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    freq: str = "MS"
) -> Dict:
    """
    Panel integrity + structural sparsity classification:
    - separates missingness vs sparse processes
    - identifies regime type per group
    - Force recompute if input changes
        data_dict = _aggregate_counts(df, force_refresh=True)
    """

    # -------------------------------------------------
    # Step 0: Aggregate
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df)
    data = data_dict["data"]
    start = data_dict["start_date"]
    end = data_dict["end_date"]

    # -------------------------------------------------
    # Step 1: Full grid
    # -------------------------------------------------
    full_months = pd.date_range(start=start, end=end, freq=freq)
    groups = data[group_col].dropna().unique()

    full_index = pd.MultiIndex.from_product(
        [groups, full_months],
        names=[group_col, date_col]
    )

    # -------------------------------------------------
    # Step 2: Observed coverage
    # -------------------------------------------------
    existing_index = pd.MultiIndex.from_frame(data[[group_col, date_col]])
    missing_index = full_index.difference(existing_index)
    missing_df = missing_index.to_frame(index=False)

    # -------------------------------------------------
    # Step 3: Missingness (true panel gaps)
    # -------------------------------------------------
    missing_by_group = (
        missing_df.groupby(group_col, observed=False)
        .size()
        .sort_values(ascending=False)
    )
    missing_by_group = missing_by_group[missing_by_group > 0]

    # -------------------------------------------------
    # Step 4: Coverage ratio
    # -------------------------------------------------
    coverage_ratio = (
        data.groupby(group_col, observed=False)[date_col].nunique()
        / len(full_months)
    ).sort_values()

    # -------------------------------------------------
    # Step 5: Structural regime classification
    # -------------------------------------------------
    def classify(r):
        if r < 0.30:
            return "rare_event_zero_inflated"
        elif r < 0.80:
            return "intermittent_sparse"
        else:
            return "stable_panel"

    regime = coverage_ratio.apply(classify)

    # -------------------------------------------------
    # Step 6: Summary stats
    # -------------------------------------------------
    expected = len(full_index)
    actual = len(data)
    missing = len(missing_df)
    duplicates = int(data.duplicated([group_col, date_col]).sum())

    completeness = 1 - (missing / expected)

    print("\n📊 Crime Data Integrity Summary")
    print("-" * 40)
    print(f"Date range:       {start} to {end}")
    print(f"Total groups:     {len(groups)}")
    print(f"Expected rows:    {expected}")
    print(f"Actual rows:      {actual}")
    print(f"Missing rows:     {missing}")
    print(f"Duplicates:       {duplicates}")
    print(f"Completeness:     {completeness:.2%}")

    # -------------------------------------------------
    # Step 7: Missingness (ONLY true gaps)
    # -------------------------------------------------
    if missing > 0:
        print("\n🔍 True missing panel gaps:")
        print(missing_by_group.to_string())

    # -------------------------------------------------
    # Step 8: Structural regime view
    # -------------------------------------------------
    print("\n🧠 Structural Crime Type classification:")
    print(regime.value_counts().to_string())

    print("\n📉 Sparse / risky groups:")
    print(regime[regime != "stable_panel"].to_string())

    # -------------------------------------------------
    # Step 9: Return structured output
        # -------------------------------------------------
    return {
        "data": data,
        "missing": missing_df,
        "missing_by_group": missing_by_group,
        "coverage": coverage_ratio,
        "duplicates": duplicates,
        "expected_rows": len(full_index),
        "actual_rows": len(data),
        "missing_rows": len(missing_df)
    }

# --- Main function for crime data integrity reporting
def get_integrity_report(
    data_df: pd.DataFrame,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    value_col: str = _COUNTER_KEY,
    freq: str = "MS",
    top_n: int = 5,
    verbose: bool = True
) -> Dict:
    """
    High-performance integrity check using MultiIndex reindexing.
    """
    # -------------------------------------------------
    # 0. Aggregate safely to remove duplicates
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df)
    data = data_dict["data"]
    start = data_dict["start_date"]
    end = data_dict["end_date"]

    if data.duplicated([group_col, date_col]).any():
        raise ValueError("Duplicate group-date pairs detected after aggregation.")

    # -------------------------------------------------
    # 1. Build full panel grid
    # -------------------------------------------------
    unique_groups = data[group_col].unique()
    full_range = pd.date_range(start=start, end=end, freq=freq, name=date_col)

    full_index = pd.MultiIndex.from_product(
        [unique_groups, full_range],
        names=[group_col, date_col]
    )

    # -------------------------------------------------
    # 2. Reindex to full panel (introduces missing rows)
    # -------------------------------------------------
    indexed_df = (
        data.set_index([group_col, date_col])
        .sort_index()
        .reindex(full_index)
    )
    # -------------------------------------------------
    # 3. Vectorized missing detection
    # -------------------------------------------------
    vals = indexed_df[value_col].values
    is_missing = np.isnan(vals)

    indexed_df["is_missing"] = is_missing
    indexed_df["is_zero"] = np.equal(vals, 0)

    # -------------------------------------------------
    # 4. NEW: Build explicit missing-data report table
    # -------------------------------------------------
    missing_df = (
        indexed_df.loc[indexed_df["is_missing"]]
        .reset_index()
        .copy()
    )
    # -------------------------------------------------
    # 5. Add year-month feature for reporting
    # -------------------------------------------------
    missing_df["year_month"] = missing_df[date_col].dt.to_period("M").astype(str)
    # -------------------------------------------------
    # 5a. Optional: reorder columns for readability
    # -------------------------------------------------
    missing_df = missing_df[[group_col, date_col, "year_month", value_col]]

    # -------------------------------------------------
    # 6. Coverage stats
    # -------------------------------------------------
    coverage = 1 - indexed_df.groupby(level=0, observed=True)["is_missing"].mean()
    sparse_groups = coverage.sort_values().head(top_n)

    # -------------------------------------------------
    # 7. Verbose summary
    # -------------------------------------------------
    if verbose:
        results = {
            "date_range": f"{start} to {end}",
            "total_groups": len(unique_groups),
            "total_months": len(full_range),
            "total_records": len(indexed_df),
            "total_missing": int(is_missing.sum()),
            "coverage_mean": coverage.mean(),
            "coverage_min": coverage.min(),
            "coverage_max": coverage.max(),
            "sparse_groups": sparse_groups
        }
        _print_summary(results)
    # -------------------------------------------------
    # 8. Return full diagnostics
    # -------------------------------------------------
    return {
        "integrity_df": indexed_df.reset_index(),
        "missing_df": missing_df,
        "coverage": coverage,
        "sparse_groups": sparse_groups,
        "total_missing": int(is_missing.sum())
    }


# --- Helper function to print a clean summary report
def _print_summary(results: Dict)-> None:
    """
    Prints a clean report from the results dictionary.
    """
    print("\n" + "="*30)
    print("📊 Crime Data Integrity Report")
    print("="*30)
    print(f"Date Range:           {results['date_range']}")
    print(f"Total Groups:         {results['total_groups']}")
    print(f"Total Months:         {results['total_months']}")
    print(f"Total Records:        {results['total_records']:,}")
    print(f"Total Missing Records: {results['total_missing']}")
    print(f"Average Coverage:      {results['coverage_mean']:.2%}")
    print(f"Minimum Coverage:      {results['coverage_min']:.2%}")
    print(f"Maximum Coverage:      {results['coverage_max']:.2%}")
    print("\n⚠️  Most Sparse Groups:")
    print(results['sparse_groups'])
    print("="*30 + "\n")


# --- Main function for filling missing data and reporting
def fill_missing(
    data_df: pd.DataFrame,
    group_col: str = _GROUP_KEY,
    date_col: str = _DATE_KEY,
    value_col: str = _COUNTER_KEY,
    freq: str = "MS",
    fill_value: float = 0,
    verbose: bool = True
) -> Dict:
    """
    Rebuilds full panel and fills missing crime-time observations.

    Strategy:
    - Missing group-date pairs are treated as structural zeros (default).
    - Keeps audit flags for transparency.
    """
    # -------------------------------------------------
    # 0. Aggregate (remove duplicates safely)
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df)
    data = data_dict["data"]
    start = data_dict["start_date"]
    end = data_dict["end_date"]

    # -------------------------------------------------
    # 1. Build full panel index
    # -------------------------------------------------
    unique_groups = data[group_col].unique()
    full_range = pd.date_range(start=start, end=end, freq=freq, name=date_col)

    full_index = pd.MultiIndex.from_product(
        [unique_groups, full_range],
        names=[group_col, date_col]
    )
    # -------------------------------------------------
    # 2. Reindex to full panel
    # -------------------------------------------------
    panel = (
        data.set_index([group_col, date_col])
        .sort_index()
        .reindex(full_index)
    )

    # -------------------------------------------------
    # 3. Missing flags BEFORE fill
    # -------------------------------------------------
    panel["was_missing"] = panel[value_col].isna()

    # -------------------------------------------------
    # 4. Fill strategy (default = 0)
    # -------------------------------------------------
    panel[value_col] = panel[value_col].fillna(fill_value)
    
    # -------------------------------------------------
    # 5. Derived diagnostics
    # -------------------------------------------------
    panel["is_zero_after_fill"] = panel[value_col].eq(0)

    # -------------------------------------------------
    # 6. Summary stats
    # -------------------------------------------------
    summary = {
    "total_groups": len(unique_groups),
    "total_periods": len(full_range),
    "total_rows": len(panel),
    "filled_missing": int(panel["was_missing"].sum()),
    "fill_value": fill_value
    }
    
    if verbose:
         print("\n" + "="*30)
         print("📊 Crime Data Filling Summary")
         print("="*30)
         print(f"Date Range:          {start} to {end}")
         print(f"Total Groups:        {len(unique_groups)}")
         print(f"Total Periods:       {len(full_range)}")
         print(f"Total Rows:          {len(panel):,}")
         print(f"Filled Missing:      {int(panel['was_missing'].sum())}")
         print(f"Fill Value Used:     {fill_value}")
         print("="*30 + "\n")

    # -------------------------------------------------
    # 7. Return results
    # -------------------------------------------------
    return {
        "filled_df": panel.reset_index(),
        "summary": summary
    }


# --- Validation function to check crime data integrity after filling
def validate_crime_data(
    data: pd.DataFrame,
    group_col: str=_GROUP_KEY,
    date_col: str=_DATE_KEY,
    value_col: str=_COUNTER_KEY
) -> None:
    """
    Validates panel integrity after fill_missing().
    Prints issues if found.
    """

    print("\n" + "="*40)
    print("🔍 CRIME DATA VALIDATION CHECK")
    print("="*40)

    # -------------------------------------------------
    # 1. Ensure datetime + sorted globally
    # -------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        print("❌ Date column is NOT datetime")
    else:
        print("✅ Date column is datetime")

    if data.sort_values([group_col, date_col]).reset_index(drop=True).equals(
        data.reset_index(drop=True)
    ):
        print("✅ Data is globally sorted by group and date")
    else:
        print("❌ Data is NOT sorted properly")

    # -------------------------------------------------
    # 2. Check monotonic order within each group
    # -------------------------------------------------
    bad_groups = []

    for g, sub in data.groupby(group_col, observed=False):  # ✅ FIX
        if not sub[date_col].is_monotonic_increasing:
            bad_groups.append(g)

    if len(bad_groups) == 0:
        print("✅ All groups have strictly increasing dates")
    else:
        print(f"❌ {len(bad_groups)} groups have unordered dates")

    # -------------------------------------------------
    # 3. Check full monthly continuity
    # -------------------------------------------------
    expected_counts = data.groupby(group_col, observed=False)[date_col].nunique()  # ✅ FIX

    if expected_counts.nunique() == 1:
        print(f"✅ All groups have consistent period count: {expected_counts.iloc[0]}")
    else:
        print("❌ Inconsistent number of periods across groups")

    # -------------------------------------------------
    # 4. Check for invalid values
    # -------------------------------------------------
    if (data[value_col] < 0).any():
        print("❌ Negative counts detected")
    else:
        print("✅ No negative values")

    if data[value_col].isna().any():
        print("❌ Still contains NaNs after fill")
    else:
        print("✅ No NaNs remain")

    # -------------------------------------------------
    # 5. Zero inflation check
    # -------------------------------------------------
    zero_rate = (data[value_col] == 0).mean()
    print(f"ℹ️ Zero rate: {zero_rate:.2%}")

    if zero_rate > 0.9:
        print("⚠️ Extremely sparse data (possible overfilling)")

    print("="*40 + "\n")