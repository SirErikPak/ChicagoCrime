import pandas as pd
import numpy as np
from typing import Dict
import detection_config as cfg


# Define constants for column names
_DATE_KEY         = cfg.config["_DATE_KEY"]
_GROUP_KEY        = cfg.config["_GROUP_KEY"]
_COUNTER_KEY      = cfg.config["_COUNTER_KEY"]
# global cache for aggregated results
AGG_DICT_RESULT   = None 


# --- Helper function to aggregate counts safely (removing duplicates)
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


# -------------------------------------------------------------------
# Main function for integrity report (no filling, just analysis)
# -------------------------------------------------------------------
def run_integrity_report(
    data_df   : pd.DataFrame,
    group_col : str = _GROUP_KEY,
    date_col  : str = _DATE_KEY,
    freq      : str = "MS"
) -> Dict:
    """
    Panel integrity and structural sparsity report for crime count data.

    Performs seven diagnostic checks on the raw aggregated panel:
      1. Aggregate raw data into monthly crime counts per category
      2. Build the complete expected panel grid (all groups x all months)
      3. Identify observed vs expected coverage
      4. Detect true panel gaps  (months present in grid but absent in data)
      5. Compute per-group coverage ratios
      6. Summarize completeness statistics
      7. Flag sparse groups below full coverage

    This function does NOT fill or impute missing values.
    It is a pre-processing diagnostic intended to be run before the
    CLR transformation to distinguish structural zeros from true
    missingness.

    Parameters
    ----------
    data_df   : pd.DataFrame
        Raw crime incident records. Must contain group_col and date_col.
    group_col : str
        Column identifying crime category.
        Default: _GROUP_KEY = 'fbi_code_desc'
    date_col  : str
        Column identifying observation month.
        Default: _DATE_KEY = 'year_month'
    freq      : str
        Pandas date frequency string. 'MS' = month start. Default 'MS'.

    Returns
    -------
    dict with keys:
        date_range       : tuple(start, end)  full date range of the panel
        missing          : DataFrame          all (group, month) gaps
        missing_by_group : Series             gap count per group, sorted desc
        coverage_ratio   : Series             observed months / expected per group
        duplicates       : int                duplicate (group, month) rows
        expected_rows    : int                total rows in complete panel grid
        actual_rows      : int                total rows in observed data
        missing_rows     : int                expected minus actual
        completeness     : float              1 minus (missing / expected)

    Notes
    -----
    Force refresh the aggregation cache if the input data has changed:
        data_dict = _aggregate_counts(data_df, force_refresh=True)

    True panel gaps (missing_by_group) reflect months where a crime
    category was not recorded at all, and are distinct from structural
    zeros where the category exists but had zero incidents. Both affect
    CLR pseudocount sensitivity and should be reviewed before fitting.
    """

    # Step 0 - Aggregate raw records into monthly counts
    # _aggregate_counts returns a dict with 'data', 'start_date', 'end_date'
    data_dict = _aggregate_counts(data_df)
    data      = data_dict["data"]
    start     = data_dict["start_date"]
    end       = data_dict["end_date"]

    # Step 1 - Build the complete expected panel grid
    # All combinations of (crime category x month) that should exist
    # given the observed date range and crime groups
    full_months = pd.date_range(start=start, end=end, freq=freq)
    groups      = data[group_col].dropna().unique()

    full_index  = pd.MultiIndex.from_product(
        [groups, full_months],
        names=[group_col, date_col]
    )

    # Step 2 - Identify observed vs expected coverage
    # Compute the set difference to find all missing (group, month) pairs
    existing_index = pd.MultiIndex.from_frame(data[[group_col, date_col]])
    missing_index  = full_index.difference(existing_index)
    missing_df     = missing_index.to_frame(index=False)

    # Step 3 - True panel gaps per group
    # Count completely absent months per crime category
    # These are NOT structural zeros - the row does not exist at all
    missing_by_group = (
        missing_df
        .groupby(group_col, observed=False)
        .size()
        .sort_values(ascending=False)
    )
    missing_by_group = missing_by_group[missing_by_group > 0]

    # Step 4 - Per-group coverage ratio
    # Fraction of expected months actually observed per crime category
    # coverage < 1.0 means at least one month is absent for that group
    coverage_ratio = (
        data
        .groupby(group_col, observed=False)[date_col]
        .nunique()
        / len(full_months)
    ).sort_values()

    # Step 5 - Summary statistics
    expected     = len(full_index)
    actual       = len(data)
    missing      = len(missing_df)
    duplicates   = int(data.duplicated([group_col, date_col]).sum())
    completeness = 1 - (missing / expected)

    print(f"\n{'='*66}")
    print(f"{'CRIME DATA INTEGRITY SUMMARY':^66}")
    print(f"{'='*66}")
    print(f"Date range    : {start} to {end}")
    print(f"Total groups  : {len(groups):,}")
    print(f"Expected rows : {expected:,}")
    print(f"Actual rows   : {actual:,}")
    print(f"Missing rows  : {missing:,}")
    print(f"Duplicates    : {duplicates:,}")
    print(f"Completeness  : {completeness:.2%}")

    # Step 6 - Report true panel gaps
    # Structural zeros (zero counts) are NOT reported here
    # Only months where the row is entirely absent are flagged
    if missing > 0:
        print(f"{'-'*66}")
        print("TRUE MISSING CRIME GAPS:")
        table = f"{(missing_by_group.to_string()):^66}"
        print("\n".join("    " + line for line in table.splitlines()))

    # Step 7 - Flag sparse and risky groups
    # Groups with coverage < 1.0 have at least one missing month
    # These are candidates for pseudocount sensitivity in CLR transformation
    print(f"{'-'*66}")
    print("SPARSE / RISKY CRIME GROUPS:")
    sparse = coverage_ratio[coverage_ratio < 1.0]
    if not sparse.empty:
        table = f"{(sparse.to_string()):^66}"
        print("\n".join("    " + line for line in table.splitlines()))
    else:
        print("  None - all groups fully covered across all months")

    # Step 8 - Return structured output
    return {
        "date_range"      : (start, end),
        "missing"         : missing_df,
        "missing_by_group": missing_by_group,
        "coverage_ratio"  : coverage_ratio,
        "duplicates"      : duplicates,
        "expected_rows"   : expected,
        "actual_rows"     : actual,
        "missing_rows"    : missing,
        "completeness"    : completeness,
    }


# -------------------------------------------------------------------
# Main function for filling missing data and reporting
# -------------------------------------------------------------------
def fill_missing(
    data_df   : pd.DataFrame,
    group_col : str   = _GROUP_KEY,
    date_col  : str   = _DATE_KEY,
    value_col : str   = _COUNTER_KEY,
    freq      : str   = "MS",
    fill_value: float = 0,
    verbose   : bool  = True
) -> Dict:
    """
    Rebuild the full crime panel and fill missing group-month observations.

    Constructs the complete expected panel grid from the observed date range
    and crime categories, then reindexes the aggregated data against it.
    Missing (group, month) pairs are treated as structural zeros by default
    and filled with fill_value. Audit flags are preserved for transparency.

    Parameters
    ----------
    data_df    : pd.DataFrame
        Raw crime incident records. Must contain group_col and date_col.
    group_col  : str
        Column identifying crime category.
        Default: _GROUP_KEY = 'fbi_code_desc'
    date_col   : str
        Column identifying observation month.
        Default: _DATE_KEY = 'year_month'
    value_col  : str
        Column containing the crime count to fill.
        Default: _COUNTER_KEY = 'crime_count'
    freq       : str
        Pandas date frequency string. 'MS' = month start. Default 'MS'.
    fill_value : float
        Value used to fill missing observations. Default 0.
        Treated as structural zeros for CLR pseudocount purposes.
    verbose    : bool
        If True, prints the filling summary report. Default True.

    Returns
    -------
    dict with keys:
        date_range : tuple(start, end)   full date range of the panel
        filled_df  : pd.DataFrame        complete panel with fill flags
                     columns include was_missing and is_zero_after_fill
        summary    : dict                aggregate statistics of the fill

    Notes
    -----
    The filled_df returned contains two audit columns:
        was_missing        : bool  True if the row was absent before filling
        is_zero_after_fill : bool  True if value_col equals zero after fill
                                   includes both filled and originally-zero rows

    This function is designed to run after run_integrity_report and before
    the CLR transformation. The was_missing flag enables downstream
    identification of pseudocount-driven CLR values.
    """

    # Step 0 - Aggregate raw records into monthly counts
    # Removes duplicate (group, month) pairs safely before panel construction
    data_dict = _aggregate_counts(data_df)
    data      = data_dict["data"]
    start     = data_dict["start_date"]
    end       = data_dict["end_date"]

    # Step 1 - Build the complete expected panel index
    # Cartesian product of all crime categories x all months in date range
    unique_groups = data[group_col].unique()
    full_range    = pd.date_range(
        start=start, end=end, freq=freq, name=date_col
    )
    full_index    = pd.MultiIndex.from_product(
        [unique_groups, full_range],
        names=[group_col, date_col]
    )

    # Step 2 - Reindex observed data against the full panel
    # Rows absent from the observed data become NaN after reindex
    panel = (
        data
        .set_index([group_col, date_col])
        .sort_index()
        .reindex(full_index)
    )

    # Step 3 - Flag missing rows BEFORE filling
    # was_missing=True identifies rows that did not exist in the raw data
    # Downstream: these rows have CLR values driven by pseudocount, not signal
    panel["was_missing"] = panel[value_col].isna()

    # Step 4 - Fill missing values
    # Default fill_value=0 treats absences as structural zeros
    # Change fill_value to use alternative imputation strategies
    panel[value_col] = panel[value_col].fillna(fill_value)

    # Step 5 - Derived audit flag
    # is_zero_after_fill=True covers both filled zeros and originally-zero rows
    # Used to identify all zero counts regardless of origin
    panel["is_zero_after_fill"] = panel[value_col].eq(0)

    # Step 6 - Summary statistics
    summary = {
        "total_groups"  : len(unique_groups),
        "total_periods" : len(full_range),
        "total_rows"    : len(panel),
        "filled_missing": int(panel["was_missing"].sum()),
        "fill_value"    : fill_value,
    }

    if verbose:
        print("\n" + "=" * 40)
        print(f"{'CRIME DATA FILLING SUMMARY':^40}")
        print("=" * 40)
        print(f"Date range      : {start} to {end}")
        print(f"Total groups    : {len(unique_groups):,}")
        print(f"Total periods   : {len(full_range):,}")
        print(f"Total rows      : {len(panel):,}")
        print(f"Filled missing  : {int(panel['was_missing'].sum()):,}")
        print(f"Fill value used : {fill_value}")
        print("=" * 40 + "\n")

    # Step 7 - Return structured output
    return {
        "date_range": (start, end),
        "filled_df" : panel.reset_index(),
        "summary"   : summary,
    }


# -------------------------------------------------------------------
# Validation function to check crime data integrity after filling
# -------------------------------------------------------------------
def validate_crime_data(
    data                : pd.DataFrame,
    zero_rate_threshold : float = 0.90,
    group_col           : str   = _GROUP_KEY,
    date_col            : str   = _DATE_KEY,
    value_col           : str   = _COUNTER_KEY
) -> None:
    """
    Validate panel integrity after fill_missing().

    Runs ten structural checks on the filled panel and prints
    a diagnostic report. Designed to be called immediately after
    fill_missing() and before the CLR transformation to confirm
    the panel is complete, sorted, and free of invalid values.

    Checks performed:
      1.  Date column dtype is datetime64
      2.  Data is globally sorted by group and date
      3.  Dates within each group are strictly monotonic increasing
      4.  All groups have the same number of periods
      5.  No negative counts
      6.  No remaining NaNs
      7.  No duplicate (group, date) pairs
      8.  Total row count matches n_groups x n_periods
      9.  Audit columns from fill_missing() are present
      10. Zero inflation rate — overall panel and per-group
          (flags any group exceeding zero_rate_threshold)

    Parameters
    ----------
    data                : pd.DataFrame
        Filled panel output from fill_missing()['filled_df'].
        Must contain group_col, date_col, and value_col.
    zero_rate_threshold : float
        Threshold for flagging zero inflation.
        Applied to both the overall panel zero rate and
        per-group zero rates independently. Default 0.90.
    group_col           : str
        Column identifying crime category.
        Default: _GROUP_KEY = 'fbi_code_desc'
    date_col            : str
        Column identifying observation month.
        Default: _DATE_KEY = 'year_month'
    value_col           : str
        Column containing crime counts to validate.
        Default: _COUNTER_KEY = 'crime_count'

    Returns
    -------
    None
        Prints validation results to stdout.
        Raises no exceptions - all failures are reported as
        printed messages.

    Notes
    -----
    Zero rate thresholding (Step 10) applies independently to
    the overall panel and to each group. A group above the
    threshold may reflect structural sparsity rather than
    overfilling. Cross-reference with run_integrity_report()
    coverage_ratio to distinguish the two.

    Audit columns (was_missing, is_zero_after_fill) are added
    by fill_missing(). Their absence at validation time means
    the DataFrame was not produced by fill_missing().
    """
    print("\n" + "=" * 66)
    print(f"{'CRIME DATA VALIDATION CHECK':^66}")
    print("=" * 66)

    # Step 1 - Validate date column dtype
    # CLR transformation requires datetime index for era slicing
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        print("FAIL  Date column is not datetime64")
    else:
        print("PASS  Date column is datetime64")

    # Step 2 - Check global sort order
    # Panel must be sorted by (group, date) for reindex and era slicing
    globally_sorted = (
        data
        .sort_values([group_col, date_col])
        .reset_index(drop=True)
        .equals(data.reset_index(drop=True))
    )
    if globally_sorted:
        print("PASS  Data is globally sorted by group and date")
    else:
        print("FAIL  Data is not sorted by group and date")

    # Step 3 - Check monotonic date order within each group
    # Each crime category must have strictly increasing monthly timestamps
    # Non-monotonic groups indicate panel construction errors
    bad_groups = [
        g for g, sub in data.groupby(group_col, observed=False)
        if not sub[date_col].is_monotonic_increasing
    ]
    if not bad_groups:
        print("PASS  All groups have monotonically increasing dates")
    else:
        print(f"FAIL  {len(bad_groups)} group(s) have unordered dates")
        for g in bad_groups:
            print(f"        {g}")

    # Step 4 - Check period count consistency across groups
    # All groups must span the same number of months after filling
    # Inconsistency indicates incomplete panel construction
    period_counts = data.groupby(
        group_col, observed=False
    )[date_col].nunique()

    if period_counts.nunique() == 1:
        print(f"PASS  All groups have {period_counts.iloc[0]} periods")
    else:
        print("FAIL  Inconsistent period counts across groups")
        print(
            period_counts[
                period_counts != period_counts.mode()[0]
            ].to_string()
        )

    # Step 5 - Check for negative counts
    # Negative values indicate aggregation or fill errors
    n_neg = int((data[value_col] < 0).sum())
    if n_neg == 0:
        print("PASS  No negative values")
    else:
        print(f"FAIL  {n_neg} negative count(s) detected")

    # Step 6 - Check for remaining NaNs
    # NaNs after fill_missing() indicate incomplete panel construction
    n_nan = int(data[value_col].isna().sum())
    if n_nan == 0:
        print("PASS  No NaNs remain")
    else:
        print(f"FAIL  {n_nan} NaN(s) remain after fill")

    # Step 7 - Check for duplicate (group, date) pairs
    # fill_missing() removes duplicates via _aggregate_counts
    # but downstream operations could reintroduce them
    n_dupes = int(data.duplicated([group_col, date_col]).sum())
    if n_dupes == 0:
        print("PASS  No duplicate (group, date) pairs")
    else:
        print(f"FAIL  {n_dupes} duplicate (group, date) pair(s) found")

    # Step 8 - Verify total row count matches expected panel size
    # After filling, rows must equal n_groups x n_periods exactly
    n_groups  = data[group_col].nunique()
    n_periods = data[date_col].nunique()
    expected  = n_groups * n_periods
    actual    = len(data)
    if actual == expected:
        print(f"PASS  Row count matches panel size "
              f"({n_groups} x {n_periods} = {expected:,})")
    else:
        print(f"FAIL  Row count mismatch  "
              f"expected={expected:,}  "
              f"actual={actual:,}  "
              f"diff={actual - expected:+,}")

    # Step 9 - Confirm audit columns exist from fill_missing()
    # was_missing and is_zero_after_fill are added by fill_missing()
    # Their absence means validate was called on an unfilled DataFrame
    for col in ["was_missing", "is_zero_after_fill"]:
        if col in data.columns:
            print(f"PASS  Audit column '{col}' present")
        else:
            print(f"WARN  Audit column '{col}' missing - "
                  f"was fill_missing() called before validation?")

    # Step 10 - Zero inflation diagnostic
    # Applied to overall panel and per group independently
    # Informational only at panel level - WARN if group exceeds threshold
    # Cross-reference with run_integrity_report() coverage_ratio
    # to distinguish structural sparsity from overfilling
    zero_rate = (data[value_col] == 0).mean()
    print(f"INFO  Overall zero rate : {zero_rate:.2%}")
    if zero_rate > .90:
        print(f"WARN  Overall zero rate exceeds threshold "
              f"({zero_rate:.0%}) - "
              f"check for overfilling or structural sparsity")

    print("\n" + "-" * 66)
    group_zero_rates = (
        data
        .groupby(group_col, observed=False)[value_col]
        .apply(lambda x: (x == 0).mean())
        .sort_values(ascending=False)
    )
    flagged = group_zero_rates[group_zero_rates > zero_rate_threshold]

    if not flagged.empty:
        print(f"Groups exceeding zero rate threshold "
              f"({zero_rate_threshold:.2%}):")
        for group, rate in flagged.items():
            print(f"-> {group:<45} {1-rate:.2%} zero rate")
    else:
        print(f"  No groups exceed zero rate threshold "
              f"({zero_rate_threshold:.2%})")

    print("=" * 66 + "\n")