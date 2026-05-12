import pandas as pd
from typing import Dict

# ---------------------------------------------------------------------------------
# 0. Global constants for CLR transformation and pseudocount handling
# ---------------------------------------------------------------------------------
"""
_DATE_KEY: The key for the date column in the dataset, used for grouping and analysis.
_COUNTER_KEY: The key for the count of crimes, used for aggregating crime data.
_GROUP_KEY: The key for the crime type or category, used for grouping crime data.
_EPS_GRID: A list of small values representing the Jeffreys prior grid for smoothing in 
    statistical analysis, which helps to prevent overfitting and provides a more robust 
    estimation of probabilities in the presence of sparse data.
"""
config = {
    "_DATE_KEY"     : "year_month",
    "_COUNTER_KEY"  : "crime_count",
    "_GROUP_KEY"    : "fbi_code_desc",
}

# ---------------------------------------------------------------------------------
# AGG_DICT_RESULT is a module‑level cache storing the output of _aggregate_counts().
# 
# Why this exists:
#   • Both the integrity report and the fill‑missing routines depend on the same
#     aggregated count dictionary.
#   • Computing these counts repeatedly is expensive for large datasets.
#   • By caching the result, we avoid redundant work and significantly speed up
#     sequential operations.
#
# How it works:
#   • The first call to _aggregate_counts() populates AGG_DICT_RESULT.
#   • Subsequent calls reuse the cached dictionary unless force_refresh=True is passed.
#   • force_refresh=True recomputes the aggregation when the underlying data changes.
#
# This pattern ensures correctness while providing efficient repeated access.
# ---------------------------------------------------------------------------------
AGG_DICT_RESULT = None

# ---------------------------------------------------------------------------------
# Helper function to aggregate counts safely (removing duplicates)
# ---------------------------------------------------------------------------------
def _aggregate_counts(
    data_df: pd.DataFrame,
    group_col: str = config["_GROUP_KEY"],
    date_col: str = config["_DATE_KEY"],
    counter_col: str = config["_COUNTER_KEY"],
    force_refresh: bool = False
) -> Dict:
    """
    Aggregate raw records into monthly group-level counts, with caching.

    This function groups the input DataFrame by `(group_col, date_col)` and
    computes the number of occurrences for each group–month combination.
    Dates are parsed as YYYYMM monthly timestamps, invalid dates are removed,
    and the date column is normalized to a proper datetime dtype. The group
    column is coerced to a categorical type for consistency and efficiency.

    A module‑level cache is used to avoid recomputing the aggregation on
    repeated calls. The cached result is returned unless `force_refresh=True`
    or the cache is empty.

    Returns
    -------
    Dict
        A dictionary containing:
        • "data"       : aggregated DataFrame with counts per group–month
        • "start_date" : earliest valid month in 'YYYY‑MM' format
        • "end_date"   : latest valid month in 'YYYY‑MM' format
    """   
    # MUST be declared: ensures assignments update the shared module-level 
    # cache instead of shadowing it locally
    global AGG_DICT_RESULT

    # -------------------------------------------------
    # A. Fast path: reuse previously computed aggregation
    #    Only recompute when force_refresh=True or cache is empty
    # -------------------------------------------------
    if AGG_DICT_RESULT is not None and not force_refresh:
        return AGG_DICT_RESULT

    # -------------------------------------------------
    # B. Aggregate counts:
    #    Group by (group_col, date_col) and compute the number of rows in each group.
    # -------------------------------------------------
    data = (
        data_df.groupby([group_col, date_col], sort=True, observed=False)
        .size()                     # count rows per group/date combination
        .rename(counter_col)        # rename the count column
        .reset_index()              # convert MultiIndex → flat DataFrame
    )

    # -------------------------------------------------
    # C. Convert date column to datetime (safe now that grouping is done)
    #    Using '%Y%m' ensures YYYYMM strings become proper monthly timestamps.
    # -------------------------------------------------
    data[date_col] = pd.to_datetime(data[date_col], format='%Y%m', errors='coerce')

    # -------------------------------------------------
    # D. Remove any rows with invalid dates
    #    These arise when pd.to_datetime() returned NaT
    # -------------------------------------------------
    data = data.dropna(subset=[date_col])

    # -------------------------------------------------
    # E. Guarantee datetime dtype.
    #    Even after cleaning, dtype may still be object; enforce datetime
    #    so sorting, merging, and resampling behave correctly.
    # -------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        data[date_col] = pd.to_datetime(data[date_col])

    # -------------------------------------------------
    # F. Compute the valid date range.
    #    After dropping invalid rows and enforcing datetime dtype,
    #    extract the earliest and latest timestamps and format them
    #    as YYYY‑MM for consistent monthly summaries.
    # -------------------------------------------------
    start_date = data[date_col].min().strftime('%Y-%m')
    end_date   = data[date_col].max().strftime('%Y-%m')

    # -------------------------------------------------
    # G. Normalize dtypes:
    #    Convert group column to a canonical type: string → category.
    #    (Ensures consistent grouping behavior and reduces memory usage.)
    # -------------------------------------------------
    data[group_col] = data[group_col].astype("string").astype("category")

    # -------------------------------------------------
    # H. Store results in the global cache
    #    (makes aggregated data + date range available for reuse)
    # -------------------------------------------------
    AGG_DICT_RESULT = {
        "data": data,
        "start_date": start_date,
        "end_date": end_date
    }

    return AGG_DICT_RESULT


# -------------------------------------------------------------------
# 1. Main function for integrity report (no filling, just analysis)
# -------------------------------------------------------------------
def run_integrity_report(
    data_df       : pd.DataFrame,
    force_refresh : bool = False,
    group_col     : str  = config["_GROUP_KEY"],
    date_col      : str  = config["_DATE_KEY"],
    freq          : str  = "MS"
) -> Dict:
    """
    Generate a full panel‑integrity and sparsity diagnostic for crime count data.

    This function evaluates the completeness and structural soundness of a
    monthly crime‑count panel. It aggregates raw records, constructs the full
    expected (group × month) grid, identifies true missing panel entries,
    computes per‑group coverage ratios, summarizes completeness statistics,
    and flags groups with incomplete temporal coverage.

    The report distinguishes between:
      • true panel gaps — months where a (group, month) row is entirely absent
      • structural zeros — months present in the data but with zero incidents
    Only true gaps are treated as missingness relevant for CLR preprocessing.

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw crime incident records containing at least `group_col` and `date_col`.
    force_refresh : bool, default False
        If True, recompute the monthly aggregation even when a cached result exists.
    group_col : str
        Column identifying the crime category.
    date_col : str
        Column identifying the observation month (YYYYMM or datetime‑like).
    freq : str, default "MS"
        Pandas frequency string defining the monthly grid (e.g., "MS" = month start).

    Returns
    -------
    dict
        A structured dictionary containing:
          • date_range       : (start, end) bounds of the panel in YYYY‑MM format
          • missing          : DataFrame of true missing (group, month) rows
          • missing_by_group : Series of missing‑month counts per group
          • coverage_ratio   : Series of observed / expected months per group
          • duplicates       : Number of duplicate (group, month) rows
          • expected_rows    : Total rows in the complete panel grid
          • actual_rows      : Total rows present in the aggregated data
          • missing_rows     : Number of missing panel entries
          • completeness     : Overall panel completeness ratio in [0, 1]

    Notes
    -----
    This function performs diagnostics only; it does not fill, impute, or
    modify the underlying data. It is intended to be run prior to CLR
    transformation to assess sparsity, detect structural gaps, and evaluate
    pseudocount sensitivity.

    To force regeneration of the aggregated monthly counts, call with:
        run_integrity_report(data_df, force_refresh=True)
    """

    # -------------------------------------------------
    # Step A — Aggregate raw records into monthly counts
    #    _aggregate_counts returns:
    #       - 'data'       : cleaned + aggregated DataFrame
    #       - 'start_date' : earliest YYYY‑MM in the dataset
    #       - 'end_date'   : latest YYYY‑MM in the dataset
    #    (Uses cached result unless force_refresh=True)
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df, force_refresh=force_refresh)
    data      = data_dict["data"]
    start     = data_dict["start_date"]
    end       = data_dict["end_date"]

    # -------------------------------------------------
    # Step B — Construct the complete panel structure.
    #    full_months: continuous monthly range
    #    groups: all valid group identifiers
    #    full_index: full grid (group × month) ensuring no missing combinations
    # -------------------------------------------------
    full_months = pd.date_range(start=start, end=end, freq=freq)
    groups      = data[group_col].dropna().unique()

    full_index  = pd.MultiIndex.from_product(
        [groups, full_months],
        names=[group_col, date_col]
    )

    # -------------------------------------------------
    # Step C — Detect gaps in the panel.
    #    existing_index: observed (group, date) combinations
    #    missing_index: all expected combinations not present in the data
    #    missing_df: tidy DataFrame of missing rows for downstream filling
    # -------------------------------------------------
    existing_index = pd.MultiIndex.from_frame(data[[group_col, date_col]])
    missing_index  = full_index.difference(existing_index)
    missing_df     = missing_index.to_frame(index=False)

    # -------------------------------------------------
    # Step D — Summarize missingness by group.
    #    Produces a Series: group → count of missing (group, month) entries.
    #    Sorted so the worst offenders appear first.
    # -------------------------------------------------
    missing_by_group = (
        missing_df
        .groupby(group_col, observed=False)
        .size()
        .sort_values(ascending=False)
    )
    missing_by_group = missing_by_group[missing_by_group > 0]

    # -------------------------------------------------
    # Step E — Per‑group coverage ratio
    #    Fraction of expected months actually observed for each group.
    #    coverage < 1.0 indicates at least one missing month.
    # -------------------------------------------------
    coverage_ratio = (
        data
        .groupby(group_col, observed=False)[date_col]
        .nunique()              # number of months actually present per group
        / len(full_months)      # divide by total expected months
    ).sort_values()

    # -------------------------------------------------
    # Step F — Summary statistics
    #    expected     :total (group × month) combinations in the full panel
    #    actual       : number of observed rows in the dataset
    #    missing      : number of missing (group, month) entries
    #    duplicates   : count of duplicate group–month rows in the observed data
    #    completeness : overall panel completeness ratio (1 = fully complete)
    # -------------------------------------------------
    expected     = len(full_index)
    actual       = len(data)
    missing      = len(missing_df)
    duplicates   = int(data.duplicated([group_col, date_col]).sum())
    completeness = 1 - (missing / expected)

    # -------------------------------------------------
    # Step G — Print integrity summary
    #    Provides a human‑readable overview of:
    #       - date range covered
    #       - number of groups
    #       - expected vs. actual rows
    #       - missing and duplicate entries
    #       - overall completeness ratio
    # -------------------------------------------------
    # Layout widths for summary printing
    #    width  → total line width
    # -------------------------------------------------
    width = 66
    print(f"\n{'='*width}")
    print(f"{'CRIME DATA INTEGRITY SUMMARY':^{width}}")
    print(f"{'='*width}")
    print(f"Date range    : {start} to {end}")
    print(f"Total groups  : {len(groups):,}")
    print(f"Expected rows : {expected:,}")
    print(f"Actual rows   : {actual:,}")
    print(f"Missing rows  : {missing:,}")
    print(f"Duplicates    : {duplicates:,}")
    print(f"Completeness  : {completeness:.2%}")

    # -------------------------------------------------
    # Step H — Report true panel gaps
    #    Structural zeros (count = 0) are NOT flagged here.
    #    Only months where the (group, month) row is entirely missing
    #    are reported as true gaps in the panel.
    # -------------------------------------------------
    if missing > 0:
        print(f"{'-'*width}")
        print("TRUE MISSING CRIME GAPS:")
        table = f"{(missing_by_group.to_string()):^{width}}"
        print("\n".join("    " + line for line in table.splitlines()))

    # -------------------------------------------------
    # Step I — Flag sparse and risky groups
    #    Groups with coverage < 1.0 are missing at least one month.
    #    These groups may be sensitive to pseudocount choices in CLR transforms.
    # -------------------------------------------------
    print(f"{'-'*width}")
    print("SPARSE / RISKY CRIME GROUPS:")

    sparse = coverage_ratio[coverage_ratio < 1.0]   # groups with incomplete coverage

    if not sparse.empty:
        table = f"{(sparse.to_string()):^{width}}"  # center‑aligned table for display
        print("\n".join("    " + line for line in table.splitlines()))
    else:
        print("  None - all groups fully covered across all months")

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
# 2. Main function to fill missing (group, month) entries in the panel
# -------------------------------------------------------------------
def fill_missing(
    data_df   : pd.DataFrame,
    group_col : str   = config["_GROUP_KEY"],
    date_col  : str   = config["_DATE_KEY"],
    value_col : str   = config["_COUNTER_KEY"],
    freq      : str   = "MS",
    fill_value: float = 0,
    verbose   : bool  = True
) -> Dict:
    """
    Construct a complete crime panel and fill missing group–month entries.

    This function rebuilds the full expected (group × month) panel from the
    observed crime categories and date range, aligns the aggregated data to
    that structure, and fills missing observations with a specified value.
    It preserves audit flags indicating which entries were originally absent
    and which values are zero after filling, enabling downstream analyses to
    distinguish structural zeros from true gaps.

    Parameters
    ----------
    data_df : pd.DataFrame
        Raw crime incident records containing at least group_col and date_col.
    group_col : str
        Column identifying the crime category.
    date_col : str
        Column identifying the observation month (YYYYMM or datetime-like).
    value_col : str
        Column containing the crime count to be filled.
    freq : str, default "MS"
        Pandas frequency string defining the monthly grid.
    fill_value : float, default 0
        Value used to fill missing (group, month) entries.
    verbose : bool, default True
        If True, prints a summary of the fill operation.

    Returns
    -------
    dict
        A dictionary containing:
          • date_range : (start, end) bounds of the panel in YYYY‑MM format
          • filled_df  : DataFrame with the complete panel and audit flags
                         (was_missing, is_zero_after_fill)
          • summary    : Dictionary of aggregate fill statistics

    Notes
    -----
    The returned DataFrame includes:
        was_missing        : True for entries absent before reindexing
        is_zero_after_fill : True where the final value equals zero

    This function is intended for use after run_integrity_report and before
    CLR transformation, ensuring that structural zeros and true gaps are
    explicitly tracked for pseudocount-sensitive workflows.
    """
    # -------------------------------------------------
    # Step A — Aggregate raw records into monthly counts
    #    Retrieves:
    #       • data       → cleaned monthly (group, month) counts
    #       • start_date → earliest valid month in the dataset
    #       • end_date   → latest valid month in the dataset
    #    Ensures duplicates are removed and dates normalized before panel building.
    # -------------------------------------------------
    data_dict = _aggregate_counts(data_df)
    data      = data_dict["data"]
    start     = data_dict["start_date"]
    end       = data_dict["end_date"]

    # -------------------------------------------------
    # Step B — Build the complete expected panel index
    #    unique_groups : all crime categories present in the data
    #    full_range    : continuous monthly date range for the panel
    #    full_index    : Cartesian product (group × month) defining full panel
    # -------------------------------------------------
    unique_groups = data[group_col].unique()
    full_range    = pd.date_range(
        start=start, end=end, freq=freq, name=date_col
    )
    full_index    = pd.MultiIndex.from_product(
        [unique_groups, full_range],
        names=[group_col, date_col]
    )

    # -------------------------------------------------
    # Step C — Align observed data to the full panel grid
    #    Reindex onto full_index to:
    #       • insert explicit NaNs for true missing (group, month) rows
    #       • ensure sorted, complete panel structure for downstream checks
    # -------------------------------------------------
    panel = (
        data
        .set_index([group_col, date_col])   # use (group, month) as hierarchical index
        .sort_index()                       # enforce deterministic ordering
        .reindex(full_index)                # align to full grid, introducing gaps
    )

    # -------------------------------------------------
    # Step D — Mark true panel gaps
    #    was_missing -> True where the (group, month) entry was absent
    #    (i.e., NaN introduced during reindexing)
    # -------------------------------------------------
    panel["was_missing"] = panel[value_col].isna()

    # -------------------------------------------------
    # Step E - Fill structural gaps with a chosen value
    #    Replaces NaNs (introduced during reindexing) with `fill_value`
    #    so downstream analyses (e.g., CLR transforms) have explicit counts.
    # -------------------------------------------------
    panel[value_col] = panel[value_col].fillna(fill_value)

    # -------------------------------------------------
    # Step F — Flag structural zeros after filling
    #    is_zero_after_fill -> True where the filled value equals zero
    #    Helps distinguish true zeros from originally missing entries
    # -------------------------------------------------
    panel["is_zero_after_fill"] = panel[value_col].eq(0)

    # -------------------------------------------------
    # Step G — Summarize panel fill results
    #    Provides high‑level metrics describing the constructed panel:
    #       - total_groups   : number of unique crime categories
    #       - total_periods  : number of months in the panel range
    #       - total_rows     : full size of the (group × month) panel
    #       - filled_missing : count of true gaps filled during reindexing
    #       - fill_value     : value used to replace missing entries
    # -------------------------------------------------
    summary = {
        "total_groups"  : len(unique_groups),
        "total_periods" : len(full_range),
        "total_rows"    : len(panel),
        "filled_missing": int(panel["was_missing"].sum()),
        "fill_value"    : fill_value,
    }

    # -------------------------------------------------
    # Optional — Verbose summary of fill operation
    #    Prints a compact human‑readable overview of:
    #       • date range
    #       • number of groups and periods
    #       • full panel size
    #       • number of true gaps filled
    #       • fill value applied
    # -------------------------------------------------
    # Layout widths for summary printing
    #    width  -> total line width
    # -------------------------------------------------
    if verbose:
        width = 40
        print("\n" + "=" * width)
        print(f"{'CRIME DATA FILLING SUMMARY':^{width}}")
        print("=" * width)
        print(f"Date range      : {start} to {end}")
        print(f"Total groups    : {len(unique_groups):,}")
        print(f"Total periods   : {len(full_range):,}")
        print(f"Total rows      : {len(panel):,}")
        print(f"Filled missing  : {int(panel['was_missing'].sum()):,}")
        print(f"Fill value used : {fill_value}")
        print("=" * width + "\n")

    return {
        "date_range": (start, end),
        "filled_df" : panel.reset_index(),
        "summary"   : summary,
    }


# -------------------------------------------------------------------
# 3. Validation function to check crime data integrity after filling
# -------------------------------------------------------------------
def validate_crime_data(
    data                : pd.DataFrame,
    zero_rate_threshold : float = 0.90,
    group_col           : str   = config["_GROUP_KEY"],
    date_col            : str   = config["_DATE_KEY"],
    value_col           : str   = config["_COUNTER_KEY"]
) -> None:
    """
    Validate structural integrity of a filled crime‑count panel.

    This function performs a comprehensive, multi‑stage validation of the
    DataFrame produced by `fill_missing()`. It verifies that the panel is
    complete, chronologically consistent, properly sorted, and free of
    invalid values before applying the CLR transformation.

    The validation pipeline consists of the following steps:

      A. Date column dtype is datetime64
      B. Date column is globally monotonic increasing
      C. DataFrame is globally sorted by (group, date)
      D. Dates within each group are monotonically increasing
      E. All groups have the same number of periods
      F. No negative crime‑count values
      G. No remaining NaN values after filling
      H. No duplicate (group, date) pairs
      I. Row count matches n_groups × n_periods (full panel check)
      J. Audit columns from fill_missing() are present
      K. Zero‑inflation diagnostics (overall and per‑group)

    Parameters
    ----------
    data : pd.DataFrame
        Output DataFrame from `fill_missing()['filled_df']`. Must contain
        `group_col`, `date_col`, and `value_col`.

    zero_rate_threshold : float, default 0.90
        Threshold for flagging excessive zero inflation. Applied to both
        the overall panel and individual groups.

    group_col : str, default config["_GROUP_KEY"]
        Column identifying the crime category.

    date_col : str, default config["_DATE_KEY"]
        Column identifying the observation month.

    value_col : str, default config["_COUNTER_KEY"]
        Column containing crime‑count values.

    Returns
    -------
    None
        Prints a structured validation report. No exceptions are raised;
        all issues are reported via printed messages.

    Notes
    -----
    - Zero‑rate thresholding (Step K) is applied independently to the
      overall panel and to each group. High zero rates may reflect
      structural sparsity rather than overfilling; compare with
      `run_integrity_report()` coverage ratios.

    - Audit columns (`was_missing`, `is_zero_after_fill`) are added by
      `fill_missing()`. Their absence indicates the DataFrame was not
      produced by that function.
    """
    # -------------------------------------------------
    # Step A — Print validation header
    #    Displays a standardized banner for the
    #    crime‑data validation summary section.
    # -------------------------------------------------
    # Layout widths for printing
    #    width  -> total line width
    #    width_2 -> secondary column width
    # -------------------------------------------------  
    width, width_2 = 66, 45
    print("\n" + "=" * width)
    print(f"{'CRIME DATA VALIDATION CHECK':^{width}}")
    print("=" * width)

    # -------------------------------------------------
    # Step A — Validate date column dtype
    #    CLR transformation requires a proper datetime64
    #    column to support era slicing and monthly alignment.
    # -------------------------------------------------
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        print("FAIL  Date column is not datetime64")
    else:
        print("PASS  Date column is datetime64")

    # -------------------------------------------------
    # Step B — Validate global sort order
    #    Confirms the dataset is ordered by (group, date)
    #    exactly as required for panel construction.
    #    Ensures each group’s timeline is contiguous and
    #    prevents interleaving that can break gap detection.
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Step C — Validate within‑group date ordering
    #    Ensures each crime category has dates that increase
    #    monotonically. Detects groups where the timeline is
    #    scrambled, which can break panel continuity checks.
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Step D — Validate period count consistency
    #    Ensures every group spans the same number of
    #    monthly periods. Detects truncated or irregular
    #    timelines that would break panel alignment.
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Step E — Validate non‑negativity of counts
    #    Ensures no crime‑count values are negative.
    #    Negative counts indicate data corruption or
    #    preprocessing errors that must be corrected.
    # -------------------------------------------------
    n_neg = int((data[value_col] < 0).sum())
    if n_neg == 0:
        print("PASS  No negative values")
    else:
        print(f"FAIL  {n_neg} negative count(s) detected")

    # -------------------------------------------------
    # Step F — Validate absence of NaNs after filling
    #    Confirms that all missing values were handled
    #    during preprocessing. Any remaining NaNs indicate
    #    upstream data issues requiring correction.
    # -------------------------------------------------
    n_nan = int(data[value_col].isna().sum())
    if n_nan == 0:
        print("PASS  No NaNs remain")
    else:
        print(f"FAIL  {n_nan} NaN(s) remain after fill")

    # -------------------------------------------------
    # Step G — Validate absence of duplicate (group, date) pairs
    #    Ensures each group has exactly one record per period.
    #    Duplicate (group, date) rows indicate aggregation or
    #    ingestion errors that must be resolved before panel use.
    # -------------------------------------------------
    n_dupes = int(data.duplicated([group_col, date_col]).sum())
    if n_dupes == 0:
        print("PASS  No duplicate (group, date) pairs")
    else:
        print(f"FAIL  {n_dupes} duplicate (group, date) pair(s) found")

    # -------------------------------------------------
    # Step H — Validate full panel size
    #    Confirms the dataset forms a complete panel:
    #    every group must appear in every period exactly once.
    #    Detects missing rows, truncated groups, or overfilled panels.
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Step I — Validate presence of audit columns
    #    Confirms that diagnostic columns created during
    #    fill_missing() are present. Their absence suggests
    #    the fill step may not have been executed.
    # -------------------------------------------------
    for col in ["was_missing", "is_zero_after_fill"]:
        if col in data.columns:
            print(f"PASS  Audit column '{col}' present")
        else:
            print(f"WARN  Audit column '{col}' missing - "
                  f"was fill_missing() called before validation?")

    # -------------------------------------------------
    # Step J — Evaluate zero‑value prevalence
    #    Reports overall sparsity and flags groups whose
    #    zero‑rate exceeds a configured threshold. High
    #    zero‑rates may indicate structural sparsity or
    #    over‑aggressive filling during preprocessing.
    # -------------------------------------------------
    zero_rate = (data[value_col] == 0).mean()
    print(f"INFO  Overall zero rate : {zero_rate:.2%}")
    if zero_rate > .90:
        print(f"WARN  Overall zero rate exceeds threshold "
              f"({zero_rate:.0%}) - "
              f"check for overfilling or structural sparsity")

    print("\n" + "-" * width)
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
            print(f"-> {group:<{width_2}} {1-rate:.2%} zero rate")
    else:
        print(f"  No groups exceed zero rate threshold "
              f"({zero_rate_threshold:.2%})")

    print("=" * width + "\n")