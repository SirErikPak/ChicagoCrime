import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.seasonal import STL
import warnings

# ── Helper Functions ────────────────────────────────────────────────
def prepare_era_data(data, era_label):
    """Converts year_month to datetime and assigns era label."""
    data_df = data.copy()
    data_df['year_month'] = pd.to_datetime(data_df['year_month'].astype(str), format='%Y%m')
    data_df['era'] = era_label
    return data_df


# ── 1. Find Missing Crime Months ────────────────────────────────────────────────
def find_missing_crime_months(data_df: pd.DataFrame, era_label: str):
    """
    Identifies gaps in the time-series grid for a specific era by comparing 
    actual observations against a complete Cartesian product of dates and categories.

    Args:
        data_df (pd.DataFrame): The crime dataset. Must contain 'year_month' and 
            'fbi_code_desc'.('year_month' is derived internally via prepare_era_data).
        era_label (str): A descriptive label for the time period being analyzed.

    Returns:
        tuple: (expected_rows, actual_rows, missing_only_df)
            - expected_rows (int): Theoretical count (months * categories).
            - actual_rows (int): The current row count in the era.
            - missing_only_df (pd.DataFrame): Rows present in the grid but not in data.

    Raises:
        ValueError: If required columns are missing from data_df.
    """
    # 1. Input Validation
    required_cols = {'year_month', 'fbi_code_desc'}
    if not required_cols.issubset(data_df.columns):
        missing = required_cols - set(data_df.columns)
        raise ValueError(
            f"find_missing_crime_months failed: data_df missing required columns: {missing}"
        )

    # 2. Preparation and Grid Calculation
    d = prepare_era_data(data_df, era_label)
    
    all_months = d['year_month'].unique()
    all_crimes = d['fbi_code_desc'].unique()
    
    expected_rows = len(all_months) * len(all_crimes)
    actual_rows = len(d)
    
    # Initialize an empty DataFrame for missing combinations
    missing_only = pd.DataFrame(columns=['year_month', 'fbi_code_desc'])
    
    # 3. Cartesian Product Logic
    if expected_rows > actual_rows:
        # Create the full Cartesian product grid
        full_grid = pd.DataFrame(
            list(product(all_months, all_crimes)),
            columns=['year_month', 'fbi_code_desc']
        )
        
        # Identify holes in the data using a left join indicator
        missing_df = full_grid.merge(
            d[['year_month', 'fbi_code_desc']], 
            on=['year_month', 'fbi_code_desc'], 
            how='left', 
            indicator=True
        )
        missing_only = missing_df[missing_df['_merge'] == 'left_only'].drop(columns='_merge').copy()
        
    return expected_rows, actual_rows, missing_only


# ── 2. Era Integrity Report ────────────────────────────────────────────────   
def run_era_integrity_report(era_dict, expected_counts):
    """
    Orchestrates the integrity verification of crime data eras and generates 
        a diagnostic missingness report.

        This function operates in two primary phases:
        1. Temporal Gatekeeper: Validates that the number of unique months in 
        each era matches expectations. If a mismatch is found, it raises an 
        AssertionError to prevent downstream analysis on truncated data.
        2. Data Density Audit: Performs a row-level comparison to identify 
        missing crime category/month combinations and prints a formatted 
        summary of gaps and potential duplicates.

        Args:
            era_dict (dict): Dictionary mapping era labels (str) to long-form 
                DataFrames containing 'year_month' and 'fbi_code_desc'.
            expected_counts (dict): Dictionary mapping era labels (str) to the 
                total number of unique months expected (int) for that period.

        Raises:
            AssertionError: If the unique month count in any era DataFrame 
                does not match the provided expected_counts.

        Output:
            Prints a summary table to stdout, including expected vs. actual 
            row counts and a breakdown of missing months grouped by crime category.
    """
    # 1. First, verify all era month counts before printing the table
    # This acts as a 'Gatekeeper' check
    era_months_actual = {}
    for label, raw_df in era_dict.items():
        actual_months = raw_df['year_month'].nunique()
        expected_months = expected_counts.get(label, 0)

        if label not in expected_counts:
            raise KeyError(f"run_era_integrity_report: no expected_counts entry for era '{label}'")
        expected_months = expected_counts[label]

        assert actual_months == expected_months, (
            f"Era month mismatch: '{label}' expected {expected_months} months, got {actual_months}. "
            f"Check feather file era boundary definitions."
        )
        era_months_actual[label] = actual_months

    # If the loop finishes without an assertion error, print the success message
    print(f"Era month counts verified: {era_months_actual}")

    # 2. Proceed to the detailed row-level report
    print(f"\n{'Era':<15} | {'Expected':<10} | {'Actual':<10} | {'Missing'}")
    print("-" * 55)

    for label, raw_df in era_dict.items():
        exp_rows, act_rows, missing_df = find_missing_crime_months(raw_df, label)
        actual_missing  = len(missing_df)
        duplicate_count = act_rows - (exp_rows - actual_missing)
        if duplicate_count > 0:
            print(f"WARNING: {duplicate_count} duplicate rows detected")  
        print(f"{label:<15} | {exp_rows:<10} | {act_rows:<10} | {actual_missing}")
        
        if not missing_df.empty:
            print(f" └── Missing counts by crime category:")
            gaps = missing_df.groupby('fbi_code_desc').size().sort_values(ascending=False)
            print(gaps.to_string())
        print()


# ── 3. Fill Missing Values ────────────────────────────────────────────────   
def fill_missing(data_df):
    """
    Fill missing crime/month combinations with zero counts.

    Missing rows represent months where no incidents were recorded —
    true zeros, not missing data. Required for a consistent time series
    across all 26 crimes before baseline computation and DTW.

    Parameters:
        data_df : pd.DataFrame - must contain columns: year_month, fbi_code_desc, crime_count, era

    Returns:
        pd.DataFrame - complete grid with 0-filled missing combos
    """
    # Input validation: Required columns exist
    required_cols = ['year_month', 'fbi_code_desc', 'crime_count', 'era']
    missing_cols  = [c for c in required_cols if c not in data_df.columns]
    if missing_cols:
        raise ValueError(f"fill_missing: missing required columns: {missing_cols}")
    if data_df.empty:
        raise ValueError("fill_missing: input dataframe is empty")

    # Get all unique values
    all_months = data_df['year_month'].unique()
    all_crimes = data_df['fbi_code_desc'].unique()
    eras = data_df['era'].unique()
    if len(eras) > 1:
        raise ValueError(f"fill_missing: expected 1 era, found {len(eras)}: {eras}")
    era = eras[0]
    # Every possible combination of month and crime type
    full_grid = pd.DataFrame(
        list(product(all_months, all_crimes)),
        columns=['year_month', 'fbi_code_desc']
    )
    # Adding the era label to every row in your full grid
    full_grid['era'] = era
    # aligning the complete grid with the actual data and introducing missing values where data doesn’t exist
    filled = full_grid.merge(
        data_df[['year_month', 'fbi_code_desc', 'crime_count']],
        on=['year_month', 'fbi_code_desc'],
        how='left'
    )
    # Replaces missing values with 0
    filled['crime_count'] = filled['crime_count'].fillna(0).astype(int)
    
    return filled


# ── 4. Compute Baseline Statistics ────────────────────────────────────────────────   
# Routing thresholds
_PRESENCE_RATE_THRESH     = 30.0   # % of months with any count; below -> presence
_MEAN_FLOOR               = 2.0    # avg monthly count; below -> presence (guards false STL structure)
_CV_ROBUST_THRESH         = 60.0   # CV %; above -> robust only if unstructured
_STL_MIN_MONTHS           = 24     # minimum months required to fit STL reliably
# Layer 1: Structural eligibility
_ABS_TREND_THRESH         = 0.75   # Absolute threshold for trend strength to qualify for decomposition
_ABS_SEASONAL_THRESH      = 0.60   # Absolute threshold for seasonal strength to qualify for decomposition
# Absolute override
_STRONG_TREND_OVERRIDE    = 0.85   # Strong trend override threshold — if exceeded, route to decomp regardless of other metrics
_STRONG_SEASONAL_OVERRIDE = 0.75   # Strong seasonal override threshold — if exceeded, route to decomp regardless of other metrics

# Domain knowledge──
# Calibrated against Chicago crime data pre-intervention period.
# Entries reflect cases where domain expectation and STL routing agree.
# Not independently validated — revisit if:
#   - pre-period window changes
#   - new crime categories are added or removed
_KNOWN_ROUTES: dict[str, str] = {
    # Strong seasonal + trend signals confirmed by STL
    'Motor Vehicle Theft': 'decomp',
    'Burglary':            'decomp',
    'Robbery':             'decomp',
    # High CV but structure is real — enforcement/policy cycles
    'Gambling':            'decomp',
    'Prostitution':        'decomp',
    # Near-perfect trend (0.975) — standard assumes stationarity, will bias baseline
    'Drug Abuse Violations': 'decomp',
    # Genuinely sparse
    'Involuntary Manslaughter / Reckless Homicide': 'presence',
    # Business decision: stationary baselines despite detectable trend
    # Fraud trend (0.94) reflects long-run reporting changes, not true seasonality
    # Revisit if pre-period is extended beyond the current window
    'Fraud':                      'standard',
    'Forgery and Counterfeiting': 'standard',
}

# Business overrides (supersede statistical routing)
# Derive _FORCE_STANDARD from _KNOWN_ROUTES:
_FORCE_STANDARD: set[str] = {k for k, v in _KNOWN_ROUTES.items() if v == 'standard'}

def _verify_routing(baseline: pd.DataFrame) -> None:
    """
    Warns when data-driven routing disagrees with domain expectations.
    Diagnostic only — never mutates the baseline.
    """
    flag_cols = ['use_presence', 'use_decomp', 'use_robust', 'use_standard']
    for crime, expected in _KNOWN_ROUTES.items():
        row = baseline.loc[baseline['fbi_code_desc'] == crime]
        if row.empty:
            continue
        # Identify which routing flag is True for this crime
        true_flags = [col for col in flag_cols if row[col].iloc[0]]
        if not true_flags:
            warnings.warn(f"No routing flag set for '{crime}'", UserWarning)
            continue
        # There should be exactly one True flag due to the integrity check 
        # in compute_baseline_stats
        actual = true_flags[0].replace('use_', '')

        if actual != expected:
            warnings.warn(
                f"Routing mismatch — '{crime}': "
                f"expected '{expected}', got '{actual}' "
                f"(seasonal={row['seasonal_strength'].iloc[0]:.2f}, "
                f"trend={row['trend_strength'].iloc[0]:.2f}, "
                f"presence={row['presence_rate'].iloc[0]:.1f}%, "
                f"cv={row['cv'].iloc[0]:.1f})",
                UserWarning,
                stacklevel=2,
            )


def compute_baseline_stats(pre_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates crime categories and routes them to optimal baseline models via a 
    multi-layered hierarchical decision matrix.

    This function acts as a diagnostic engine that classifies time series based on 
    sparsity, temporal structure (STL), and volatility. It ensures each crime 
    category is assigned to exactly one mutually exclusive modeling path.

    Decision Hierarchy & Logic:
    1. Sparse (use_presence): 
    Triggered if presence rate < _PRESENCE_RATE_THRESH, MAD is 0 (constant counts), 
    or mean monthly counts < _MEAN_FLOOR. These series lack sufficient density 
    for distributional modeling.
    
    2. Structured (use_decomp): 
    Triggered for non-sparse series meeting either:
    - Absolute Override: Trend > 0.85 or Seasonal > 0.75.
    - Relative Strength: Valid structure (Trend > 0.75 or Seasonal > 0.60) 
        AND a weighted structure Z-score > 0.5.
    Uses STL decomposition to isolate trend/seasonality before baseline calculation.

    3. Noisy (use_robust): 
    Non-sparse, non-structured series with high volatility (CV > _CV_ROBUST_THRESH). 
    Routes to Median/MAD-based robust Z-scores to mitigate outlier influence.

    4. Stable (use_standard): 
    The default path for stationary, low-variance, and well-behaved series. 
    Uses standard parametric Mean/StdDev Z-scores.

    Business Overrides:
    - Categories in _FORCE_STANDARD are hard-routed to use_standard regardless of metrics.

    Args:
        pre_df (pd.DataFrame): Long-form monthly crime data. 
            Required columns: ['year_month', 'fbi_code_desc', 'crime_count'].

    Returns:
        pd.DataFrame: A diagnostic summary (one row per fbi_code_desc) containing:
            - Descriptive stats (mean, std, median, mad, cv).
            - Presence metrics (months_present, presence_rate).
            - STL components (seasonal_strength, trend_strength).
            - Boolean routing flags (use_presence, use_decomp, use_robust, use_standard).

    Mathematical Basis:
        - Strength of Trend (Ft): max(0, 1 - Var(R) / Var(R+T))
        - Strength of Seasonality (Fs): max(0, 1 - Var(R) / Var(R+S))
        - Structure Score: 0.6 * Ft + 0.4 * Fs
    """
    
    # 1. Define study period — validate fill_missing was applied first
    months_per_crime = pre_df.groupby('fbi_code_desc')['year_month'].nunique()
    total_months = months_per_crime.max()

    assert (months_per_crime == total_months).all(), (
        f"Uneven month counts per crime — was fill_missing skipped? "
        f"Min: {months_per_crime.min()}, Max: {months_per_crime.max()}"
    )

    # 2. Descriptive statistics
    baseline = (
        pre_df.groupby('fbi_code_desc')['crime_count']
        .agg(
            mean='mean',
            std='std',
            median='median',
            mad=lambda x: (x - x.median()).abs().median(),
            months='count',
        )
        .reset_index()
        .assign(
            zero_mad=lambda d: d['mad'] == 0,
            cv=lambda d: np.where(d['mean'] > 0,
                                  (d['std'] / d['mean']) * 100,
                                  np.nan).round(1),
        )
    )

    baseline['cv_flag'] = pd.cut(
        baseline['cv'],
        bins=[-np.inf, 30, _CV_ROBUST_THRESH, float('inf')],
        labels=['reliable', 'caution', 'noisy'],
    )

    # 3. Presence rate
    presence = (
        pre_df.loc[pre_df['crime_count'] > 0]
        .groupby('fbi_code_desc')['year_month']
        .nunique()
        .rename('months_present')
        .reset_index()
        .assign(presence_rate=lambda d: (d['months_present'] / total_months * 100).round(1))
    )

    baseline = baseline.merge(
        presence[['fbi_code_desc', 'months_present', 'presence_rate']],
        on='fbi_code_desc',
        how='left'
    ).fillna({'presence_rate': 0.0, 'months_present': 0})

    # 4. STL strengths
    def _stl_strengths(sub: pd.DataFrame) -> pd.Series:
        ts = sub.sort_values('year_month')['crime_count'].to_numpy(dtype=float)

        if len(ts) < _STL_MIN_MONTHS:
            return pd.Series({'seasonal_strength': 0.0, 'trend_strength': 0.0})

        try:
            fit = STL(ts, period=12, robust=True).fit()
            var_R = np.nanvar(fit.resid)

            def _strength(component):
                var_RC = np.nanvar(fit.resid + component)
                if var_RC <= 0 or np.isnan(var_RC):
                    return 0.0
                return float(np.clip(1 - var_R / var_RC, 0.0, 1.0))

            return pd.Series({
                'seasonal_strength': _strength(fit.seasonal),
                'trend_strength': _strength(fit.trend),
            })

        except (ValueError, np.linalg.LinAlgError):
            return pd.Series({'seasonal_strength': 0.0, 'trend_strength': 0.0})

    strengths = (
        pre_df.groupby('fbi_code_desc')
        .apply(_stl_strengths, include_groups=False)
        .reset_index()
    )
  
    baseline = baseline.merge(strengths, on='fbi_code_desc', how='left')

    # 5. ROUTING SYSTEM
    # Sparse check
    is_sparse = (baseline['presence_rate'] < _PRESENCE_RATE_THRESH) | \
                (baseline['zero_mad']) | \
                (baseline['mean'] < _MEAN_FLOOR)
    # Layer 1: Structural validity (absolute thresholds)
    is_structurally_valid = (
        (baseline['trend_strength'] > _ABS_TREND_THRESH) |
        (baseline['seasonal_strength'] > _ABS_SEASONAL_THRESH)
    )
    # Absolute override for strong structure regardless of peer comparison
    is_strong_absolute = (
        (baseline['trend_strength'] > _STRONG_TREND_OVERRIDE) |
        (baseline['seasonal_strength'] > _STRONG_SEASONAL_OVERRIDE)
    )

    # Layer 2: Relative scoring
    structure_score = (
        0.6 * baseline['trend_strength'] +
        0.4 * baseline['seasonal_strength']
    )
    # Z-score normalization to identify strong structure relative to peers
    score_z = (
        structure_score - structure_score.mean()
    ) / (structure_score.std() + 1e-8)

    is_strong_structure = score_z > 0.5
    is_noisy = baseline['cv'] > _CV_ROBUST_THRESH

    # Final routing (correct hierarchy)
    baseline['use_presence'] = is_sparse

    baseline['use_decomp'] = (
        ~is_sparse &
        (
            is_strong_absolute |                      # ✅ override
            (is_structurally_valid & is_strong_structure)
        )
    )
    # Noisy but not structurally valid series get routed to robust layer, 
    # which uses the same Z-score metric but with median/MAD instead of mean/std    
    baseline['use_robust'] = (
        ~is_sparse &
        ~baseline['use_decomp'] &
        is_noisy
    )
    # Stable, well-behaved series get the standard parametric approach
    baseline['use_standard'] = (
        ~is_sparse &
        ~baseline['use_decomp'] &
        ~baseline['use_robust']
    )

    # 6. Override layer (business rules)
    force_standard = baseline['fbi_code_desc'].isin(_FORCE_STANDARD)
    baseline.loc[force_standard, 'use_standard'] = True
    # After both assignments, re-cast all four routing cols to ensure clean bool dtype:
    for col in ['use_presence', 'use_decomp', 'use_robust', 'use_standard']:
        baseline[col] = baseline[col].astype(bool)

    # 7. Integrity check
    assert (baseline[['use_presence', 'use_decomp', 'use_robust', 'use_standard']].sum(axis=1) == 1).all()

    _verify_routing(baseline)

    return baseline


# ── 4a. Print Baseline Report ──────────────────────────────────────────────── 
def print_baseline_report(baseline, expected_months=None, verbose=False):
    """
    Prints formatted diagnostics and summary.
 
    Time Grid Integrity verifies that each crime category contains the full expected 
    number of monthly observations in the baseline period. Any mismatch signals incomplete 
    preprocessing, such as skipped missing-value handling or misaligned era boundaries, which 
    can bias all downstream statistics. Ensuring a complete and consistent time grid is 
    essential for reliable baseline estimation and valid comparisons.
 
    Args:
        baseline (pd.DataFrame): Output of compute_baseline_stats.
        expected_months (int, optional): Expected month count per crime type.
            If omitted, the grid integrity check is skipped.
        verbose (bool): If True, prints per-crime month counts before the summary.
    """
    if verbose:
        print("=== Months per crime ===")
        print(baseline[['fbi_code_desc', 'months']].sort_values('months').to_string(index=False))
    cols = [c for c in baseline.columns if c != 'months']
    print("\n=== Complete Baseline Summary ===")
    print(baseline[cols].sort_values('cv', ascending=False).to_string(index=False, justify='left',
                                     formatters={
                                            'mean': "{:.4f}".format,
                                            'std': "{:.4f}".format,
                                            'seasonal_strength': "{:.4f}".format,
                                            'trend_strength': "{:.4f}".format,
        }))

    print(f"\n=== System Health Summary ===")
    print(f"Total crime types:    {len(baseline)}")
    print(f"CV Distribution:      {baseline['cv_flag'].value_counts().to_dict()}")
    # The expected_months check is a critical integrity gatekeeper. A mismatch suggests
    # fill_missing was skipped or era boundaries shifted, which would bias all downstream stats
    if expected_months is not None:
        grid_ok = (baseline['months'] == expected_months).all()
        print(f"Time Grid Integrity:  {'PASS' if grid_ok else 'FAIL'} (expected {expected_months} months)")
    else:
        print(f"Time Grid Integrity:  (skipped — no expected_months provided)")
    print(f"Decomp Routing:       {baseline['use_decomp'].sum()} crimes")
    print(f"Robust Z-Score:       {baseline['use_robust'].sum()} crimes")
    print(f"Standard Baseline:    {baseline['use_standard'].sum()} crimes")
    print(f"Presence Tracking:    {baseline['use_presence'].sum()} crimes")
    print(f"Zero MAD Alerts:      {baseline['zero_mad'].sum()} crimes")


# ── 5. Compute Metrics ────────────────────────────────────────────────  
def compute_metrics(era_df, era_label, baseline, decomp_df):
    """
    Computes per-crime deviation metrics for a given era against the pre-period baseline.

    Each crime is assigned its primary metric based on its routing flag:
        use_standard  -> z_gap    : standard z-score (mean/std)
        use_decomp    -> sadj_z   : seasonally adjusted z-gap (STL residual mean/std)
        use_robust    -> robust_z : robust z-score (median/MAD)
        use_presence  -> pct_change : magnitude of presence change

    All crimes additionally receive r_spike, pct_change, and presence_rate as
    supplementary signals regardless of routing.

    Note: pct_change and r_spike are mathematically equivalent after normalization.
    They are assigned non-overlapping weights by routing category — r_spike serves
    standard/decomp/robust crimes (weight=0.75), pct_change serves presence crimes
    (weight=1.0) — functioning as a single magnitude signal split across two paths.

    Args:
        era_df (pd.DataFrame): Long-form monthly crime data for the target era.
            Required columns: ['fbi_code_desc', 'crime_count', 'year_month'].
        era_label (str): Label assigned to the 'era' column in the output.
        baseline (pd.DataFrame): Output of compute_baseline_stats. Must contain
            routing flags and descriptive stats per crime type.
        decomp_df (pd.DataFrame): STL decomposition residual stats. Must contain
            ['fbi_code_desc', 'resid_mean', 'resid_std'].

    Returns:
        pd.DataFrame: One row per crime type with all computed metrics and the
            era label attached.
    """ 
    # Per-crime summary stats for this era
    era_stats = (
        era_df.groupby('fbi_code_desc')['crime_count']
        .agg(era_mean='mean', era_median='median')
        .reset_index()
    )

    # Merge with baseline routing flags and stats
    merged = era_stats.merge(
        baseline[[
            'fbi_code_desc', 'mean', 'std', 'median', 'mad',
            'cv', 'cv_flag', 'zero_mad', 'use_decomp',
            'use_robust', 'use_presence'
        ]],
        on='fbi_code_desc'
    )

    # Standard z-gap: how many std deviations from baseline mean
    # Reliable for low-CV crimes, less reliable for noisy ones
    merged['z_gap'] = (
        (merged['era_mean'] - merged['mean']) / merged['std'].replace(0, float('nan'))
    ).round(4)

    # Robust z-score: uses median and MAD instead of mean and std
    # More resistant to outliers - used for noisy crimes with non-zero MAD
    # 1.4826 rescales MAD to be comparable to std under a normal distribution
    merged['robust_z'] = (
        (merged['era_median'] - merged['median']) /
        (1.4826 * merged['mad'].replace(0, float('nan')))
    ).round(4)

    # R-spike: ratio of era average to baseline average
    # 1.0 = no change, 1.5 = 50% increase, 0.7 = 30% decrease
    safe_mean = merged['mean'].replace(0, float('nan'))
    merged['r_spike']    = (merged['era_mean'] / safe_mean).round(4)
    # Pct change: percentage increase/decrease from baseline mean
    merged['pct_change'] = ((merged['era_mean'] - safe_mean) / safe_mean * 100).round(2)

    # Presence rate: % of era months where at least one incident occurred
    # Primary metric for Involuntary Manslaughter (median=0, MAD=0)
    # Supplementary for all other crimes
    total_months = era_df['year_month'].nunique()
    presence = (
        era_df[era_df['crime_count'] > 0]
        .groupby('fbi_code_desc')['year_month']
        .nunique()
        .reset_index()
        .rename(columns={'year_month': 'months_present'})
    )
    presence['presence_rate'] = (
        presence['months_present'] / total_months * 100
    ).round(1)
    merged = merged.merge(
        presence[['fbi_code_desc', 'presence_rate']],
        on='fbi_code_desc',
        how='left'
    )
    merged['presence_rate'] = merged['presence_rate'].fillna(0)

    # Seasonally adjusted z-gap: uses decomp residual mean and std as reference
    # Removes seasonal variance before measuring COVID deviation
    # Only computed for caution crimes + Liquor Laws (use_decomp == True)
    # For all others, sadj_z is NaN
    #
    # Implementation: vectorized merge replaces the iterrows anti-pattern
    # iterrows() disables vectorization and is O(n) Python loop overhead.
    # Reference: https://pandas.pydata.org/docs/user_guide/enhancingperf.html
    decomp_lookup = decomp_df[['fbi_code_desc', 'resid_mean', 'resid_std']].copy()
    merged        = merged.merge(decomp_lookup, on='fbi_code_desc', how='left')
    merged['sadj_z'] = np.where(
        merged['resid_std'].notna() & (merged['resid_std'] > 0),
        ((merged['era_mean'] - merged['resid_mean']) / merged['resid_std']).round(4),
        float('nan')
    )
    merged = merged.drop(columns=['resid_mean', 'resid_std'])

    merged['era'] = era_label
    return merged
