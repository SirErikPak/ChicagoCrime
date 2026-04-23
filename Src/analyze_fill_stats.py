import pandas as pd
import numpy as np
from itertools import product

# ── Helper Functions ────────────────────────────────────────────────
def prepare_era_data(data, era_label):
    """Converts year_month to datetime and assigns era label."""
    data_df = data.copy()
    data_df['date'] = pd.to_datetime(data_df['year_month'].astype(str), format='%Y%m')
    data_df['era'] = era_label
    return data_df

# ── 1. Find Missing Crime Months ────────────────────────────────────────────────
def find_missing_crime_months(data_df, era_label):
    """
    Identifies gaps in the time-series grid for a specific era.
    Returns: (expected_rows, actual_rows, missing_only_df)
    """
    d = prepare_era_data(data_df, era_label)
    
    all_months = d['date'].unique()
    all_crimes = d['fbi_code_desc'].unique()
    
    expected_rows = len(all_months) * len(all_crimes)
    actual_rows = len(d)
    
    missing_only = pd.DataFrame()
    
    if expected_rows > actual_rows:
        # Create the full Cartesian product grid
        full_grid = pd.DataFrame(
            list(product(all_months, all_crimes)),
            columns=['date', 'fbi_code_desc']
        )
        
        # Identify holes in the data
        missing_df = full_grid.merge(
            d[['date', 'fbi_code_desc']], 
            on=['date', 'fbi_code_desc'], 
            how='left', 
            indicator=True
        )
        missing_only = missing_df[missing_df['_merge'] == 'left_only'].copy()
        
    return expected_rows, actual_rows, missing_only


# ── 2. Era Integrity Report ────────────────────────────────────────────────   
def run_era_integrity_report(era_dict, expected_counts):
    """
    The main reporter function that orchestrates verification and printing.
    """
    # 1. First, verify all era month counts before printing the table
    # This acts as a 'Gatekeeper' check
    era_months_actual = {}
    for label, raw_df in era_dict.items():
        actual_months = raw_df['year_month'].nunique()
        expected_months = expected_counts.get(label, 0)
        
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
        missing_count = exp_rows - act_rows

        print(f"{label:<15} | {exp_rows:<10} | {act_rows:<10} | {missing_count}")
        
        if not missing_df.empty:
            print(f" └── Missing counts by crime category:")
            gaps = missing_df.groupby('fbi_code_desc').size().sort_values(ascending=False)
            print(gaps.to_string())
        print()

    return


# ── 3. Fill Missing Values ────────────────────────────────────────────────   
def fill_missing(data_df):
    """
    Fill missing crime/month combinations with zero counts.

    Missing rows represent months where no incidents were recorded —
    true zeros, not missing data. Required for a consistent time series
    across all 26 crimes before baseline computation and DTW.

    Parameters:
        data : pd.DataFrame - must contain columns: date, fbi_code_desc, crime_count, era

    Returns:
        pd.DataFrame - complete grid with 0-filled missing combos
    """
    # Input validation: Required columns exist
    required_cols = ['date', 'fbi_code_desc', 'crime_count', 'era']
    missing_cols  = [c for c in required_cols if c not in data_df.columns]
    if missing_cols:
        raise ValueError(f"fill_missing: missing required columns: {missing_cols}")
    if data_df.empty:
        raise ValueError("fill_missing: input dataframe is empty")

    # Get all unique values
    all_months = data_df['date'].unique()
    all_crimes = data_df['fbi_code_desc'].unique()
    era        = data_df['era'].iloc[0]
    # Every possible combination of month and crime type
    full_grid = pd.DataFrame(
        list(product(all_months, all_crimes)),
        columns=['date', 'fbi_code_desc']
    )
    full_grid['era'] = era  # adding the era label to every row in your full grid
    # aligning the complete grid with the actual data and introducing missing values where data doesn’t exist
    filled = full_grid.merge(
        data_df[['date', 'fbi_code_desc', 'crime_count']],
        on=['date', 'fbi_code_desc'],
        how='left'
    )
    # Replaces missing values with 0
    filled['crime_count'] = filled['crime_count'].fillna(0).astype(int)
    
    return filled


# ── 4. Compute Baseline Statistics ────────────────────────────────────────────────   
def compute_baseline_stats(pre_df):
    """
    Computes descriptive statistics, reliability flags, and 
    methodology routing for crime categories.
    """
    total_months = pre_df['date'].nunique()
    
    # 1. Aggregation: Core Stats
    # MAD is calculated as: Median(|x_i - median(x)|)
    baseline = (
        pre_df.groupby('fbi_code_desc')['crime_count']
        .agg(
            mean   = 'mean',
            std    = 'std',
            median = 'median',
            mad    = lambda x: (x - x.median()).abs().median(),
            months = 'count'
        )
        .reset_index()
    )

    # 2. Reliability & Presence
    baseline['cv'] = (baseline['std'] / baseline['mean'] * 100).round(1)
    
    baseline['cv_flag'] = pd.cut(
        baseline['cv'],
        bins   = [0, 30, 60, float('inf')],
        labels = ['reliable', 'caution', 'noisy']
    )

    # Calculate Presence Rate (percentage of months where count > 0)
    presence = (
        pre_df[pre_df['crime_count'] > 0]
        .groupby('fbi_code_desc')['date']
        .nunique()
        .reset_index()
        .rename(columns={'date': 'months_present'})
    )
    presence['presence_rate_pre'] = (presence['months_present'] / total_months * 100).round(1)

    # Merge Presence back
    baseline = baseline.merge(
        presence[['fbi_code_desc', 'presence_rate_pre']], 
        on='fbi_code_desc', 
        how='left'
    ).fillna({'presence_rate_pre': 0})

    # 3. Logic-Based Routing Flags
    baseline['zero_mad']    = baseline['mad'] == 0
    
    baseline['use_decomp']  = (
        baseline['cv_flag'].isin(['caution']) | 
        (baseline['fbi_code_desc'] == 'Liquor Laws')
    )
    
    baseline['use_robust']  = (
        (baseline['cv_flag'] == 'noisy') & 
        (~baseline['zero_mad'])
    )
    
    # Rare crime exception
    baseline['use_presence'] = (
        baseline['fbi_code_desc'] == 'Involuntary Manslaughter / Reckless Homicide'
    )

    return baseline


# ── 4a. Print Baseline Report ────────────────────────────────────────────────   
def print_baseline_report(baseline):
    """Prints formatted diagnostics and summary."""
    print("=== Months per crime (Target: 230) ===")
    print(baseline[['fbi_code_desc', 'months']].sort_values('months').to_string(index=False))

    print("\n=== Complete Baseline Summary ===")
    cols = [
        'fbi_code_desc', 'mean', 'std', 'median', 'mad',
        'cv', 'cv_flag', 'presence_rate_pre',
        'zero_mad', 'use_decomp', 'use_robust', 'use_presence'
    ]
    print(baseline[cols].sort_values('cv', ascending=False).to_string(index=False))

    print(f"\n=== System Health Summary ===")
    print(f"Total crime types:    {len(baseline)}")
    print(f"CV Distribution:      {baseline['cv_flag'].value_counts().to_dict()}")
    print(f"Time Grid Integrity:  {'PASS' if (baseline['months'] == 230).all() else 'FAIL'}")
    print(f"Decomp Routing:       {baseline['use_decomp'].sum()} crimes")
    print(f"Robust Z-Score:       {baseline['use_robust'].sum()} crimes")
    print(f"Presence Tracking:    {baseline['use_presence'].sum()} crimes")
    print(f"Zero MAD Alerts:      {baseline['zero_mad'].sum()} crimes")



# ── 5. Compute Metrics ────────────────────────────────────────────────   
# For each crime, compute the appropriate metric based on cv_flag and routing flags:
#   reliable  -> z_gap  (mean/std based)
#   caution   -> sadj_z (seasonally adjusted z-gap using decomp residuals)
#   noisy     -> robust_z (median/MAD based)
#   Involuntary Manslaughter -> presence_rate only
# All crimes also get r_spike, pct_change, and presence_rate as supplementary metrics.
# r_spike: is a time-series metric designed to detect abnormally large changes between consecutive time points
#
# Note: pct_change and r_spike are mathematically identical after normalization.
# They are assigned non-overlapping weights by routing category:
#   r_spike    serves reliable / use_decomp / use_robust crimes  (weight = 0.75)
#   pct_change serves use_presence crimes only                    (weight = 1.0)
# Together they function as a single magnitude signal split across two routing paths.

def compute_metrics(era_df, era_label, baseline, decomp_df):

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
        (merged['era_mean'] - merged['mean']) / merged['std']
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
    merged['r_spike'] = (merged['era_mean'] / merged['mean']).round(4)

    # Pct change: more interpretable version of r_spike
    # Easier to communicate: "+40% during COVID" vs "r_spike = 1.4"
    merged['pct_change'] = (
        (merged['era_mean'] - merged['mean']) / merged['mean'] * 100
    ).round(2)

    # Presence rate: % of era months where at least one incident occurred
    # Primary metric for Involuntary Manslaughter (median=0, MAD=0)
    # Supplementary for all other crimes
    total_months = era_df['date'].nunique()
    presence = (
        era_df[era_df['crime_count'] > 0]
        .groupby('fbi_code_desc')['date']
        .nunique()
        .reset_index()
        .rename(columns={'date': 'months_present'})
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
