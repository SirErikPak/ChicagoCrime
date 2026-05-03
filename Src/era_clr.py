
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.multitest import multipletests


# -------------------------------------------------------------------
# 1. Function to slice CLR matrix into eras
# --------------------------------------------------------------------
def slice_clr_into_eras(clr_dict, eps=0.02):
    """
    Slice a CLR matrix into Pre‑COVID, COVID, and Post‑COVID eras using
    fixed administrative boundaries and verified shapes + date alignment.
    """
    # Extract the CLR DataFrame for the specified epsilon
    clr_df = clr_dict[eps] # DataFrame (300, 26) - K=26 pre-exclusion

    # Administrative boundaries based on crime data availability and COVID timeline:
    PRE_END   = 230   # Jan 2001 – Feb 2020
    COVID_END = 264   # Mar 2020 – Dec 2022

    # Slice eras based on verified indices and expected date ranges
    pre_covid  = clr_df.iloc[:PRE_END]
    covid      = clr_df.iloc[PRE_END:COVID_END]
    post_covid = clr_df.iloc[COVID_END:]

    # Verification prints to confirm shapes and date alignment
    print("CLR matrix shape:", clr_df.shape)
    print("First 3 index values:", clr_df.index[:3].tolist())
    print("Last 3 index values:", clr_df.index[-3:].tolist())

    print(f"\n{'Era Shape Verification:':<15}")
    print(f"{'Pre-COVID':<12} : {str(pre_covid.shape):<10} -> expected (230, 26)")
    print(f"{'COVID':<12} : {str(covid.shape):<10} -> expected (34, 26)")
    print(f"{'Post-COVID':<12} : {str(post_covid.shape):<10} -> expected (36, 26)")

    print(f"\n{'Era boundary verification:':<20}")
    print(f"{'Pre-COVID ends':<18}: {pre_covid.index[-1].strftime('%Y-%m')} -> expected 2020-02")
    print(f"{'COVID starts':<18}: {covid.index[0].strftime('%Y-%m')} -> expected 2020-03")
    print(f"{'COVID ends':<18}: {covid.index[-1].strftime('%Y-%m')} -> expected 2022-12")
    print(f"{'Post-COVID starts':<18}: {post_covid.index[0].strftime('%Y-%m')} -> expected 2023-01")
    print(f"{'Post-COVID ends':<18}: {post_covid.index[-1].strftime('%Y-%m')} -> expected 2025-12")

    return {'pre_covid':  pre_covid, 'covid': covid, 'post_covid': post_covid}


# -------------------------------------------------------------------
# 2. Function to compute distributional parameters for each era
# --------------------------------------------------------------------
def compute_era_distribution_parameters(eras_dict, 
                                        top_display = True, 
                                        n_rows = 10):
    """
    Compute per-era distributional parameters (mean, std, covariance)
    and absolute differences between eras.
    """

    # Normalize keys
    eras = {
        'Pre-COVID': eras_dict['pre_covid'],
        'COVID': eras_dict['covid'],
        'Post-COVID': eras_dict['post_covid']
    }

    # Mean vector per era (central tendency of crime distribution in CLR space)
    era_means = pd.DataFrame({name: df.mean(axis=0) for name, df in eras.items()})

    # Std deviation per era (variability around mean)
    era_stds = pd.DataFrame({name: df.std(axis=0) for name, df in eras.items()})

    # Covariance matrices per era (interrelationships between crime types in CLR space)
    era_covs = {name: df.cov() for name, df in eras.items()}

    # Differences between eras (magnitude of distributional shifts)
    era_means["Pre_minus_COVID"] = (era_means["Pre-COVID"] - era_means["COVID"])
    era_means["COVID_minus_Post"] = (era_means["COVID"] - era_means["Post-COVID"])
    # Note: Std dev differences are less interpretable than mean shifts, but we include them for completeness.
    era_stds["Pre_minus_COVID"] = (era_stds["Pre-COVID"] - era_stds["COVID"])
    era_stds["COVID_minus_Post"] = (era_stds["COVID"] - era_stds["Post-COVID"])

    # --- Print summary ---
    print("=" * 110)
    print("MEAN CLR VECTOR PER ERA (rows=crime type, cols=era & comparisons)")
    print("=" * 110)
    print(era_means.round(4).to_string())
    print()

    print("=" * 110)
    print("STD DEV PER ERA")
    print("=" * 110)
    print(era_stds.round(4).to_string())
    print()

    print("=" * 110)
    print(f"{'COVARIANCE MATRIX SHAPES':^60}")
    print("=" * 110)
    for name, cov in era_covs.items():
        print(f" {name:<12}: {str(cov.shape):<10} -> expected (26, 26)")

    if top_display:
        # Display top mean shifts for interpretability (sorted by absolute magnitude)
        print("\n" + "=" * 70)
        print(f"TOP {n_rows} MEAN SHIFTS")
        print("=" * 70)
        print("Note: Differences between eras, sorted by Absolute magnitude.")
        print("Pre-COVID -> COVID and COVID -> Post-COVID are ranked separately.")
        print("This highlights which crime types had the largest distributional shifts.")
        print(f"{'-' * 70}\n")

        # Rank mean shifts Pre -> COVID
        print(f"TOP {n_rows} MEAN SHIFTS - Pre-COVID -> COVID")
        print("=" * 55)
        # Reindex to sort by absolute  difference, then round for display
        top_pre_covid = (
        era_means['Pre_minus_COVID']
        .reindex(
            era_means['Pre_minus_COVID']
            .abs()
            .sort_values(ascending=False)
            .head(n_rows)
            .index
        )
        .round(4)
        )
        print(top_pre_covid.to_string())

        print()

        # Rank mean shifts COVID -> Post-COVID
        print(f"TOP {n_rows} MEAN SHIFTS - COVID -> Post-COVID")
        print("=" * 55)
        top_covid_post = (
        era_means['COVID_minus_Post']
        .reindex(
            era_means['COVID_minus_Post']
            .abs()
            .sort_values(ascending=False)
            .head(n_rows)
            .index
        )
        .round(4)
        )
        print(top_covid_post.to_string())

    # Volatility shifts (std dev differences) are less interpretable but we include them for completeness
    print("\n" + "=" * 70)
    print(f"TOP {n_rows} VOLATILITY SHIFTS")
    print("=" * 70)
    print("Note: Differences in variance between eras, sorted by absolute magnitude.")
    print("Pre-COVID -> COVID and COVID -> Post-COVID are ranked separately.")
    print("This highlights which crime types had the largest volatility shifts.")
    print(f"{'-' * 70}\n")

    # Rank volatility shifts Pre -> COVID
    print(f"TOP {n_rows} VOLATILITY SHIFTS - Pre-COVID -> COVID")
    print("=" * 55)
    top_var_pre_covid = (
        era_stds['Pre_minus_COVID']
        .reindex(
            era_stds['Pre_minus_COVID']
            .abs()
            .sort_values(ascending=False)
            .head(n_rows)
            .index
        )
        .round(4)
    )
    print(top_var_pre_covid.to_string())

    print()

    # Rank volatility shifts COVID -> Post-COVID
    print(f"TOP {n_rows} VOLATILITY SHIFTS - COVID -> Post-COVID")
    print("=" * 55)
    top_var_covid_post = (
        era_stds['COVID_minus_Post']
        .reindex(
            era_stds['COVID_minus_Post']
            .abs()
            .sort_values(ascending=False)
            .head(n_rows)
            .index
        )
        .round(4)
    )
    print(top_var_covid_post.to_string())


    return {
        'era_means': era_means,
        'era_stds': era_stds,
        'era_covs': era_covs
    }


# -------------------------------------------------------------------
# 3. Function to plot heatmaps of mean CLR per era and shifts
# -------------------------------------------------------------------
def plot_mean_era_heatmaps(data_dict, save_image=None, verbose=False):
    # Define the columns and titles to iterate through
    plot_configs = [
        (['Pre-COVID', 'COVID', 'Post-COVID'], 'Mean CLR per Era', 'CLR Mean'),
        (['Pre_minus_COVID'], 'Mean CLR Shift\nPre-COVID -> COVID', 'Δ CLR'),
        (['COVID_minus_Post'], 'Mean CLR Shift\nCOVID -> Post-COVID', 'Δ CLR')
    ]

    # Sort by absolute Pre_minus_COVID for readability
    era_means_sorted = data_dict['era_means']['Pre_minus_COVID'].reindex(
        data_dict['era_means']['Pre_minus_COVID']
        .abs()
        .sort_values(ascending=False)
        .index
    )
    
    # Sort the data once
    era_means_sorted = data_dict['era_means'].reindex(
        data_dict['era_means']['Pre_minus_COVID'].abs().sort_values(ascending=False).index
    )
    
    # Plotting Loop
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    
    for ax, (cols, title, label) in zip(axes, plot_configs):
        sns.heatmap(
            era_means_sorted[cols],
            ax=ax,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            linewidths=0.4,
            cbar_kws={'label': label}
        )
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Global Styling
    plt.suptitle(
        'Mean CLR per Era & Shifts - Chicago Crime 2001–2025\n'
        'Sorted by |Pre_minus_COVID|',
        fontsize=14, fontweight='bold', y=1.02 # Increased y for better spacing
    )
    
    plt.tight_layout()
    if save_image:
        plt.savefig(f"{save_image}clr_mean_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    if verbose:
        # Print
        print("\nSorted by |Pre_minus_COVID|:")
        cols_to_print = ['Pre-COVID', 'COVID', 'Post-COVID', 'Pre_minus_COVID', 'COVID_minus_Post']
        print(era_means_sorted[cols_to_print].round(3).to_string())


# -------------------------------------------------------------------
# 4. Function to plot heatmaps of Volatility per era and shifts
# -------------------------------------------------------------------
def plot_std_era_heatmaps(data_dict, save_image=None, verbose=False):

    # Sort by absolute Pre_minus_COVID std for readability
    era_stds_sorted = data_dict['era_stds'].reindex(
        data_dict['era_stds']['Pre_minus_COVID']
        .abs()
        .sort_values(ascending=False)
        .index
    )
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))
    
    # Left: Raw std dev per era
    sns.heatmap(
        era_stds_sorted[['Pre-COVID', 'COVID', 'Post-COVID']],
        ax=axes[0],
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        linewidths=0.4,
        cbar_kws={'label': 'CLR Std Dev'}
    )
    axes[0].set_title(
        'CLR Std Dev per Era',
        fontsize=13, fontweight='bold'
    )
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    
    # Middle: Pre-COVID -> COVID volatility shift
    sns.heatmap(
        era_stds_sorted[['Pre_minus_COVID']],
        ax=axes[1],
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.4,
        cbar_kws={'label': 'Δ Std Dev'}
    )
    axes[1].set_title(
        'Volatility Shift\nPre-COVID -> COVID',
        fontsize=13, fontweight='bold'
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    
    # Right: COVID -> Post-COVID volatility shift
    sns.heatmap(
        era_stds_sorted[['COVID_minus_Post']],
        ax=axes[2],
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.4,
        cbar_kws={'label': 'Δ Std Dev'}
    )
    axes[2].set_title(
        'Volatility Shift\nCOVID -> Post-COVID',
        fontsize=13, fontweight='bold'
    )
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    
    plt.suptitle(
        'CLR Volatility per Era & Shifts - Chicago Crime 2001–2025\n'
        'Sorted by |Pre_minus_COVID| Std Dev',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_image:
        plt.savefig(f"{save_image}clr_volatility_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    if verbose:
        # Print output
        print("\nSorted by |Pre_minus_COVID| Std Dev:")
        print(era_stds_sorted[[
            'Pre-COVID',
            'COVID',
            'Post-COVID',
            'Pre_minus_COVID',
            'COVID_minus_Post'
        ]].round(3).to_string())


# -------------------------------------------------------------------
# 5. Function to run stationarity tests on each crime type's time series
# -------------------------------------------------------------------
def _run_single_stationarity_test(col_name, series, is_sparse, zero_rate, index):
    """Internal: Runs ADF, KPSS, and ZA tests for a single series."""
    # ADF test
    adf_stat, adf_p, _, _, _, _ = adfuller(series, autolag='AIC')

    # KPSS test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_stat, kpss_p, _, _ = kpss(series, regression='c', nlags='auto')

    # Zivot-Andrews test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Regression 'c' for constant; 'ct' if you expect a trend break
        za_stat, za_p, _, _, za_breakpoint = zivot_andrews(series, trim=0.15, regression='c')
        
        # Convert index to date
        za_date = index[int(za_breakpoint)].strftime('%Y-%m') if za_breakpoint is not None else 'N/A'

    return {
        'crime': col_name,
        'is_sparse': is_sparse,
        'zero_rate': zero_rate,
        'adf_p': adf_p,
        'kpss_p': kpss_p,
        'za_stat': za_stat,
        'za_p': za_p,
        'za_date': za_date
    }


def _evaluate_conclusion(row):
    """Internal: Logic to harmonize ADF, KPSS, and ZA results."""
    if row['is_sparse']:
        return 'Unreliable ⚠️', 'Unreliable ⚠️', 'Excluded - sparse'

    # Thresholds
    α = 0.05
    adf_rej = row['adf_p_adj'] < α
    kpss_rej = row['kpss_p'] < α
    za_rej = row['za_p_adj'] < α

    # ADF + KPSS Logic
    if not adf_rej and kpss_rej:
        adf_kpss = 'Non-Stationary'
    elif adf_rej and not kpss_rej:
        adf_kpss = 'Stationary'
    elif adf_rej and kpss_rej:
        adf_kpss = 'Trend-Stationary'
    else:
        adf_kpss = 'Inconclusive'

    # Zivot-Andrews Logic
    za_conclusion = 'Stationary (ZA)' if za_rej else 'Non-Stationary (ZA)'

    # Final Agreement
    if adf_kpss == 'Non-Stationary' and za_conclusion == 'Non-Stationary (ZA)':
        agreement = 'Agree - Non-Stationary'
    elif adf_kpss in ['Stationary', 'Trend-Stationary'] and za_conclusion == 'Stationary (ZA)':
        agreement = 'Agree - Stationary'
    else:
        agreement = 'Disagree'

    return adf_kpss, za_conclusion, agreement


def _print_stationarity_report(df, sparse_cats, zero_rates):
    """Internal: Handles the standardized console output for stationarity."""
    n_total = len(df)
    n_sparse = len(sparse_cats)
    n_dense = n_total - n_sparse

    print("=" * 70)
    print(f"STATIONARITY ANALYSIS: {n_total:>5} Categories {'|':>5} {n_sparse:>5} Sparse Excluded")
    print("=" * 70)
    
    # New summary block as requested
    print(f"Total sparse : {n_sparse} categories")
    print(f"Total dense  : {n_dense} categories")
    print(f"\nSparse category results reported but excluded")
    print(f"from formal stationarity conclusions.")
    print("-" * 70)

    # List specific sparse categories
    if n_sparse > 0:
        for cat in sparse_cats:
            print(f"  {cat:<50}: {zero_rates[cat]:.1%} zeros  ⚠️")
    print("-" * 70)
    # Conclusion summaries for high-quality (dense) data
    dense_df = df[~df['is_sparse']]
    
    print("\nCONCLUSION COUNTS - ADF + KPSS (dense only):")
    print(dense_df['adf_kpss'].value_counts().to_string())

    print("\nCONCLUSION COUNTS - Zivot - Andrews (dense only):")
    print(dense_df['za_conclusion'].value_counts().to_string())

    print("\n" + "=" * 70)
    print("STEP 6a-iv: AGREEMENT - ADF + KPSS vs ZIVOT - ANDREWS (dense only)")
    print("=" * 70)
    print(dense_df['agreement'].value_counts().to_string())

    
def run_stationarity_analysis(clr_df, filled_df, sparse_threshold=0.05, verbose=True):
    """Public: Orchestrates the stationarity testing suite."""
    
    # Sparse Identification
    zero_rates = filled_df.groupby('fbi_code_desc', observed=True)['crime_count'].apply(lambda x: (x == 0).mean())
    sparse_cats = zero_rates[zero_rates > sparse_threshold].index.tolist()

    # Run Tests
    results = []
    for col in clr_df.columns:
        results.append(_run_single_stationarity_test(
            col, clr_df[col].values, col in sparse_cats, zero_rates.get(col, 0.0), clr_df.index
        ))
    
    stat_df = pd.DataFrame(results).set_index('crime')

    # 3. BH-FDR Correction (Dense only)
    dense_mask = ~stat_df['is_sparse']
    for p_col in ['adf_p', 'za_p']:
        adj_col = p_col + '_adj'
        stat_df[adj_col] = np.nan
        _, stat_df.loc[dense_mask, adj_col], _, _ = multipletests(stat_df.loc[dense_mask, p_col], method='fdr_bh')

    # Consolidate Conclusions
    concl_cols = ['adf_kpss', 'za_conclusion', 'agreement']
    stat_df[concl_cols] = stat_df.apply(lambda r: pd.Series(_evaluate_conclusion(r)), axis=1)

    if verbose:
        _print_stationarity_report(stat_df, sparse_cats, zero_rates)

    return stat_df


# -------------------------------------------------------------------
# 6. Run Bootstrap Comparison of Eras
# -------------------------------------------------------------------
def _calculate_hedges_g(x1, x2):
    """Internal: Hedges' g effect size with small sample correction."""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    
    # Calculate pooled standard deviation
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    
    if pooled_sd == 0:
        return 0
    
    d = (np.mean(x2) - np.mean(x1)) / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction


def _block_bootstrap_logic(x1, x2, block_size, n_bootstrap, seed):
    """Internal: Vectorized block bootstrap to preserve temporal autocorrelation."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(x1), len(x2)
    observed_diff = np.mean(x2) - np.mean(x1)
    
    combined = np.concatenate([x1, x2])
    n_total = len(combined)
    offsets = np.arange(block_size)
    
    def get_boot_means(n_target, blocks_needed):
        # Generate random start indices for all bootstrap iterations at once
        starts = rng.integers(0, n_total - block_size + 1, size=(n_bootstrap, blocks_needed))
        # Efficiently extract blocks using broadcasting: (n_bootstrap, blocks, block_size)
        samples = combined[starts[:, :, np.newaxis] + offsets]
        # Flatten blocks and truncate to original sample size, then compute mean per iteration
        return samples.reshape(n_bootstrap, -1)[:, :n_target].mean(axis=1)

    # Calculate blocks needed for each group
    blocks1, blocks2 = int(np.ceil(n1 / block_size)), int(np.ceil(n2 / block_size))
    
    boot_diffs = get_boot_means(n2, blocks2) - get_boot_means(n1, blocks1)
    p_val = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))
    
    return observed_diff, p_val


def _print_bootstrap_report(df, title, n1_info, n2_info, block_size, n_bootstrap):
    """Internal: Standardized console output formatter."""
    print(f"\n{'='*85}")
    print(f"BLOCK BOOTSTRAP REPORT: {title}")
    print(f"Settings: Block Size={block_size} | Iterations={n_bootstrap:,} | Effect=Hedges' g | Correction=BH-FDR")
    print(f"Samples:  {n1_info[0]} (n={n1_info[1]}) vs {n2_info[0]} (n={n2_info[1]})")
    print(f"{'='*85}")
    
    cols = ['delta_mean', 'hedges_g', 'p_bootstrap', 'p_adj', 'mean_sig']
    print(df[cols].round(4).to_string())
    
    sig_count = df['mean_sig'].sum()
    print(f"\nSummary: {sig_count} of {len(df)} categories showed significant shifts (α=0.05).")


def run_era_comparison(data, era1_df, era2_df, label1="Era 1", label2="Era 2", 
                       verbose=False, block_size=12, n_bootstrap=10_000, seed=1776):
    """
    Runs the bootstrap test across all categories.
    Set verbose=True to trigger the formatted print output.
    """
    results = []
    
    for col in data.columns:
        x1, x2 = era1_df[col].values, era2_df[col].values
        
        diff, p = _block_bootstrap_logic(x1, x2, block_size, n_bootstrap, seed=seed)
        g = _calculate_hedges_g(x1, x2)
        
        results.append({
            'crime': col,
            'delta_mean': diff,
            'hedges_g': g,
            'p_bootstrap': p
        })
    
    # Build results DataFrame
    res_df = pd.DataFrame(results).set_index('crime')
    
    # Multiple testing correction
    _, res_df['p_adj'], _, _ = multipletests(res_df['p_bootstrap'], method='fdr_bh')
    res_df['mean_sig'] = res_df['p_adj'] < 0.05
    
    # Sort by absolute effect size magnitude
    res_df = res_df.sort_values(by='hedges_g', key=abs, ascending=False)
    
    # Conditional Print Trigger
    if verbose:
        title = f"{label1} vs {label2}"
        _print_bootstrap_report(res_df, title, (label1, len(era1_df)), 
                                (label2, len(era2_df)), block_size, n_bootstrap)
        
    return res_df