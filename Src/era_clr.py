
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.multitest import multipletests
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------------------
# 1. Function to slice CLR matrix into eras
# --------------------------------------------------------------------
def slice_clr_into_eras(clr_dict, era_boundaries, eps):
    """
    Slice a CLR matrix into Pre‑COVID, COVID, and Post‑COVID eras using
    fixed administrative boundaries and verified shapes + date alignment.
    """
    # Extract the CLR DataFrame for the specified epsilon
    clr_df = clr_dict[eps]

    # Administrative boundaries based on crime data availability and COVID timeline:
    for k, v in era_boundaries.items():
        if k == 'Pre-COVID':
            loc = clr_df.index.get_loc(v)
            PRE_END = loc.stop
        elif k == 'COVID':
            loc = clr_df.index.get_loc(v)
            COVID_END = loc.stop
    
    # Slice eras based on verified indices and expected date ranges
    pre_covid  = clr_df.iloc[:PRE_END]
    covid      = clr_df.iloc[PRE_END:COVID_END]
    post_covid = clr_df.iloc[COVID_END:]

    # Verification prints to confirm shapes and date alignment
    print("CLR matrix shape:", clr_df.shape)
    print("First 3 index values:", clr_df.index[:3].tolist())
    print("Last 3 index values:", clr_df.index[-3:].tolist())

    print(f"\n{'Era Shape Verification:':<15}")
    print(f"{'Pre-COVID':<12} : {str(pre_covid.shape):<10}")
    print(f"{'COVID':<12} : {str(covid.shape):<10}")
    print(f"{'Post-COVID':<12} : {str(post_covid.shape):<10}")

    print(f"\n{'Era boundary verification:':<20}")
    print(f"{'Pre-COVID ends':<18}: {pre_covid.index[-1].strftime('%Y-%m')}")
    print(f"{'COVID starts':<18}: {covid.index[0].strftime('%Y-%m')}")
    print(f"{'COVID ends':<18}: {covid.index[-1].strftime('%Y-%m')}")
    print(f"{'Post-COVID starts':<18}: {post_covid.index[0].strftime('%Y-%m')}")
    print(f"{'Post-COVID ends':<18}: {post_covid.index[-1].strftime('%Y-%m')}")

    return {'pre_covid':  pre_covid, 'covid': covid, 'post_covid': post_covid}


# -------------------------------------------------------------------
# 2. Function to compute distributional parameters for each era
# --------------------------------------------------------------------
def compute_era_distribution_parameters(eras_dict,
                                        top_display=True,
                                        n_rows=10):
    """
    Compute per-era distributional parameters (mean, std, covariance)
    and absolute differences between eras.

    Covariance estimation                   
    ─────────────────────  
    Raw sample covariance (pd.cov) is stored for reference. 
    Ledoit-Wolf shrinkage is applied for all downstream matrix-based
    procedures (PCA, Hotelling T², KL divergence). Shrinkage is
    especially important for COVID (T=34) and Post-COVID (T=40) eras,
    where T/K ratios of 1.31 and 1.54 produce ill-conditioned 
    sample covariance matrices. Ledoit-Wolf regularization improves stability
    and inference validity.
    """
    eras = {
        'Pre-COVID' : eras_dict['pre_covid'],
        'COVID'     : eras_dict['covid'],
        'Post-COVID': eras_dict['post_covid']
    }

    # Mean vector per era (central tendency of crime distribution in CLR space)
    era_means = pd.DataFrame({name: df.mean(axis=0) for name, df in eras.items()})

    # Std deviation per era (variability around mean)
    era_stds  = pd.DataFrame({name: df.std(axis=0)  for name, df in eras.items()})

    # Covariance matrices - raw + regularized for stability and inference
    era_covs     = {} 
    era_covs_lw  = {}
    lw_shrinkage = {}
    cond_numbers = {}

    for name, df in eras.items():
        # Raw sample covariance (matrix of interrelationships between 
        # crime types in CLR space)
        raw_cov              = df.cov()
        era_covs[name]       = raw_cov

        # Ledoit-Wolf regularized covariance estimation 
        # (improves stability for PCA, Hotelling T², KL divergence)
        lw                   = LedoitWolf(assume_centered=False).fit(df)
        era_covs_lw[name]    = pd.DataFrame(
            lw.covariance_,
            index=df.columns,
            columns=df.columns
        )
        lw_shrinkage[name]   = lw.shrinkage_

        # Condition numbers (lower = more stable)
        cond_numbers[name]   = {
            'raw' : np.linalg.cond(raw_cov.values),
            'lw'  : np.linalg.cond(lw.covariance_)
        }

    # Differences between eras
    era_means["Pre_minus_COVID"]  = era_means["Pre-COVID"] - era_means["COVID"]
    era_means["COVID_minus_Post"] = era_means["COVID"]     - era_means["Post-COVID"]
    era_means["Pre_minus_Post"]   = era_means["Pre-COVID"] - era_means["Post-COVID"]

    era_stds["Pre_minus_COVID"]   = era_stds["Pre-COVID"]  - era_stds["COVID"]
    era_stds["COVID_minus_Post"]  = era_stds["COVID"]      - era_stds["Post-COVID"]
    era_stds["Pre_minus_Post"]    = era_stds["Pre-COVID"]  - era_stds["Post-COVID"]

    # --- Print summary ---
    print("=" * 125)
    print("MEAN CLR VECTOR PER ERA (rows=crime type, cols=era & comparisons)")
    print("=" * 125)
    print(era_means.round(4).to_string())
    print()

    print("=" * 125)
    print("STD DEV PER ERA")
    print("=" * 125)
    print(era_stds.round(4).to_string())
    print()

    print("=" * 70)
    print(f"{'COVARIANCE MATRIX SHAPES':^70}")
    print("=" * 70)
    for name, cov in era_covs.items():
        print(f" {name:<12}: {str(cov.shape):<10} -> expected (26, 26)")

    # Condition number + shrinkage report
    print()
    print("=" * 70)
    print(f"{'COVARIANCE STABILITY REPORT':^70}")
    print("=" * 70)
    # Updated headers with slightly wider spacing for scientific notation
    print(f"  {'Era':<12} {'Cond# Raw':>12} {'Cond# LW':>12} "
        f"{'Shrinkage α':>12}  {'Stability':>10}") 
    print(f"  {'-'*68}")

    for name in eras:
        cn = cond_numbers[name]
        α  = lw_shrinkage[name]
        
        # Usability based on digits of precision lost
        digits_lost = np.log10(cn['lw']) if cn['lw'] > 0 else 0
        usable      = digits_lost < 8
        flag        = '✅' if usable else '⚠️' 
        
        # Use .1e for Raw to handle the massive scale without breaking alignment
        # Use .1f for LW since shrinkage usually brings it down to manageable levels
        print(
            f"  {name:<12} {cn['raw']:>12.1e} {cn['lw']:>12.1f} "
            f"{α:>12.4f}  {flag:^10}" 
        )
    print("=" * 70)
    print()
    print("  Cond# Raw  : sample covariance - reference only, CLR rank deficiency expected")
    print("  Cond# LW   : Ledoit-Wolf regularized - used for all matrix-based inference")
    print("  Shrinkage α: 0.0 = no shrinkage applied, 1.0 = full shrinkage applied")
    print("  Stability  : ✅ if log₁₀(Cond# LW) < 8  (>7 digits of precision retained)")

    if top_display:
        # Display top mean shifts for interpretability (sorted by absolute magnitude)
        print("\n" + "=" * 70)
        print(f"TOP {n_rows} MEAN SHIFTS")
        print("=" * 70)
        print("Note: Differences between eras, sorted by Absolute magnitude.")
        print("Pre-COVID -> COVID & COVID -> Post-COVID & Pre-COVID -> Post-COVID ")
        print(f"are ranked separately.")
        print("This highlights which crime types had the largest distributional shifts.")
        print(f"{'-' * 70}\n")

        # Rank mean shifts Pre -> COVID
        print(f"TOP {n_rows} MEAN SHIFTS - Pre-COVID -> COVID")
        print("=" * 55)
        top_pre_covid = (
            era_means['Pre_minus_COVID']                           # ← FIXED indent
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
            era_means['COVID_minus_Post']                          # ← FIXED indent
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

        print()

        # Rank mean shifts Pre-COVID -> Post-COVID
        print(f"TOP {n_rows} MEAN SHIFTS - Pre-COVID -> Post-COVID")
        print("=" * 55)
        top_pre_post = (
            era_means['Pre_minus_Post']                            # ← FIXED indent
            .reindex(
                era_means['Pre_minus_Post']
                .abs()
                .sort_values(ascending=False)
                .head(n_rows)
                .index
            )
            .round(4)
        )
        print(top_pre_post.to_string())

        print()

        # Volatility shifts
        print("\n" + "=" * 70)
        print(f"TOP {n_rows} VOLATILITY SHIFTS")
        print("=" * 70)
        print("Note: Differences in variance between eras, sorted by absolute magnitude.")
        print("Pre-COVID -> COVID & COVID -> Post-COVID & Pre-COVID -> Post-COVID are")
        print("ranked separately.")
        print()
        print("** This highlights which crime types had the largest volatility shifts.")
        print(f"{'-' * 70}\n")

        print()

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

        print()

        print(f"TOP {n_rows} VOLATILITY SHIFTS - Pre-COVID -> Post-COVID")
        print("=" * 55)
        top_var_pre_post = (
            era_stds['Pre_minus_Post']
            .reindex(
                era_stds['Pre_minus_Post']
                .abs()
                .sort_values(ascending=False)
                .head(n_rows)
                .index
            )
            .round(4)
        )
        print(top_var_pre_post.to_string())

    return {
        'era_means'   : era_means,
        'era_stds'    : era_stds,
        'era_covs'    : era_covs,        # raw - reference only
        'era_covs_lw' : era_covs_lw,    
        'lw_shrinkage': lw_shrinkage,   
        'cond_numbers': cond_numbers,
    }


# -------------------------------------------------------------------
# 3. Function to plot heatmaps of mean CLR per era and shifts
# -------------------------------------------------------------------
def plot_mean_era_heatmaps(data_dict, save_image=None, verbose=False):
    # Define the columns and titles to iterate through
    plot_configs = [
        (['Pre-COVID', 'COVID', 'Post-COVID'], 'Mean CLR per Era', 'CLR Mean'),
        (['Pre_minus_COVID'], 'Mean CLR Shift\nPre-COVID -> COVID', 'Δ CLR'),
        (['COVID_minus_Post'], 'Mean CLR Shift\nCOVID -> Post-COVID', 'Δ CLR'),
        (['Pre_minus_Post'], 'Mean CLR Shift\nPre-COVID -> Post-COVID', 'Δ CLR')
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
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    
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
        cols_to_print = ['Pre-COVID', 'COVID', 'Post-COVID', 'Pre_minus_COVID', 'COVID_minus_Post', 'Pre_minus_Post']
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
    fig, axes = plt.subplots(1, 4, figsize=(28, 10))
    
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
    
    # Middle-left: Pre-COVID -> COVID volatility shift
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
    
    # Middle-right: COVID -> Post-COVID volatility shift
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
    
    # Right: Pre-COVID -> Post-COVID volatility shift
    sns.heatmap(
        era_stds_sorted[['Pre_minus_Post']],
        ax=axes[3],
        cmap='RdBu_r',
        center=0,
        annot=True,
        fmt='.2f',
        linewidths=0.4,
        cbar_kws={'label': 'Δ Std Dev'}
    )
    axes[3].set_title(
        'Volatility Shift\nPre-COVID -> Post-COVID',
        fontsize=13, fontweight='bold'
    )
    axes[3].set_xlabel('')
    axes[3].set_ylabel('')
    
    plt.suptitle(
        'CLR Volatility per Era & Shifts - Chicago Crime 2001–2025\n'
        'Sorted by |Pre_minus_COVID| Std Dev',
        fontsize=14, fontweight='bold', y=1.02
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
            'COVID_minus_Post',
            'Pre_minus_Post'
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

    # Zivot-Andrews test - run both 'c' and 'ct' to check for trend sensitivity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # regression='c'  - break in intercept only (primary for crime data)
        za_stat_c, za_p_c, _, _, za_bp_c = zivot_andrews(
            series, trim=0.15, regression='c'
        )
        za_date_c = (
            index[int(za_bp_c)].strftime('%Y-%m')
            if za_bp_c is not None else 'N/A'
        )

        # regression='ct' - break in intercept + trend (primary)
        za_stat_ct, za_p_ct, _, _, za_bp_ct = zivot_andrews(
            series, trim=0.15, regression='ct'
        )
        za_date_ct = (
            index[int(za_bp_ct)].strftime('%Y-%m')
            if za_bp_ct is not None else 'N/A'
        )

    # Compare break dates - flag if they diverge by more 
    # than 12 months (indicating potential trend sensitivity) 
    if za_bp_c is not None and za_bp_ct is not None: 
        bp_diff        = abs(int(za_bp_c) - int(za_bp_ct))
        trend_sensitive = bp_diff > 12
    else:
        bp_diff         = None # Cannot compute difference if one is None
        trend_sensitive = False 

    return {
        'crime'           : col_name,
        'is_sparse'       : is_sparse,
        'zero_rate'       : zero_rate,
        'adf_p'           : adf_p,
        'kpss_p'          : kpss_p,
        'za_stat'         : za_stat_ct,
        'za_p'            : za_p_ct,
        'za_date'         : za_date_ct,
        'za_stat_c'       : za_stat_c,
        'za_p_c'          : za_p_c,
        'za_date_c'       : za_date_c,
        'trend_sensitive' : trend_sensitive,
        'bp_diff_months'  : bp_diff,
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


def _print_stationarity_report(df, sparse_cats, zero_rates, era_boundaries):
    """Internal: Handles the standardized console output for stationarity."""
    n_total  = len(df)
    n_sparse = len(sparse_cats)
    n_dense  = n_total - n_sparse

    print("=" * 70)
    print(f"STATIONARITY ANALYSIS: {n_total:>5} Categories {'|':>5} {n_sparse:>5} Sparse Excluded")
    print("=" * 70)

    print(f"Total sparse : {n_sparse} categories")
    print(f"Total dense  : {n_dense} categories")
    print(f"\nSparse category results reported but excluded")
    print(f"from formal stationarity conclusions.")
    print("-" * 70)

    if n_sparse > 0:
        for cat in sparse_cats:
            print(f"  {cat:<50}: {zero_rates[cat]:<5.1%} zeros:  ⚠️")
    print("-" * 70)

    dense_df = df[~df['is_sparse']]

    print("\nCONCLUSION COUNTS - ADF + KPSS (dense only):")
    print(dense_df['adf_kpss'].value_counts().to_string())

    print("\nCONCLUSION COUNTS - Zivot-Andrews 'ct' primary (dense only):")
    print(dense_df['za_conclusion'].value_counts().to_string())

    print("\n" + "=" * 70)
    print("STEP 6a-iv: AGREEMENT - ADF + KPSS vs ZIVOT - ANDREWS (dense only)")
    print("=" * 70)
    print(dense_df['agreement'].value_counts().to_string())

    # Trend-sensitive categories (where ZA break date differs by >12 months between 'c' and 'ct')
    trend_df = dense_df[dense_df['trend_sensitive']]
    print("\n" + "=" * 70)
    print(f"TREND-SENSITIVE CATEGORIES (break date divergence > 12 months)")
    print(f"Primary model: 'ct'  |  Secondary model: 'c'")
    print("=" * 70)
    if trend_df.empty:
        print("  None detected - 'c' and 'ct' agree within 12 months for all.")
    else:
        report_cols = ['za_date', 'za_date_c', 'bp_diff_months']
        print(
            trend_df[report_cols]
            .rename(columns={
                'za_date'       : "Break 'ct'",
                'za_date_c'     : "Break 'c'",
                'bp_diff_months': 'Diff (months)'
            })
            .to_string()
        )

    # Disagreement detail
    dis_df = dense_df[dense_df['agreement'] == 'Disagree']
    if not dis_df.empty:
        print()
        print("=" * 70)
        print("DISAGREEMENT DETAIL - ADF + KPSS vs ZIVOT - ANDREWS")
        print("=" * 70)

        for crime, row in dis_df.iterrows():
            print(f"\n  {crime}")
            print(f"    {'ADF + KPSS':<12}: {row['adf_kpss']}")
            
            # ZA ('ct') Row
            za_ct_txt = f"{row['za_conclusion']:<15} break = {row['za_date']:<}"
            print(f"    {'ZA (ct)':<12}: {za_ct_txt} p_adj = {row['za_p_adj']:>10.6f}")
            
            # ZA ('c') Row
            za_c_txt  = f"{'':<15} break = {row['za_date_c']:<8}" # Empty space to align with 'Stationary' above
            print(f"    {'ZA (c)':<12}: {za_c_txt} p_c  = {row['za_p_c']:>10.6f}")
            
            # Note Row
            print(f"    {'Note':<12}: zero_rate = {row['zero_rate']:.4f} "
                f"- quasi-sparse, interpret with caution")
    

    # Break date alignment summary
    print()
    print("=" * 70)
    print("BREAK DATE ALIGNMENT - PRIMARY 'ct' MODEL")
    print("=" * 70)

    # ── Unpack from era_boundaries and convert to Timestamps for comparison
    covid_start = pd.Timestamp(era_boundaries['Pre-COVID']) + pd.DateOffset(months=1) 
    covid_end   = pd.Timestamp(era_boundaries['COVID'])

    # Define dates once for cleaner f-strings
    start_str = covid_start.strftime('%Y-%m')
    end_str   = covid_end.strftime('%Y-%m')

    print(f"  Era boundaries from data:")
    print(f"    COVID start : {start_str}")
    print(f"    COVID end   : {end_str}")
    print()

    # Classify break dates into Pre-COVID, COVID-aligned, and Post-COVID based on 'ct' break date
    aligned, pre, post = [], [], []
    for crime, row in dense_df.iterrows():
        date = pd.Timestamp(row['za_date'] + '-01')
        if covid_start <= date <= covid_end:
            aligned.append((crime, row['za_date']))
        elif date < covid_start:
            pre.append((crime, row['za_date']))
        else:
            post.append((crime, row['za_date']))

    # Using a width of 45 for the label + date range part
    print(f"  {'COVID-aligned (' + start_str + ' - ' + end_str + ')':<35} : "
        f"{len(aligned):>3} of {len(dense_df)}")

    print(f"  {'Pre-COVID (before ' + start_str + ')':<35} : "
        f"{len(pre):>3} of {len(dense_df)}")

    print(f"  {'Post-COVID (after ' + end_str + ')':<35} : "
        f"{len(post):>3} of {len(dense_df)}")
    print()




    # print(f"  COVID-aligned "
    #       f"({covid_start.strftime('%Y-%m')} - "
    #       f"{covid_end.strftime('%Y-%m')}) : "
    #       f"{len(aligned):>2} of {len(dense_df)}")
    # print(f"  Pre-COVID  (before {covid_start.strftime('%Y-%m')})    : "
    #       f"{len(pre):>2} of {len(dense_df)}")
    # print(f"  Post-COVID (after  {covid_end.strftime('%Y-%m')})      : "
    #       f"{len(post):>2} of {len(dense_df)}")
    # print()
    # if pre:
    #     print("  Pre-COVID breaks (secular trends):")
    #     for c, d in sorted(pre, key=lambda x: x[1]):
    #         print(f"    {c:<45} {d}")

    
def run_stationarity_analysis(clr_df, filled_df, era_boundaries = None, 
                              sparse_threshold=0.05, verbose=True):
    """Orchestrates the stationarity testing suite."""

    # Validation check for era_boundaries
    if era_boundaries is None:
        raise ValueError(
            "era_boundaries is required. Pass a dict with keys: "
            "'Pre-COVID', 'COVID', 'Post-COVID'"
        )  
    
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
    dense_bool_mask = ~stat_df['is_sparse']
    for p_col in ['adf_p', 'za_p']:
        adj_col = p_col + '_adj'
        stat_df[adj_col] = np.nan
        _, stat_df.loc[dense_bool_mask, adj_col], _, _ = multipletests(stat_df.loc[dense_bool_mask, p_col], method='fdr_bh')

    # Consolidate Conclusions
    concl_cols = ['adf_kpss', 'za_conclusion', 'agreement']
    stat_df[concl_cols] = stat_df.apply(lambda r: pd.Series(_evaluate_conclusion(r)), axis=1)

    if verbose:
        _print_stationarity_report(stat_df, sparse_cats, zero_rates, era_boundaries)

    return stat_df


# -------------------------------------------------------------------
# 6. Run Bootstrap Comparison of Eras
# -------------------------------------------------------------------
def _calculate_hedges_g(x1, x2):
    """
    Hedges' g effect size with small sample correction.

    Sign convention
    ──────────────
    g = (mean(x2) - mean(x1)) / pooled_sd
    Positive g -> x2 (later era) is higher than x1 (earlier era)
    Negative g -> x2 (later era) is lower  than x1 (earlier era)

    Note: this is the OPPOSITE sign to the CLR difference tables,
    where Pre_minus_COVID = Pre - COVID.
    Use hedges_g_clr in the results DataFrame for a consistent sign.
    """
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
    print(f"\n{'='*105}")
    print(f"BLOCK BOOTSTRAP REPORT: {title}")
    print(f"Settings: Block Size={block_size} | Iterations={n_bootstrap:,} | Effect=Hedges' g | Correction=BH-FDR")
    print(f"Samples:  {n1_info[0]} (n={n1_info[1]}) vs {n2_info[0]} (n={n2_info[1]})")
    print(f"{'='*105}")

    # Sign convention note
    print(
        f"Sign note: hedges_g = mean(era2) - mean(era1). "
        f"hedges_g_clr = -hedges_g, matching CLR table convention."
    ) 
    
    # Split dense vs sparse for display
    dense_df  = df[~df['is_sparse']]
    sparse_df = df[ df['is_sparse']]

    cols = ['delta_mean', 'hedges_g', 'hedges_g_clr', 'p_bootstrap', 'p_adj', 'mean_sig'] 

    print("\nCONFIRMED CATEGORIES (dense ✅)")
    print(dense_df[cols].round(4).to_string())

    if not sparse_df.empty:
        print("\nFLAGGED CATEGORIES (sparse ⚠️ - excluded from FDR correction)")
        sparse_cols = ['delta_mean', 'hedges_g', 'hedges_g_clr', 'p_bootstrap']  
        print(sparse_df[sparse_cols].round(4).to_string())

    sig_count = dense_df['mean_sig'].sum()
    print(
        f"\nSummary: {sig_count} of {len(dense_df)} confirmed categories "
        f"showed significant shifts (α=0.05). "
        f"{len(sparse_df)} sparse categories excluded from inference."
    )

# ============================================================
# Bootstrap comparison function to run across all categories
# ============================================================
def run_era_comparison(data, era1_df, era2_df, label1="Era 1", label2="Era 2",
                       verbose=False, block_size=12, n_bootstrap=10_000,
                       seed=1776, sparse_cats=None):
    """
    Runs the bootstrap test across all categories.
    Set verbose=True to trigger the formatted print output.

    Notes
    -----
    block_size is auto-adjusted if the shortest era produces fewer
    than 5 effective blocks. The original value is preserved
    in the warning.

    sparse_cats : list of str, optional
        Category names to flag as sparse. These are still computed
        for reference but excluded from BH-FDR correction.
        Their p_adj is set to NaN and mean_sig to False.
    """
    # Block size auto-adjustment
    min_T         = min(len(era1_df), len(era2_df))
    original_size = block_size

    if min_T / block_size < 5:
        block_size = max(3, min_T // 5)
        warnings.warn(
            f"\nBlock bootstrap - block size auto-adjusted:"
            f"\n  Shortest era      : T = {min_T}"
            f"\n  Requested size    : {original_size}"
            f"\n  Effective blocks  : ~{min_T/original_size:.1f}"
            f"  (minimum recommended: 5)"
            f"\n  Adjusted size     : {block_size}"
            f"\n  Effective blocks  : ~{min_T/block_size:.1f}"
            f"\n  Interpret results for this comparison cautiously.",
            UserWarning,
            stacklevel=2
        )

    # Normalise sparse list
    sparse_cats = set(sparse_cats) if sparse_cats else set()

    results = []
    for col in data.columns:
        x1, x2 = era1_df[col].values, era2_df[col].values

        diff, p = _block_bootstrap_logic(x1, x2, block_size, n_bootstrap, seed=seed)
        g       = _calculate_hedges_g(x1, x2)

        results.append({
            'crime'       : col,
            'delta_mean'  : diff,
            'hedges_g'    : g,
            'hedges_g_clr': -g,
            'p_bootstrap' : p,
            'is_sparse'   : col in sparse_cats
        })

    res_df     = pd.DataFrame(results).set_index('crime')
    dense_mask = ~res_df['is_sparse']

    # BH-FDR on dense categories only 
    res_df['p_adj']    = np.nan
    res_df['mean_sig'] = False

    _, res_df.loc[dense_mask, 'p_adj'], _, _ = multipletests(
        res_df.loc[dense_mask, 'p_bootstrap'], method='fdr_bh'
    )
    res_df.loc[dense_mask, 'mean_sig'] = ( 
        res_df.loc[dense_mask, 'p_adj'] < 0.05
    )

    res_df = res_df.sort_values(by='hedges_g', key=abs, ascending=False)

    if verbose:
        title = f"{label1} vs {label2}"
        _print_bootstrap_report(
            res_df, title,
            (label1, len(era1_df)),
            (label2, len(era2_df)),
            block_size, n_bootstrap
        )

    return res_df


# ============================================================
# LAYER 1 - DATA PREPARATION
# ============================================================
def _prepare_break_data(clr_df, category, break_date, window):
    """
    Extract CLR series and compute break statistics.
    Uses vectorized bool_masks and pre-computed stats dict.
    """
    series   = pd.Series(
        clr_df[category].values,
        index=pd.to_datetime(clr_df.index),
        name=category
    )
    break_dt  = pd.Timestamp(break_date)

    # Vectorized bool_masks
    pre_bool_mask  = series.index <  break_dt
    post_bool_mask = ~pre_bool_mask
    pre_vals  = series[pre_bool_mask]
    post_vals = series[post_bool_mask]

    stats = {
        'pre_mean'  : pre_vals.mean(),
        'post_mean' : post_vals.mean(),
        'shift'     : post_vals.mean() - pre_vals.mean(),
        'rolling'   : series.rolling(
                          window=window, center=True
                      ).mean(),
        'pre_std'   : pre_vals.std(),
        'post_std'  : post_vals.std(),
        'pre_n'     : len(pre_vals),
        'post_n'    : len(post_vals),
    }

    return series, break_dt, pre_vals, post_vals, stats


# ============================================================
# LAYER 2 - PLOTTING FUNCTIONS
# ============================================================
def _plot_time_series(ax, series, break_dt, stats,
                      era_config, window, category,
                      za_stat, za_p_adj, legend_loc='lower left'):
    """
    Top panel: CLR time series with era shading,
    rolling mean, segment means, break line and annotation.
    """
    # Era shading
    for label, (start, end, color) in era_config.items():
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end),
            alpha=0.07, color=color, label=label
        )

    # Raw CLR series
    ax.plot(
        series.index, series.values,
        color='gray', alpha=0.3,
        lw=0.8, label='CLR Raw'
    )

    # Rolling mean
    ax.plot(
        stats['rolling'].index,
        stats['rolling'].values,
        color='#2c7bb6', lw=2.5,
        label=f'{window}-month rolling mean'
    )

    # Vertical break line + dynamic label
    ax.axvline(
        break_dt, color='#d62728',
        lw=2.0, ls='--', alpha=0.8
    )
    ax.text(
        break_dt, ax.get_ylim()[1],
        f'  ZA Break: {break_dt.strftime("%Y-%m")}',
        color='#d62728', fontweight='bold',
        va='top', fontsize=10
    )

    # Segment means
    ax.hlines(
        y=[stats['pre_mean'], stats['post_mean']],
        xmin=[series.index[0], break_dt],
        xmax=[break_dt,        series.index[-1]],
        colors=['#1a9641', '#d7191c'],
        lw=3.0,
        label='Segment means'
    )

    # Mean shift annotation - curved arrow + white bbox
    mid_y = (stats['pre_mean'] + stats['post_mean']) / 2
    ax.annotate(
        f"Mean Shift: {stats['shift']:+.3f}",
        xy=(break_dt, mid_y),
        xytext=(pd.Timestamp('2015-01-01'), mid_y + 0.4),
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=.2',
            color='#d62728', lw=1.5
        ),
        fontsize=12, fontweight='bold',
        color='#d62728',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax.set_title(
        f'CLR Time Series - {category}',
        fontsize=14, fontweight='bold'
    )
    ax.set_ylabel('CLR Value', fontsize=11)
    ax.legend(loc=legend_loc, ncol=2, frameon=True)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)



# -----------------------------------------------------------
# 7. Structural Break - Distribution Plotting Function
# -----------------------------------------------------------
def _plot_distribution(ax, pre_vals, post_vals,
                       stats, break_date, legend_loc='lower left'):
    """
    Bottom panel: Density distributions before and after
    the structural break with KDE and mean markers.
    """
    # Histogram + KDE
    sns.histplot(
        pre_vals, bins=30, kde=True,
        color='#2c7bb6', alpha=0.4,
        label=f'Pre-break  (n={stats["pre_n"]})',
        stat='density', ax=ax
    )
    sns.histplot(
        post_vals, bins=30, kde=True,
        color='#d62728', alpha=0.4,
        label=f'Post-break (n={stats["post_n"]})',
        stat='density', ax=ax
    )

    # Mean markers
    ax.axvline(
        stats['pre_mean'],  color='#2c7bb6',
        ls='--', lw=2.0,
        label=f'Pre mean:  {stats["pre_mean"]:.3f}'
    )
    ax.axvline(
        stats['post_mean'], color='#d62728',
        ls='--', lw=2.0,
        label=f'Post mean: {stats["post_mean"]:.3f}'
    )

    ax.set_title(
        f'CLR Density - Before vs After Break '
        f'({pd.Timestamp(break_date).strftime("%Y-%m")})',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlabel('CLR Value', fontsize=11)
    ax.set_ylabel('Density',   fontsize=11)
    ax.legend(loc=legend_loc, fontsize=9)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)


# ============================================================
# LAYER 3 - VERBOSE OUTPUT
# ============================================================
def _print_break_summary(category, break_date,
                          pre_vals, post_vals,
                          stats, za_stat, za_p_adj,
                          era_boundaries=None):
    """
    Print segment statistics using pd.describe()
    for clean DataFrame alignment.
    """
    summary = pd.DataFrame({
        'Pre-Break' : pre_vals.describe(),
        'Post-Break': post_vals.describe()
    })

    # p-value + conclusion logic
    if pd.isna(za_p_adj):
        p_display  = "nan  (sparse - excluded from FDR) ⚠️"
        conclusion = "Unreliable ⚠️ - sparse category"
    elif za_p_adj < 0.05:
        p_display  = f"{za_p_adj:.6f}  (below α=0.05 ✅)"
        conclusion = "Stationary around break"
    else:
        p_display  = f"{za_p_adj:.6f}  (above α=0.05 ❌)"
        conclusion = "Non-Stationary (ZA)"

    print("=" * 65)
    print(f"STRUCTURAL BREAK ANALYSIS - {category}")
    print(f"Break point : {pd.Timestamp(break_date).strftime('%Y-%m')}")
    print(f"ZA statistic: {za_stat:.6f}")
    print(f"za_p_adj    : {p_display}")
    print(f"Conclusion  : {conclusion}") 
    print("=" * 65)

    print("\nSEGMENT STATISTICS:")
    print(
        summary.T[['count','mean','std','min','max']]
        .round(4)
        .to_string()
    )

    print(f"\nMean shift at break : {stats['shift']:+.4f}")
    print(f"Pre-break  std      : {stats['pre_std']:.4f}")
    print(f"Post-break std      : {stats['post_std']:.4f}")

    # Post-break era composition note
    if era_boundaries is not None:
        break_dt    = pd.Timestamp(break_date)
        covid_start = (pd.Timestamp(era_boundaries['Pre-COVID'])
                       + pd.DateOffset(months=1))
        covid_end   = pd.Timestamp(era_boundaries['COVID'])
        post_start  = pd.Timestamp(era_boundaries['Post-COVID'])

        if break_dt < covid_end:
            pre_covid_in_post = len(
                post_vals[post_vals.index < covid_start]
            )
            covid_months = len(
                post_vals[
                    (post_vals.index >= covid_start) &
                    (post_vals.index <= covid_end)
                ]
            )
            post_months = len(
                post_vals[post_vals.index > covid_end]
            )

            era_label = ("two" if pre_covid_in_post == 0
                         else "multiple")
            print(f"\nNote: Post-break segment (n={len(post_vals)}) "
                  f"spans {era_label} administrative eras:")
            if pre_covid_in_post > 0:
                pre_covid_end = covid_start - pd.DateOffset(months=1)
                print(f"  Pre-COVID era  : {pre_covid_in_post:<4} months "
                      f"({break_dt.strftime('%Y-%m')} - "
                      f"{pre_covid_end.strftime('%Y-%m')})")
            print(f"  COVID era      : {covid_months:<4} months "
                  f"({covid_start.strftime('%Y-%m')} - "
                  f"{covid_end.strftime('%Y-%m')})")
            print(f"  Post-COVID era : {post_months:<4} months "
                  f"({post_start.strftime('%Y-%m')} - "
                  f"{post_vals.index[-1].strftime('%Y-%m')})")
            print(f"  ZA break does not align with the administrative "
                  f"era boundary.")



# ============================================================
# LAYER 4 - EXECUTION INTERFACE
# ============================================================
def plot_structural_break(clr_df, category, break_date,
                           era_config=None, window=None,
                           za_stat=None,
                           za_p_adj=None,
                           era_boundaries=None,
                           save_path=None,
                           verbose=True,
                           legend_loc='lower left'):
    """
    Full structural break diagnostic pipeline.

    Parameters:
    -----------
    clr_df         : CLR DataFrame (T=304, K=26)
    category       : crime category column name
    break_date     : ZA-identified break date (YYYY-MM-DD)
    era_config     : dict of era label -> (start, end, color)
    window         : rolling mean window in months
    za_stat        : Zivot-Andrews test statistic
    za_p_adj       : BH-FDR adjusted ZA p-value
    era_boundaries : dict with keys Pre-COVID, COVID, Post-COVID
    save_path      : output file path
    verbose        : print segment statistics
    legend_loc     : legend position, default 'lower left'
    """
    # Step 1 - Prepare data
    series, break_dt, pre_vals, post_vals, stats = \
        _prepare_break_data(clr_df, category, break_date, window)

    # Step 2 - Build figure
    fig, (ax_ts, ax_dist) = plt.subplots(
        2, 1, figsize=(16, 12),
        gridspec_kw={'height_ratios': [1.2, 0.8]}
    )

    # Step 3 - Time series panel
    _plot_time_series(
        ax_ts, series, break_dt, stats,
        era_config, window, category,
        za_stat, za_p_adj, legend_loc
    )

    # Step 4 - Distribution panel
    _plot_distribution(
        ax_dist, pre_vals, post_vals,
        stats, break_date, legend_loc
    )

    # Step 5 - Suptitle
    p_label = (f"Adj p-value: {za_p_adj:.4f}"
               if not pd.isna(za_p_adj)
               else "Adj p-value: NaN ⚠️ sparse")

    plt.suptitle(
        f'Structural Break Diagnostics - {category}\n'
        f'ZA Statistic: {za_stat:.3f}  |  '
        f'{p_label}  |  '
        f'Break: {pd.Timestamp(break_date).strftime("%Y-%m")}',
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(
            f"{save_path}structural_break.png",
            dpi=300, bbox_inches='tight'
        )
    plt.show()

    # Step 6 - Verbose output
    if verbose:
        _print_break_summary(
            category, break_date,
            pre_vals, post_vals,
            stats, za_stat, za_p_adj,
            era_boundaries
        )


    
# ============================================================
# COVARIANCE STRUCTURE ANALYSIS
# ============================================================
# Layer 1 - Data Preparation
# Layer 2 - Box's M + Permutation Test
# Layer 3 - Correlation Heatmaps (per era + differences)
# Layer 4 - Log-Determinant Comparison
# Layer 5 - Orchestrator
# ============================================================

# ============================================================
# LAYER 1 - DATA PREPARATION
# ============================================================
def _prepare_cov_matrices(eras_stat, clr_era):
    """
    Extract and validate all covariance matrices and CLR arrays.

    Parameters
    ----------
    eras_stat : output of compute_era_distribution_parameters
                must contain 'era_covs' and 'era_covs_lw'
    clr_era   : output of slice_clr_into_eras
                must contain 'pre_covid', 'covid', 'post_covid'

    Returns
    -------
    dict with keys Pre-COVID, COVID, Post-COVID.
    Each value is a dict:
        clr     : (T, K) DataFrame
        T       : number of observations
        K       : number of crime types
        cov_raw : (K, K) np.ndarray  raw sample covariance
        cov_lw  : (K, K) np.ndarray  LW regularized covariance
        cols    : list of crime type column names
    """
    era_map = {
        'Pre-COVID' : 'pre_covid',
        'COVID'     : 'covid',
        'Post-COVID': 'post_covid',
    }

    matrices = {}
    for era_key, clr_key in era_map.items():
        clr_mat            = clr_era[clr_key]
        matrices[era_key]  = {
            'clr'    : clr_mat,
            'T'      : len(clr_mat),
            'K'      : clr_mat.shape[1],
            'cov_raw': eras_stat['era_covs'][era_key].values,
            'cov_lw' : eras_stat['era_covs_lw'][era_key].values,
            'lw_alpha': eras_stat['lw_shrinkage'][era_key], 
            'cols'   : clr_mat.columns.tolist(),
        }

    # Validation report
    print("=" * 65)
    print("COVARIANCE MATRIX VALIDATION")
    print("=" * 65)
    for era, d in matrices.items():
        K    = d['K']
        flag = '✅' if d['cov_lw'].shape == (K, K) else '❌'
        print(f"  {era:<15}: T={d['T']:<4}  "
              f"cov_raw={d['cov_raw'].shape}  "
              f"cov_lw={d['cov_lw'].shape}  {flag}")

    return matrices


# ============================================================
# LAYER 2 - BOX'S M TEST + PERMUTATION
# ============================================================
def _boxm_statistic(groups):
    """
    Compute Box's M statistic for k groups of observations.

    M = (N - k) * log|S_pool| - Σ_i (n_i - 1) * log|S_i|

    where S_pool is the pooled sample covariance.

    Parameters
    ----------
    groups : list of np.ndarray  each shape (n_i, K)

    Returns
    -------
    M : float
    """
    ns     = np.array([g.shape[0] for g in groups])
    N      = ns.sum()
    k      = len(groups)

    # Per-group sample covariance (unbiased)
    covs   = [np.cov(g.T, ddof=1) for g in groups]

    # Pooled covariance
    S_pool = sum((n - 1) * S for n, S in zip(ns, covs)) / (N - k)

    # Log-determinants - use slogdet for numerical stability
    ld_pool = np.linalg.slogdet(S_pool)[1]
    ld_i    = [np.linalg.slogdet(S)[1] for S in covs]

    M = (N - k) * ld_pool - sum(
        (n - 1) * ld for n, ld in zip(ns, ld_i)
    )
    return float(M)


def run_boxm_test(matrices, n_permutations=10_000,
                  seed=1776, sparse_cats=None):
    """
    Box's M test for equality of covariance matrices across eras,
    with permutation-based p-value for robustness to non-normality.

    Parameters
    ----------
    matrices       : output of _prepare_cov_matrices
    n_permutations : permutation iterations
    seed           : random seed
    sparse_cats    : list of sparse category names to flag

    Returns
    -------
    dict:
        M_obs   : float   observed M statistic
        M_perm  : ndarray permutation distribution
        p_perm  : float   permutation p-value
        n_perm  : int     number of permutations
        reject  : bool    True if p_perm < 0.05
    """
    sparse_cats = set(sparse_cats or [])
    rng         = np.random.default_rng(seed)
    era_keys    = list(matrices.keys())

    groups = [matrices[e]['clr'].values for e in era_keys]
    ns     = [g.shape[0] for g in groups]
    K      = groups[0].shape[1]
    N      = sum(ns)

    # Observed statistic
    M_obs  = _boxm_statistic(groups)

    # Permutation distribution
    all_data = np.vstack(groups)
    M_perm   = np.empty(n_permutations)

    for i in range(n_permutations):
        idx      = rng.permutation(N)
        shuffled = all_data[idx]
        perm_groups, start = [], 0
        for n in ns:
            perm_groups.append(shuffled[start:start + n])
            start += n
        M_perm[i] = _boxm_statistic(perm_groups)

    p_perm = float(np.mean(M_perm >= M_obs))
    reject = p_perm < 0.05

    # Print report
    print()
    print("=" * 65)
    print("BOX'S M TEST - EQUALITY OF COVARIANCE MATRICES")
    print("=" * 65)
    print(f"  Comparison    : {' vs '.join(era_keys)}")
    print(f"  Groups (k)    : {len(era_keys)}")
    print(f"  Variables (K) : {K}")
    print(f"  Sample sizes  : "
          f"{', '.join(f'{e}=T{n}' for e, n in zip(era_keys, ns))}")
    print(f"  Permutations  : {n_permutations:,}  seed={seed}")
    print()
    print(f"  M statistic   : {M_obs:.4f}")
    print(f"  p (permuted)  : {p_perm:.4f}  "
          f"{'✅ reject H0 - structures differ' if reject else '❌ fail to reject H0'}")
    print()
    print("  H0: Σ_pre = Σ_covid = Σ_post")
    print("  H1: At least one covariance matrix differs")
    if sparse_cats:
        print()
        print(f"  Note: {len(sparse_cats)} sparse categories "
              f"included in matrix, flagged ⚠️:")
        for cat in sorted(sparse_cats):
            print(f"    {cat}")

    return {
        'M_obs' : M_obs,
        'M_perm': M_perm,
        'p_perm': p_perm,
        'n_perm': n_permutations,
        'reject': reject,
    }


# ============================================================
# LAYER 3 - CORRELATION HEATMAPS
# ============================================================
def _corr_from_lw(cov_lw, cols):
    """
    Convert LW covariance matrix to correlation matrix.

    Parameters
    ----------
    cov_lw : (K, K) np.ndarray
    cols   : list of column labels

    Returns
    -------
    pd.DataFrame  (K, K) correlation matrix
    """
    std  = np.sqrt(np.diag(cov_lw))
    corr = cov_lw / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cols, columns=cols)


def _label_cols(cols, sparse_cats):
    """Append ⚠️ to sparse category names for plot labels."""
    return [f"{c} ⚠️" if c in sparse_cats else c for c in cols]


def plot_correlation_heatmaps(matrices, sparse_cats=None,
                               save_path=None):
    """
    Plot per-era correlation heatmaps (3 panels, LW regularized).
    Sparse categories labeled ⚠️.

    Parameters
    ----------
    matrices    : output of _prepare_cov_matrices
    sparse_cats : list of sparse category names
    save_path   : directory path for saving

    Returns
    -------
    dict:
        corrs : dict  era -> correlation DataFrame
    """
    sparse_cats = set(sparse_cats or [])
    era_keys    = list(matrices.keys())
    cols        = matrices[era_keys[0]]['cols']
    labeled     = _label_cols(cols, sparse_cats)

    # Build correlation matrices
    corrs = {
        era: _corr_from_lw(matrices[era]['cov_lw'], labeled)
        for era in era_keys
    }

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    for ax, era in zip(axes, era_keys):
        d = matrices[era]
        sns.heatmap(
            corrs[era], ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.3, annot=False,
            cbar_kws={'shrink': 0.6, 'label': 'Correlation'}
        )

        ax.set_title(
            f"{era}  (T={d['T']}, LW α={matrices[era]['lw_alpha']:.4f})",
            fontsize=12, fontweight='bold'
        )

        # ax.set_title(
        #     f"{era}  (T={d['T']}, LW regularized)",
        #     fontsize=12, fontweight='bold'
        # )
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.suptitle(
        'CLR Correlation Structure per Era - Ledoit-Wolf Regularized',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}corr_heatmaps.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {'corrs': corrs}


def plot_correlation_diff_heatmaps(matrices, sparse_cats=None,
                                    save_path=None):
    """
    Plot difference correlation heatmaps (3 panels):
      Panel 1: COVID − Pre-COVID
      Panel 2: Post-COVID − COVID
      Panel 3: Post-COVID − Pre-COVID  (net permanent change)

    Red  = correlation INCREASED in later era.
    Blue = correlation DECREASED in later era.

    Parameters
    ----------
    matrices    : output of _prepare_cov_matrices
    sparse_cats : list of sparse category names
    save_path   : directory path for saving

    Returns
    -------
    dict:
        diff_pc : DataFrame  COVID − Pre-COVID
        diff_cp : DataFrame  Post-COVID − COVID
        diff_pp : DataFrame  Post-COVID − Pre-COVID
        corrs   : dict       era -> correlation DataFrame
    """
    sparse_cats = set(sparse_cats or [])
    era_keys    = list(matrices.keys())
    cols        = matrices[era_keys[0]]['cols']
    labeled     = _label_cols(cols, sparse_cats)
    K           = len(cols)

    # Build correlation matrices
    corrs = {
        era: _corr_from_lw(matrices[era]['cov_lw'], labeled)
        for era in era_keys
    }

    # Three differences - consistent with all other era comparisons
    diff_pc = corrs['COVID']      - corrs['Pre-COVID']
    diff_cp = corrs['Post-COVID'] - corrs['COVID']
    diff_pp = corrs['Post-COVID'] - corrs['Pre-COVID']

    diffs = [diff_pc, diff_cp, diff_pp]


    # Titles with α values from each era
    α_pre  = matrices['Pre-COVID']['lw_alpha']
    α_cov  = matrices['COVID']['lw_alpha']
    α_post = matrices['Post-COVID']['lw_alpha']

    titles = [
        f"COVID (α={α_cov:.4f}) − Pre-COVID (α={α_pre:.4f})\n"
        f"(Red = stronger correlation during COVID)",
        f"Post-COVID (α={α_post:.4f}) − COVID (α={α_cov:.4f})\n"
        f"(Red = stronger correlation post-COVID)",
        f"Post-COVID (α={α_post:.4f}) − Pre-COVID (α={α_pre:.4f})\n"
        f"(Net permanent change in correlation structure)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    mask = np.eye(K, dtype=bool)

    for ax, diff, title in zip(axes, diffs, titles):
        sns.heatmap(
            diff, ax=ax,
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            mask=mask,
            square=True, linewidths=0.3, annot=False,
            cbar_kws={'shrink': 0.6, 'label': 'Δ Correlation'}
        )
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)

    plt.suptitle(
        'CLR Correlation Structure - Era Differences (LW Regularized)',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}corr_diff_heatmaps.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'diff_pc': diff_pc,
        'diff_cp': diff_cp,
        'diff_pp': diff_pp,
        'corrs'  : corrs,
    }


# ============================================================
# LAYER 4 - LOG-DETERMINANT COMPARISON
# ============================================================
def compute_log_determinants(matrices):
    """
    Compute log-determinant of raw and LW covariance matrices per era.

    log|Σ| = log generalized variance.
    Higher -> crime types more spread in CLR space.
    Lower  -> crime types more concentrated / co-moving.

    Raw log|Σ| is unreliable for COVID and Post-COVID due to
    near-singularity from CLR rank deficiency and small T/K.
    LW log|Σ| is used for all interpretation.

    Parameters
    ----------
    matrices : output of _prepare_cov_matrices

    Returns
    -------
    dict:
        per_era  : dict  era -> {logdet_raw, logdet_lw, T, T/K}
        deltas   : dict  comparison -> Δlog|Σ_lw|
    """
    print()
    print("=" * 70)
    print("LOG-DETERMINANT COMPARISON - GENERALIZED VARIANCE")
    print("=" * 70)
    print(f"  {'Era':<15} {'log|Σ_raw|':>14} {'log|Σ_lw|':>14} "
          f"{'T':>6} {'T/K':>8}  {'Status'}")
    print(f"  {'-' * 62}")

    per_era = {}
    for era, d in matrices.items():
        K          = d['K']
        ld_raw     = np.linalg.slogdet(d['cov_raw'])[1]
        ld_lw      = np.linalg.slogdet(d['cov_lw'])[1]
        tk         = d['T'] / K
        flag       = '✅' if tk >= 5 else '⚠️'

        print(f"  {era:<15} {ld_raw:>14.4f} {ld_lw:>14.4f} "
              f"{d['T']:>6} {tk:>8.2f}  {flag}")

        per_era[era] = {
            'logdet_raw': ld_raw,
            'logdet_lw' : ld_lw,
            'T'         : d['T'],
            'T/K'       : tk,
        }

    # Delta comparison using LW
    lw = {e: per_era[e]['logdet_lw'] for e in per_era}
    deltas = {
        'COVID_minus_Pre' : lw['COVID']      - lw['Pre-COVID'],
        'Post_minus_COVID': lw['Post-COVID'] - lw['COVID'],
        'Post_minus_Pre'  : lw['Post-COVID'] - lw['Pre-COVID'],
    }

    print()
    print("  Δ log|Σ_lw| (LW - used for interpretation):")
    print(f"    COVID − Pre-COVID  : {deltas['COVID_minus_Pre']:+.4f}")
    print(f"    Post  − COVID      : {deltas['Post_minus_COVID']:+.4f}")
    print(f"    Post  − Pre-COVID  : {deltas['Post_minus_Pre']:+.4f}")
    print()
    print("  Interpretation:")
    print("    Positive Δ -> crime space EXPANDED (more diverse)")
    print("    Negative Δ -> crime space CONTRACTED (more co-moving)")
    print()
    print("  Note: raw log|Σ| unreliable for COVID/Post-COVID")
    print("        (T/K < 5, CLR rank deficiency). Use LW values.")

    return {
        'per_era': per_era,
        'deltas' : deltas,
    }


# ============================================================
# LAYER 5 - ORCHESTRATOR
# ============================================================
def run_covariance_structure_analysis(eras_stat, clr_era,
                                       sparse_cats=None,
                                       n_permutations=10_000,
                                       seed=1776,
                                       save_path=None):
    """
    Full covariance structure analysis pipeline.

    Steps
    -----
    1. Validate and prepare covariance matrices
    2. Box's M test + permutation p-value
    3. Per-era correlation heatmaps (LW regularized)
    4. Difference correlation heatmaps (3 comparisons)
    5. Log-determinant comparison

    Parameters
    ----------
    eras_stat      : output of compute_era_distribution_parameters
    clr_era        : output of slice_clr_into_eras
    sparse_cats    : list of sparse category names
    n_permutations : Box's M permutation iterations
    seed           : random seed
    save_path      : directory for saving plots

    Returns
    -------
    dict:
        matrices    : validated CLR + cov dicts per era
        boxm        : Box's M results
        corr_heatmaps  : per-era correlation matrices
        corr_diffs     : three difference matrices + corrs
        log_dets    : log-determinant results
    """
    print("\n" + "=" * 65)
    print("COVARIANCE STRUCTURE ANALYSIS")
    print("=" * 65)

    # Step 1 - Prepare and validate
    matrices = _prepare_cov_matrices(eras_stat, clr_era)

    # Step 2 - Box's M + permutation
    boxm = run_boxm_test(
        matrices,
        n_permutations = n_permutations,
        seed           = seed,
        sparse_cats    = sparse_cats,
    )

    # Step 3 - Per-era correlation heatmaps
    print()
    corr_heatmaps = plot_correlation_heatmaps(
        matrices,
        sparse_cats = sparse_cats,
        save_path   = save_path,
    )

    # Step 4 - Difference heatmaps (3 comparisons)
    print()
    corr_diffs = plot_correlation_diff_heatmaps(
        matrices,
        sparse_cats = sparse_cats,
        save_path   = save_path,
    )

    # Step 5 - Log-determinants
    log_dets = compute_log_determinants(matrices)

    print("\n" + "=" * 65)
    print("COVARIANCE STRUCTURE ANALYSIS COMPLETE")
    print("=" * 65)

    return {
        'matrices'     : matrices,
        'boxm'         : boxm,
        'corr_heatmaps': corr_heatmaps,
        'corr_diffs'   : corr_diffs,
        'log_dets'     : log_dets,
    }


# ============================================================
# PCA ANALYSIS
# ============================================================
# Layer 1 - Data Preparation
# Layer 2 - Per-Era PCA
# Layer 3 - Joint PCA
# Layer 4 - Scree Plot
# Layer 5 - Loadings Table
# Layer 6 - Orchestrator
# ============================================================
# LAYER 1 - DATA PREPARATION
# ============================================================
def _prepare_pca_data(clr_era, chosen_clr, sparse_cats,
                      era_config):
    """
    Prepare CLR matrices for PCA by dropping sparse categories
    and building era label arrays for joint PCA.

    Parameters
    ----------
    clr_era     : output of slice_clr_into_eras
    chosen_clr  : (304, 26) full CLR DataFrame
    sparse_cats : list of sparse category names to drop
    era_config  : dict  era -> (start, end, color)

    Returns
    -------
    dict:
        era_matrices : dict  era -> (T, K_dense) DataFrame
        joint_matrix : (304, K_dense) DataFrame
        era_labels   : (304,) array of era strings
        era_colors   : (304,) array of color strings
        cols         : list of K_dense column names
        K            : number of dense crime types
    """
    era_map = {
        'Pre-COVID' : 'pre_covid',
        'COVID'     : 'covid',
        'Post-COVID': 'post_covid',
    }

    # Drop sparse categories from all matrices
    dense_cols = [c for c in chosen_clr.columns
                  if c not in sparse_cats]

    era_matrices = {
        era: clr_era[clr_key][dense_cols]
        for era, clr_key in era_map.items()
    }
    joint_matrix = chosen_clr[dense_cols]

    # Build era label + color arrays for joint PCA
    era_labels = np.empty(len(joint_matrix), dtype=object)
    era_colors = np.empty(len(joint_matrix), dtype=object)

    for era, clr_key in era_map.items():
        idx            = clr_era[clr_key].index
        mask           = joint_matrix.index.isin(idx)
        era_labels[mask] = era
        era_colors[mask] = era_config[era][2]

    print("=" * 65)
    print("PCA DATA PREPARATION")
    print("=" * 65)
    print(f"  Sparse categories dropped : {len(sparse_cats)}")
    for cat in sorted(sparse_cats):
        print(f"    ⚠️  {cat}")
    print(f"  Dense categories retained : {len(dense_cols)}")
    print(f"  Joint matrix shape        : {joint_matrix.shape}")
    for era, mat in era_matrices.items():
        print(f"  {era:<15}: {mat.shape}")

    return {
        'era_matrices': era_matrices,
        'joint_matrix': joint_matrix,
        'era_labels'  : era_labels,
        'era_colors'  : era_colors,
        'cols'        : dense_cols,
        'K'           : len(dense_cols),
    }


# ============================================================
# LAYER 2 - PER-ERA PCA
# ============================================================
def run_per_era_pca(pca_data, n_components=None,
                    variance_threshold=0.80):
    """
    Fit a separate PCA for each era on dense CLR matrices.
    Parameters
    ----------
    pca_data           : output of _prepare_pca_data
    n_components       : fixed number of components.
                         If None, use variance_threshold.
    variance_threshold : cumulative variance to retain
                         if n_components is None
    Returns
    -------
    dict:
        per_era : dict  era -> {
            pca       : fitted sklearn PCA object
            scores    : (T, n_components) DataFrame
            loadings  : (K, n_components) DataFrame
            var_exp   : array of explained variance ratios
            n_comp    : number of components retained
        }
    """
    era_keys = list(pca_data['era_matrices'].keys())
    per_era  = {}

    print()
    print("=" * 65)
    print("PER-ERA PCA")
    print("=" * 65)
    print(f"  Variance threshold : {variance_threshold:.0%}")
    print(f"  Centering          : per-era (subtract era mean)")
    print()

    for era in era_keys:
        X    = pca_data['era_matrices'][era].values
        T, K = X.shape

        pca_full = PCA()
        pca_full.fit(X)
        cumvar   = np.cumsum(pca_full.explained_variance_ratio_)

        if n_components is not None:
            n_comp = n_components
        else:
            n_comp = int(np.searchsorted(cumvar,
                                          variance_threshold) + 1)
            n_comp = min(n_comp, T - 1, K)

        pca          = PCA(n_components=n_comp)
        scores_arr   = pca.fit_transform(X)
        loadings_arr = pca.components_.T

        comp_names = [f"PC{i+1}" for i in range(n_comp)]

        scores   = pd.DataFrame(
            scores_arr,
            index   = pca_data['era_matrices'][era].index,
            columns = comp_names
        )
        loadings = pd.DataFrame(
            loadings_arr,
            index   = pca_data['cols'],
            columns = comp_names
        )

        var_exp = pca.explained_variance_ratio_
        cum_var = np.cumsum(var_exp)

        per_era[era] = {
            'pca'     : pca,
            'pca_full': pca_full,
            'scores'  : scores,
            'loadings': loadings,
            'var_exp' : var_exp,
            'cum_var' : cum_var,
            'n_comp'  : n_comp,
            'T'       : T,
            'K'       : K,
        }

        print(f"  {era:<15}: T={T:<3}  K={K}  "                 
              f"components retained={n_comp}  " 
              f"variance explained={cum_var[n_comp-1]:.1%}")


    print()
    print("  Per-era PC1 / PC2 breakdown:")
    print(f"  {'Era':<15} {'PC1':>6} {'PC2':>8} "
          f"{'n_comp':>8} {'Total':>8} {'PC1-PC2 gap':>12}")
    print(f"  {'-'*62}")
    for era, d in per_era.items():
        pc1   = d['var_exp'][0]
        pc2   = d['var_exp'][1]
        gap   = pc1 - pc2
        total = d['cum_var'][d['n_comp']-1]
        print(f"  {era:<15} {pc1:>7.1%} {pc2:>7.1%} "
              f"{d['n_comp']:>8} {total:>8.1%} {gap:>11.1%}")

    return {'per_era': per_era}

    
# ============================================================
# LAYER 3 - JOINT PCA
# ============================================================
def run_joint_pca(pca_data, n_components=None,
                  variance_threshold=0.80):
    """
    Fit a single PCA on the full 304-month CLR matrix.
    All three eras projected into the same latent space.

    Parameters
    ----------
    pca_data           : output of _prepare_pca_data
    n_components       : fixed number of components
    variance_threshold : cumulative variance to retain

    Returns
    -------
    dict:
        pca      : fitted sklearn PCA object
        scores   : (304, n_comp) DataFrame with era labels
        loadings : (K, n_comp) DataFrame
        var_exp  : explained variance ratios
        n_comp   : components retained
    """
    X    = pca_data['joint_matrix'].values
    T, K = X.shape

    pca_full = PCA()
    pca_full.fit(X)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is not None:
        n_comp = n_components
    else:
        n_comp = int(np.searchsorted(cumvar,
                                      variance_threshold) + 1)
        n_comp = min(n_comp, T - 1, K)

    pca          = PCA(n_components=n_comp)
    scores_arr   = pca.fit_transform(X)
    loadings_arr = pca.components_.T

    comp_names = [f"PC{i+1}" for i in range(n_comp)]

    scores = pd.DataFrame(
        scores_arr,
        index   = pca_data['joint_matrix'].index,
        columns = comp_names
    )
    scores['era']   = pca_data['era_labels']
    scores['color'] = pca_data['era_colors']

    loadings = pd.DataFrame(
        loadings_arr,
        index   = pca_data['cols'],
        columns = comp_names
    )

    var_exp = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)

    # Compute centroids from scores
    comp_cols  = [f"PC{i+1}" for i in range(n_comp)]
    centroids  = (
        scores 
        .groupby('era')[comp_cols]
        .mean()
        .round(4)
    )      

    print()
    print("=" * 65)
    print("JOINT PCA")
    print("=" * 65)
    print(f"  Full matrix       : T={T}, K={K}")
    print(f"  Components kept   : {n_comp}")
    print(f"  Variance explained: {cum_var[n_comp-1]:.1%}")
    for i, (v, cv) in enumerate(zip(var_exp, cum_var)):
        print(f"    PC{i+1}: {v:.1%}  cumulative={cv:.1%}")

    # Era centroids in PC space                                                      
    print()
    print("  Era centroids in joint PC space:")

    table = centroids.to_string(
        index=True,
        justify="right",
        float_format=lambda x: f"{x:>+8.4f}"
    )

    # shift whole block 4 spaces right
    print("\n".join("    " + line for line in table.splitlines()))                                                    

    # PC1 reversion calculation                                
    if ('Pre-COVID' in centroids.index and 'COVID' 
        in centroids.index and 'Post-COVID' in centroids.index):                        
        pre_pc1  = centroids.loc['Pre-COVID',  'PC1']         
        cov_pc1  = centroids.loc['COVID',      'PC1']         
        post_pc1 = centroids.loc['Post-COVID', 'PC1']         
        total    = cov_pc1  - pre_pc1                          
        recovery = cov_pc1  - post_pc1                         
        pct_rev  = recovery / total * 100                      
        pct_perm = 100 - pct_rev                               
        print()                                                 
        print(f"  PC1 structural shift analysis:")             
        print(f"    Pre -> COVID shift     : {total:+.4f}")       
        print(f"    COVID- > Post recovery : {recovery:+.4f}")    
        print(f"    % reversion            : {pct_rev:.1f}%")     
        print(f"    % permanent            : {pct_perm:.1f}%")    


    return {
        'pca'     : pca,
        'pca_full': pca_full,
        'scores'  : scores,
        'loadings': loadings,
        'var_exp' : var_exp,
        'cum_var' : cum_var,
        'n_comp'  : n_comp,
    }


# ============================================================
# LAYER 4 - SCREE PLOTS
# ============================================================
def plot_pca_scree(per_era_results, joint_results,
                   era_config, save_path=None):
    """
    Plot scree plots for per-era and joint PCA side by side.

    Parameters
    ----------
    per_era_results : output of run_per_era_pca
    joint_results   : output of run_joint_pca
    era_config      : dict  era -> (start, end, color)
    save_path       : directory path for saving

    Returns
    -------
    dict: empty (side-effect only)
    """
    era_keys = list(per_era_results['per_era'].keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: per-era scree ──────────────────────────────────
    ax = axes[0]
    for era in era_keys:
        d       = per_era_results['per_era'][era]
        color   = era_config[era][2]
        n_show  = min(10, len(d['pca_full']
                              .explained_variance_ratio_))
        var_exp = d['pca_full'].explained_variance_ratio_[:n_show]
        cum_var = np.cumsum(d['pca_full']
                             .explained_variance_ratio_[:n_show])
        x = np.arange(1, n_show + 1)

        ax.plot(x, var_exp * 100, 'o-',
                color=color, label=f"{era} (T={d['T']})",
                linewidth=2, markersize=6)
        ax.axvline(x=d['n_comp'], color=color,
                   linestyle='--', alpha=0.5)

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('Per-Era Scree Plot\n'
                 '(dashed = components retained)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))

    # ── Right: joint scree ───────────────────────────────────
    ax = axes[1]
    n_show  = min(10, len(joint_results['pca_full']
                           .explained_variance_ratio_))
    var_exp = joint_results['pca_full'] \
                  .explained_variance_ratio_[:n_show] * 100
    cum_var = np.cumsum(
        joint_results['pca_full']
        .explained_variance_ratio_[:n_show]
    ) * 100
    x = np.arange(1, n_show + 1)

    ax.bar(x, var_exp, color='steelblue',
           alpha=0.7, label='Individual')
    ax.plot(x, cum_var, 'ro-',
            linewidth=2, markersize=6, label='Cumulative')
    ax.axhline(y=80, color='gray', linestyle='--',
               alpha=0.7, label='80% threshold')
    ax.axvline(x=joint_results['n_comp'],
               color='darkred', linestyle='--',
               alpha=0.7,
               label=f"Components kept={joint_results['n_comp']}")

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_title('Joint PCA Scree Plot\n'
                 f"(T=304, K={joint_results['loadings'].shape[0]})",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, n_show + 1))

    plt.suptitle(
        'PCA Scree Plots - Dense CLR Crime Categories (K=23)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}pca_scree.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {}


# ============================================================
# LAYER 5 - JOINT PCA SCATTER + LOADINGS
# ============================================================
def plot_joint_pca_scatter(joint_results, era_config,
                           save_path=None):
    """
    Plot PC1 vs PC2 scatter colored by era for joint PCA.

    Returns
    -------
    dict: empty (side-effect only)
    """
    scores   = joint_results['scores']
    era_keys = list(era_config.keys())

    fig, ax = plt.subplots(figsize=(12, 8))

    for era in era_keys:
        color  = era_config[era][2]
        mask   = scores['era'] == era
        subset = scores[mask]
        ax.scatter(
            subset['PC1'], subset['PC2'],
            c=color, label=f"{era} (n={mask.sum()})",
            alpha=0.6, s=40, edgecolors='white',
            linewidths=0.3
        )

    # Era centroids
    for era in era_keys:
        color  = era_config[era][2]
        mask   = scores['era'] == era
        cx     = scores.loc[mask, 'PC1'].mean()
        cy     = scores.loc[mask, 'PC2'].mean()
        ax.scatter(cx, cy, c=color, s=200,
                   marker='*', edgecolors='black',
                   linewidths=1.0, zorder=5)
        ax.annotate(
            era, (cx, cy),
            textcoords='offset points',
            xytext=(8, 4),
            fontsize=10, fontweight='bold', color=color
        )

    var1 = joint_results['var_exp'][0] * 100
    var2 = joint_results['var_exp'][1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance explained)",
                  fontsize=12)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance explained)",
                  fontsize=12)
    ax.set_title(
        'Joint PCA - Monthly CLR Scores by Era\n'
        '★ = era centroid',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=11)
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}pca_joint_scatter.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {}


def plot_pca_loadings(joint_results, per_era_results,
                      era_config, n_top=8,
                      save_path=None):
    """
    Plot top crime type loadings for PC1 and PC2
    for joint and per-era PCA side by side.

    Parameters
    ----------
    n_top : number of top loadings to show per component

    Returns
    -------
    dict:
        joint_top  : dict  PC -> top-n loadings Series
        per_era_top: dict  era -> PC -> top-n loadings Series
    """
    era_keys   = list(per_era_results['per_era'].keys())
    joint_load = joint_results['loadings']

    fig, axes  = plt.subplots(2, 4, figsize=(28, 12))

    joint_top   = {}
    per_era_top = {era: {} for era in era_keys}

    for pc_idx, pc in enumerate(['PC1', 'PC2']):

        # ── Joint PCA loadings ────────────────────────────
        ax    = axes[pc_idx][0]
        if pc in joint_load.columns:
            load  = joint_load[pc].sort_values()
            top   = pd.concat([load.head(n_top//2),
                                load.tail(n_top//2)])
            colors_bar = ['#d73027' if v > 0
                          else '#4575b4' for v in top.values]
            ax.barh(range(len(top)), top.values,
                    color=colors_bar, alpha=0.8)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(
                [c[:28] for c in top.index],
                fontsize=8
            )
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(f"Joint PCA - {pc}",
                         fontsize=11, fontweight='bold')
            ax.set_xlabel('Loading', fontsize=9)
            joint_top[pc] = top

        # ── Per-era loadings ──────────────────────────────
        for era_idx, era in enumerate(era_keys):
            ax    = axes[pc_idx][era_idx + 1]
            color = era_config[era][2]
            d     = per_era_results['per_era'][era]
            if pc in d['loadings'].columns:
                load  = d['loadings'][pc].sort_values()
                top   = pd.concat([load.head(n_top//2),
                                    load.tail(n_top//2)])
                colors_bar = [color if v > 0
                              else '#aaaaaa'
                              for v in top.values]
                ax.barh(range(len(top)), top.values,
                        color=colors_bar, alpha=0.8)
                ax.set_yticks(range(len(top)))
                ax.set_yticklabels(
                    [c[:28] for c in top.index],
                    fontsize=8
                )
                ax.axvline(0, color='black', linewidth=0.8)
                ax.set_title(
                    f"{era} - {pc}\n(T={d['T']})",
                    fontsize=11, fontweight='bold',
                    color=color
                )
                ax.set_xlabel('Loading', fontsize=9)
                per_era_top[era][pc] = top
            else:
                ax.set_visible(False)

    plt.suptitle(
        'PCA Loadings - PC1 and PC2\n'
        'Top crime types driving each component per era',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}pca_loadings.png",
                    dpi=300, bbox_inches='tight')
    plt.show()

    return {
        'joint_top'  : joint_top,
        'per_era_top': per_era_top,
    }


# ============================================================
# LAYER 6 - ORCHESTRATOR
# ============================================================
def run_pca_analysis(clr_era, chosen_clr, sparse_cats,
                     era_config, n_components=None,
                     variance_threshold=0.80,
                     save_path=None):
    """
    Full PCA analysis pipeline.

    Steps
    -----
    1. Prepare data - drop sparse, build joint matrix
    2. Per-era PCA - separate model per era
    3. Joint PCA   - single model, all eras
    4. Scree plots - per-era + joint
    5. Joint scatter - months colored by era
    6. Loadings - top crime types per component

    Parameters
    ----------
    clr_era            : output of slice_clr_into_eras
    chosen_clr         : (304, 26) full CLR DataFrame
    sparse_cats        : list of sparse categories to drop
    era_config         : dict  era -> (start, end, color)
    n_components       : fixed components (None = use threshold)
    variance_threshold : cumulative variance target (default 0.80)
    save_path          : directory for saving plots

    Returns
    -------
    dict:
        pca_data        : prepared matrices
        per_era         : per-era PCA results
        joint           : joint PCA results
        loadings        : top loadings per component
    """
    print("\n" + "=" * 65)
    print("PCA ANALYSIS")
    print("=" * 65)

    # Step 1 - Prepare
    pca_data = _prepare_pca_data(
        clr_era, chosen_clr, sparse_cats, era_config
    )

    # Step 2 - Per-era PCA
    per_era = run_per_era_pca(
        pca_data,
        n_components       = n_components,
        variance_threshold = variance_threshold,
    )

    # Step 3 - Joint PCA
    joint = run_joint_pca(
        pca_data,
        n_components       = n_components,
        variance_threshold = variance_threshold,
    )

    # Step 4 - Scree plots
    print()
    plot_pca_scree(per_era, joint, era_config, save_path)

    # Step 5 - Joint scatter
    print()
    plot_joint_pca_scatter(joint, era_config, save_path)

    # Step 6 - Loadings
    print()
    loadings = plot_pca_loadings(
        joint, per_era, era_config,
        n_top=8, save_path=save_path
    )

    print("\n" + "=" * 65)
    print("PCA ANALYSIS COMPLETE")
    print("=" * 65)

    return {
        'pca_data': pca_data,
        'per_era' : per_era,
        'joint'   : joint,
        'loadings': loadings,
    }