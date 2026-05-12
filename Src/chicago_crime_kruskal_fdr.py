from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd

# --------------------------------------------------------------------------------------------------
# 1. Kruskal-Wallis test with Benjamini-Hochberg FDR correction and epsilon-squared effect size calculation
# --------------------------------------------------------------------------------------------------
def run_kruskal_fdr(data: pd.DataFrame, time_col: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    Runs Kruskal-Wallis test across crime types and eras,
    applies Benjamini-Hochberg FDR correction,
    and computes the epsilon-squared effect size.

    Parameters
    ----------
    data     : DataFrame with columns [fbi_code_desc, era, time_col, count]
    time_col : Temporal grouping column e.g. 'year_month' or 'year_week'
    alpha    : Significance level (default 0.05)

    Returns
    -------
    pd.DataFrame with columns:
        [crime, statistic, p_value, p_corrected, significant,
         n, epsilon_squared, effect_size]
    """
    k = 3  # number of era groups

    # ────────────────────────────────────────────────────────────────────────────
    # 1-A Aggregate by crime + era + time period
    # ----─────────────────────────────────────────────────────────────────────────
    aggregated = (
        data
        .groupby(['fbi_code_desc', 'era', time_col], as_index=False, observed=True)
        .agg(total=('count', 'sum'))
    )

    # ────────────────────────────────────────────────────────────────────────────
    # 1-B Kruskal-Wallis test for each crime type across eras
    # ----─────────────────────────────────────────────────────────────────────────
    results = []
    for crime in aggregated['fbi_code_desc'].astype(str).unique():
        subset = aggregated[aggregated['fbi_code_desc'].astype(str) == crime]
        pre   = subset[subset['era'].astype(str) == 'pre_covid']['total']
        covid = subset[subset['era'].astype(str) == 'covid']['total']
        post  = subset[subset['era'].astype(str) == 'post_covid']['total']

        # Skip if any group is empty - Kruskal-Wallis requires at least one observation per group
        if any(len(g) == 0 for g in [pre, covid, post]):
            continue
        
        # Perform Kruskal-Wallis test
        stat, p = stats.kruskal(pre, covid, post)

        # Calculate epsilon-squared effect size: (H - k + 1) / (n - k)
        n = len(pre) + len(covid) + len(post)
        epsilon_sq = round((stat - k + 1) / (n - k), 3)

        results.append({
            'crime':            crime,
            'statistic':        round(stat, 2),
            'p_value':          round(p, 6),
            'n':                n,
            'epsilon_squared':  epsilon_sq,
        })

    # ───────────────────────────────────────────────────────────────────────────
    # 1-C: Benjamini-Hochberg FDR correction
    # ----─────────────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)
    reject, p_corrected, _, _ = multipletests(
        results_df['p_value'],
        alpha=alpha,
        method='fdr_bh'
    )
    results_df['p_corrected'] = p_corrected.round(6)
    results_df['significant'] = reject

    # ──────────────────────────────────────────────────────────────────────────
    # 1-D: Label effect sizes based on epsilon-squared thresholds
    # ----─────────────────────────────────────────────────────────────────────────
    def label_effect(e):
        if e >= 0.14:   return 'large'
        elif e >= 0.06: return 'medium'
        elif e >= 0.01: return 'small'
        else:           return 'negligible'

    # Clip negative epsilon-squared values to 0 (no effect) before labeling
    results_df['epsilon_squared'] = results_df['epsilon_squared'].clip(lower=0)
    # Apply effect size labeling
    results_df['effect_size'] = results_df['epsilon_squared'].apply(label_effect)

    return results_df