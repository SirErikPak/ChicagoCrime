# ../Src/dunn_posthoc.py
"""
dunn_posthoc.py
---------------
Pairwise Dunn's post-hoc test following a significant Kruskal-Wallis result.

Mirrors the interface of kruskal_fdr.run_kruskal_fdr() for consistency.

Public API
----------
run_dunn_posthoc(data, time_col, kruskal_results, alpha=0.05) -> pd.DataFrame

Dependencies
------------
    pip install scikit-posthocs
"""

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import rankdata, mannwhitneyu
from statsmodels.stats.multitest import multipletests


# ── Constants ────────────────────────────────────────────────────────────────
ERA_COL    = 'era'
CRIME_COL  = 'fbi_code_desc'
COUNT_COL  = 'count'
ERA_ORDER  = ['pre_covid', 'covid', 'post_covid']
PAIRS      = list(combinations(ERA_ORDER, 2))   # 3 pairwise comparisons


# ── Effect Size ───────────────────────────────────────────────────────────────
def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-biserial correlation for two independent samples (Mann-Whitney U).
    Range: -1 (y dominates) to +1 (x dominates). 0 = no difference.
    Formula: r = 1 - (2U) / (n1 * n2)
    Reference: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Effect_sizes
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    u_stat, _ = mannwhitneyu(x, y, alternative='two-sided')
    return float(1 - (2 * u_stat) / (n1 * n2))


def _effect_label(r: float) -> str:
    """
    Classify rank-biserial r into magnitude labels.
    Thresholds: |r| < 0.10 negligible, < 0.30 small,
                < 0.50 medium, >= 0.50 large
    Reference: Cohen (1988) adapted for rank-biserial
    """
    abs_r = abs(r)
    if abs_r < 0.10:
        return 'negligible'
    elif abs_r < 0.30:
        return 'small'
    elif abs_r < 0.50:
        return 'medium'
    else:
        return 'large'


# ── Dunn's Test (manual — no scikit-posthocs dependency) ─────────────────────
def _dunn_pvalue(group_a: np.ndarray, group_b: np.ndarray,
                 all_values: np.ndarray) -> float:
    """
    Dunn's z-test p-value for two groups drawn from a pooled ranked sample.
    Uses the standard Dunn (1964) formula.
    Reference: https://www.tandfonline.com/doi/abs/10.1080/00401706.1964.10490181
    """
    n      = len(all_values)
    ranks  = rankdata(all_values)

    # Reconstruct per-group ranks from pooled ranking
    n_a    = len(group_a)
    n_b    = len(group_b)

    # Get rank positions for each group value in pooled array
    idx_a  = np.isin(all_values, group_a)
    idx_b  = np.isin(all_values, group_b)

    mean_rank_a = ranks[idx_a].mean()
    mean_rank_b = ranks[idx_b].mean()

    # Tie correction
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_correction = np.sum(tie_counts ** 3 - tie_counts) / (12 * (n - 1))

    se = np.sqrt(
        (n * (n + 1) / 12 - tie_correction) * (1 / n_a + 1 / n_b)
    )

    if se == 0:
        return 1.0

    z = (mean_rank_a - mean_rank_b) / se
    from scipy.stats import norm
    p = 2 * norm.sf(abs(z))
    return float(p)


# ── Main Function ─────────────────────────────────────────────────────────────
def run_dunn_posthoc(
    data:             pd.DataFrame,
    time_col:         str,
    kruskal_results:  pd.DataFrame,
    alpha:            float = 0.05
) -> pd.DataFrame:
    """
    Run pairwise Dunn's post-hoc test for all crimes that were
    statistically significant in the Kruskal-Wallis step.

    Parameters
    ----------
    data            : multi_counts_df — must contain fbi_code_desc, era, count
    time_col        : aggregation column used in kruskal step (e.g. 'year_month')
    kruskal_results : output of kruskal_fdr.run_kruskal_fdr() — used to filter
                      to significant crimes only
    alpha           : FDR significance threshold (default 0.05)

    Returns
    -------
    pd.DataFrame with columns:
        crime, pair, p_value, p_corrected, significant,
        rank_biserial, effect_size
    """

    # ── Step 1: Filter to significant crimes only ─────────────────────────
    sig_crimes = (
        kruskal_results
        .loc[kruskal_results['significant'], 'crime']
        .tolist()
    )

    # ── Step 2: Aggregate to (crime, era, time_col) → sum of counts ───────
    agg = (
        data
        .groupby([CRIME_COL, ERA_COL, time_col], observed=True)[COUNT_COL]
        .sum()
        .reset_index()
    )

    # ── Step 3: Pairwise Dunn's per significant crime ─────────────────────
    rows = []

    for crime in sig_crimes:
        crime_data  = agg.loc[agg[CRIME_COL] == crime]
        all_values  = crime_data[COUNT_COL].to_numpy(dtype=float)
        raw_pvals   = []

        for era_a, era_b in PAIRS:
            vals_a = crime_data.loc[crime_data[ERA_COL] == era_a, COUNT_COL].to_numpy(dtype=float)
            vals_b = crime_data.loc[crime_data[ERA_COL] == era_b, COUNT_COL].to_numpy(dtype=float)

            p_raw = _dunn_pvalue(vals_a, vals_b, all_values)
            r     = _rank_biserial(vals_a, vals_b)
            raw_pvals.append((era_a, era_b, p_raw, r))

        # ── FDR correction across the 3 pairs for this crime ──────────────
        p_values = [x[2] for x in raw_pvals]
        _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

        for (era_a, era_b, p_raw, r), p_corr in zip(raw_pvals, p_corrected):
            rows.append({
                'crime'         : crime,
                'pair'          : f'{era_a}  vs  {era_b}',
                'p_value'       : round(p_raw,  6),
                'p_corrected'   : round(p_corr, 6),
                'significant'   : p_corr < alpha,
                'rank_biserial' : round(r, 4),
                'effect_size'   : _effect_label(r)
            })

    return (
        pd.DataFrame(rows)
        .sort_values(['crime', 'pair'])
        .reset_index(drop=True)
    )