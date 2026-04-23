"""
=======================================
Provides hierarchical clustering utilities for crime era analysis.

Research question:
    COVID-19 immediately reorganized the directional co-movement of crime
    types, but the reorganization of temporal rhythms occurred later in
    the post-COVID period.

Functions:
    linkage_matrix                      — computes scipy linkage matrix
    inconsistency_matrix                — computes inconsistency coefficient matrix
    choose_clusters_from_inconsistency  — finds k using inconsistency threshold search
    choose_clusters_from_linkage        — finds k using largest merge distance gap
    consensus_k                         — compares both methods and reports agreement
    compute_correlation_distance_matrix — computes correlation distance matrix
    compute_dtw_distance_matrix         — computes weighted era-separated DTW distance matrix
    compute_dtw_single_era              — computes DTW distance matrix for a single era
    compare_dtw_era_rhythms             — tests rhythm reorganization hypothesis across eras
    print_cluster_summary               — prints clean cluster selection summary

Distance matrix roles:
    corr_dist   — directional co-movement similarity (correlation)
    dtw_pre     — baseline rhythm structure (pre-COVID)
    dtw_covid   — COVID-era rhythm structure
    dtw_post    — post-COVID rhythm structure
    dtw_dist_B  — weighted combined DTW (clustering input)

Hypothesis test:
    rho(dtw_pre, dtw_post) < rho(dtw_pre, dtw_covid)
    → post-COVID rhythms diverged MORE from baseline than COVID rhythms
    → confirms temporal rhythm reorganization occurred later

Usage:
    import hc
    Z    = hc.linkage_matrix(X, method='average', metric='correlation')
    I    = hc.inconsistency_matrix(Z)
    k_I  = hc.choose_clusters_from_inconsistency(Z, method='average', metric='correlation')
    k_L  = hc.choose_clusters_from_linkage(Z, method='average', metric='correlation')
    k    = hc.consensus_k(k_I, k_L)

    # Per-era DTW for hypothesis testing
    dtw_pre   = hc.compute_dtw_single_era(era_data_filled, crime_labels, era='pre_covid')
    dtw_covid = hc.compute_dtw_single_era(era_data_filled, crime_labels, era='covid')
    dtw_post  = hc.compute_dtw_single_era(era_data_filled, crime_labels, era='post_covid')
    results   = hc.compare_dtw_era_rhythms(dtw_pre, dtw_covid, dtw_post)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import inconsistent
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

# ── Optional dependency: tslearn ───────────────────────────────────────────────
# Loaded once at module level — avoids repeated import overhead on every DTW call.
# Reference: https://tslearn.readthedocs.io/en/stable/installation.html
try:
    from tslearn.metrics import cdist_dtw
    _TSLEARN_AVAILABLE = True
except ImportError:
    _TSLEARN_AVAILABLE = False

# ── Public API ─────────────────────────────────────────────────────────────────
# Defines what is exported when a caller does `from hc import *`.
# Prevents internal names (np, pd, linkage, spearmanr, etc.) from leaking.
# Reference: https://docs.python.org/3/tutorial/modules.html#importing-from-a-package
__all__ = [
    'linkage_matrix',
    'inconsistency_matrix',
    'choose_clusters_from_inconsistency',
    'choose_clusters_from_linkage',
    'consensus_k',
    'compute_correlation_distance_matrix',
    'compute_dtw_distance_matrix',
    'compute_dtw_single_era',
    'compare_dtw_era_rhythms',
    'print_cluster_summary',
]


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DEPTH       = 3      # inconsistency window depth
DEFAULT_N_STEPS     = 6      # number of thresholds to search between 75 percentile and max inconsistency
DEFAULT_WARPING_PCT = 0.10   # DTW Sakoe-Chiba band as % of series length
DEFAULT_ERA_WEIGHTS = {
    'pre_covid' : 0.15,
    'covid'     : 0.55,
    'post_covid': 0.30,
}


# ── 1. Linkage Matrix ──────────────────────────────────────────────────────────
def linkage_matrix(X, method='average', metric='correlation'):
    """
    Compute the hierarchical clustering linkage matrix.

    Parameters:
        X      : np.ndarray or pd.DataFrame - input data
                 - If metric='precomputed': square (n×n) or condensed (n*(n-1)/2,)
                   distance matrix. For crime data: 26×26 or length-325 vector.
                 - Otherwise: raw feature matrix (n_samples × n_features).
                   For crime data: (26 crimes × N time points) - rows are crimes.
        method : str - linkage method ('single', 'complete', 'average').
                 'ward' is not supported for non-Euclidean metrics.
        metric : str - any scipy-valid metric string ('correlation', 'euclidean',
                 'cityblock', 'cosine', etc.), or 'precomputed'.

    Returns:
        Z : np.ndarray - linkage matrix (n-1 × 4)
            columns: [cluster_i, cluster_j, distance, n_items]

    Raises:
        ValueError : if method='ward' with non-Euclidean metric
        ValueError : if X.ndim not in (1, 2)
        ValueError : if X is 1D and the length is not a valid condensed vector size
        ValueError : if X is 2D precomputed and the matrix is not square

    Notes:
        - For correlation distance, X must be the raw feature matrix (n × p)
          where rows are the items being clustered (e.g., 26 crimes × N years).
          A transposed matrix would cluster features instead of samples.
        - For precomputed distance, symmetry, and zero-diagonal are assumed valid
          (validated upstream in the notebook). squareform checks=False is used
          intentionally to avoid redundant validation.
        - Ward linkage is explicitly blocked for non-Euclidean metrics because
          Ward minimizes the within-cluster sum of squares, which requires Euclidean
          geometry to be mathematically valid.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    # ── Ward guard ────────────────────────────────────────────────────────────
    if method == 'ward' and metric != 'euclidean':
        raise ValueError(
            f"Ward linkage requires Euclidean metric. "
            f"Got metric='{metric}'. Use 'single', 'complete', or 'average'."
        )

    # ── Precomputed distance path ─────────────────────────────────────────────
    if metric == 'precomputed':
        if X.ndim == 2:
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Precomputed square matrix must be symmetric (n×n). "
                    f"Got shape {X.shape}."
                )
            condensed = squareform(X, checks=False)

        elif X.ndim == 1:
            # Validate: condensed vector length must satisfy n*(n-1)/2
            n_pairs = len(X)
            n       = (1 + np.sqrt(1 + 8 * n_pairs)) / 2
            if not np.isclose(n, round(n)):
                raise ValueError(
                    f"Condensed distance vector length {n_pairs} does not "
                    f"correspond to a valid square matrix. "
                    f"Expected n*(n-1)/2 elements (e.g., 325 for n=26)."
                )
            condensed = X

        else:
            raise ValueError(
                f"For metric='precomputed', X must be 1D (condensed vector) "
                f"or 2D (square matrix). Got {X.ndim}D array."
            )

        Z = linkage(condensed, method=method)

    # ── Feature matrix path (correlation, euclidean, cityblock, etc.) ─────────
    else:
        if X.ndim != 2:
            raise ValueError(
                f"For metric='{metric}', X must be a 2D feature matrix "
                f"(n_samples × n_features). Got {X.ndim}D array. "
                f"For crime data: shape should be (26, N_timepoints)."
            )
        Z = linkage(X, method=method, metric=metric)

    return Z


# ── 2. Inconsistency Matrix ────────────────────────────────────────────────────
def inconsistency_matrix(Z, depth=DEFAULT_DEPTH):
    """
    Compute the inconsistency coefficient matrix from a linkage matrix.

    Parameters:
        Z     : np.ndarray — linkage matrix from linkage_matrix()
        depth : int — number of levels to include in local window (default 10)
                Higher depth = more context for small datasets

    Returns:
        I : np.ndarray — inconsistency matrix (n-1 × 4)
            columns: [mean_distance, std_distance, n_merges, inconsistency_coeff]

    Notes:
        - inconsistency_coeff = (merge_distance - mean) / std
        - Higher coefficient = more unusual merge = potential cluster boundary
        - For datasets < 30 observations, depth=10 recommended
    """
    I = inconsistent(Z, d=depth)
    return np.round(I, 4)


# ── 2.5 Compute Thresholds ────────────────────────────────────────────────────
def compute_thresholds(I, n_steps=DEFAULT_N_STEPS):
    """
    Compute data-driven inconsistency thresholds from the
    inconsistency matrix.

    Thresholds span from the 75th percentile of non-zero inconsistency
    scores to the maximum, plus one collapse point above the maximum.
    The 75th percentile boundary is the empirically-derived signal
    threshold — merges below this are local and structurally uninformative.

    Parameters:
        I       : np.ndarray — inconsistency matrix from scipy inconsistent()
                  Column 3 contains the inconsistency coefficients.
        n_steps : int — number of evenly spaced thresholds from p75 to max
                  inclusive. One collapse point is always appended above the max.
    Returns:
        list of float — threshold values rounded to 4 decimal places,
                        sorted ascending. Length = n_steps + 1.
    Raises:
        ValueError — if no valid (non-zero) inconsistency scores exist.
    """
    # ── Extract and filter inconsistency scores ───────────────────────────────
    scores       = I[:, 3]
    valid_scores = scores[scores > 0]

    if len(valid_scores) == 0:
        raise ValueError(
            "No valid inconsistency scores found in column 3 of I. "
            "Check linkage matrix Z and depth parameter."
        )

    # ── Define range from empirical signal boundary to ceiling ───────────────
    start   = np.percentile(valid_scores, 75)   # signal starts at 75th percentile
    ceiling = np.max(valid_scores)

    # ── Build evenly spaced thresholds p75 -> max ─────────────────────────────
    thresholds = np.linspace(start, ceiling, n_steps).tolist()

    # ── Append collapse point just above max ─────────────────────────────────
    # Buffer scales with range to ensure reliable k=1 capture
    collapse_buffer = (ceiling - start) * 0.1
    thresholds.append(ceiling + collapse_buffer)

    return [round(t, 4) for t in sorted(thresholds)]


# ── 3. Choose Clusters from Inconsistency ────────────────────────────────────
def choose_clusters_from_inconsistency(
    Z, method='average', metric='correlation', 
    depth=DEFAULT_DEPTH, n_steps=DEFAULT_N_STEPS
):
    """
    Find optimal k by searching data-driven inconsistency thresholds.

    Thresholds are computed dynamically from the inconsistency matrix
    using compute_thresholds(), anchored to the 75th percentile of
    non-zero inconsistency scores. This ensures thresholds are scaled
    to each distance matrix rather than using a fixed global constant.

    Parameters:
        Z       : np.ndarray - linkage matrix
        method  : str - linkage method (for labeling only)
        metric  : str - distance metric (for labeling only)
        depth   : int - inconsistency window depth (default=DEFAULT_DEPTH).
                  Thresholds are computed dynamically from the inconsistency
                  matrix using compute_thresholds().
        n_steps : int - number of thresholds to search between 75 percentile
                  and maximum inconsistency score. (default=DEFAULT_N_STEPS)

    Returns:
        dict with keys:
            'method'      : linkage method
            'metric'      : distance metric
            'threshold_k' : {threshold: k} for each threshold
            'consensus_k' : most common k across thresholds
            'agreement'   : fraction of thresholds that agree on consensus_k
            'I'           : inconsistency matrix
    """
    # ── Compute inconsistency matrix and derive data-driven thresholds ────────
    I          = inconsistency_matrix(Z, depth=depth)
    thresholds = compute_thresholds(I, n_steps=n_steps)

    # ── Cut dendrogram at each threshold and record k ─────────────────────────
    result = {}
    for t in thresholds:
        labels    = fcluster(Z, t=t, criterion='inconsistent', depth=depth)
        k         = len(np.unique(labels))
        result[t] = k

    # ── Consensus k — most common value across thresholds ────────────────────
    k_values  = list(result.values())
    consensus = max(set(k_values), key=k_values.count)
    agreement = k_values.count(consensus) / len(k_values)

    # ── Warn on degenerate results ────────────────────────────────────────────
    n_obs = len(Z) + 1
    if consensus == 1:
        warnings.warn(
            f"consensus_k=1: all {len(k_values)} thresholds collapsed to a "
            f"single cluster. Thresholds may be too high or data is genuinely "
            f"homogeneous. Current thresholds: {thresholds}",
            UserWarning,
            stacklevel=2
        )
    elif consensus > n_obs // 3:
        warnings.warn(
            f"consensus_k={consensus} exceeds n/3 ({n_obs//3}) for "
            f"n={n_obs} observations. Thresholds may still be too low. "
            f"Inspect dendrogram before proceeding.",
            UserWarning,
            stacklevel=2
        )

    return {
        'method'      : method,
        'metric'      : metric,
        'threshold_k' : result,
        'consensus_k' : consensus,
        'agreement'   : round(agreement, 3),
        'I'           : I,
    }


# ── 4. Choose Clusters from Linkage Distance ──────────────────────────────────
def choose_clusters_from_linkage(Z, method='average', metric='correlation'):
    """
    Find optimal k by finding the largest gap in merge distances.

    Looks at the sequence of merge distances in the linkage matrix and
    finds the largest jump — the natural elbow where merging becomes
    expensive relative to previous merges.

    Parameters:
        Z      : np.ndarray — linkage matrix
        method : str — linkage method (for labeling only)
        metric : str — distance metric (for labeling only)

    Returns:
        dict with keys:
            'method'          : linkage method
            'metric'          : distance metric
            'k'               : optimal number of clusters
            'gap_magnitude'   : size of the largest gap
            'gap_location'    : merge step where gap occurs
            'merge_distances' : full sequence of merge distances
            'gaps'            : full sequence of gaps between merges
    """
    # merge distances are in column 2 of Z
    merge_distances = Z[:, 2]

    # compute gaps between consecutive merge distances
    gaps    = np.diff(merge_distances)
    max_gap = np.argmax(gaps)

    # k = number of clusters when we cut just before the largest gap
    # len(merge_distances) = n-1 merges for n observations
    # at step max_gap, there are (len(merge_distances) - max_gap) clusters remaining
    k             = len(merge_distances) - max_gap
    gap_magnitude = gaps[max_gap]

    return {
        'method'          : method,
        'metric'          : metric,
        'k'               : int(k),
        'gap_magnitude'   : round(float(gap_magnitude), 6),
        'gap_location'    : int(max_gap),
        'merge_distances' : merge_distances,
        'gaps'            : gaps,
    }


# ── 5. Consensus K ────────────────────────────────────────────────────────────
def consensus_k(k_inconsistency, k_linkage):
    """
    Reconcile k estimates from the inconsistency and linkage-gap methods.

    Parameters:
        k_inconsistency : dict — output of choose_clusters_from_inconsistency()
                          Must contain keys: 'consensus_k', 'agreement'
        k_linkage       : dict — output of choose_clusters_from_linkage()
                          Must contain keys: 'k'

    Returns:
        dict with keys:
            'k_inconsistency' : int  — consensus k from the inconsistency method
            'k_linkage'       : int  — k from linkage gap method
            'agreed'          : bool — True if both methods agree exactly
            'final_k'         : int or None — agreed k (exact), midpoint
                                (near-miss ±2), or None (large disagreement)
            'recommendation'  : str  — human-readable recommendation
            'confidence'      : str  — 'high', 'moderate', or 'low'

    Confidence rules:
        high     : exact agreement AND inconsistency supermajority (>=67%)
        moderate : exact agreement with weak inconsistency (<67%), OR
                   near-miss disagreement (|k_I - k_L| <= 2)
        low      : methods disagree by more than 2
    """
    k_I  = k_inconsistency['consensus_k']
    k_L  = k_linkage['k']

    # ── Guard: upstream methods may return None ───────────────────────────────
    if k_I is None or k_L is None:
        return {
            'k_inconsistency' : k_I,
            'k_linkage'       : k_L,
            'agreed'          : False,
            'final_k'         : None,
            'recommendation'  : (
                f"Cannot determine consensus: inconsistency returned k={k_I}, "
                f"linkage returned k={k_L}. Check upstream methods."
            ),
            'confidence'      : 'low',
        }
    
    diff = abs(k_I - k_L)

    # ── Exact agreement ───────────────────────────────────────────────────────
    if diff == 0:
        agreed  = True
        final_k = k_I
        # Supermajority threshold: 2/3 of inconsistency thresholds must agree
        if k_inconsistency['agreement'] >= 0.67:
            confidence     = 'high'
            recommendation = (
                f"Both methods agree on k={final_k}. "
                f"Strong inconsistency agreement: "
                f"{k_inconsistency['agreement']*100:.0f}% of thresholds."
            )
        else:
            confidence     = 'moderate'
            recommendation = (
                f"Both methods agree on k={final_k}, but inconsistency "
                f"agreement is weak ({k_inconsistency['agreement']*100:.0f}% "
                f"of thresholds). Inspect dendrogram to confirm."
            )

    # ── Near-miss: methods differ by ≤2 ──────────────────────────────────────
    elif diff <= 2:
        agreed         = False
        final_k        = round((k_I + k_L) / 2)   # midpoint as candidate
        confidence     = 'moderate'
        recommendation = (
            f"Methods nearly agree — inconsistency suggests k={k_I}, "
            f"linkage gap suggests k={k_L} (difference={diff}). "
            f"Midpoint k={final_k} suggested. "
            f"Inspect dendrogram and use silhouette scores to confirm."
        )

    # ── Large disagreement ────────────────────────────────────────────────────
    else:
        agreed         = False
        final_k        = None
        confidence     = 'low'
        recommendation = (
            f"Methods disagree — inconsistency suggests k={k_I}, "
            f"linkage gap suggests k={k_L} (difference={diff}). "
            f"Inspect dendrogram to resolve. "
            f"Suggested range to evaluate: k in [{min(k_I, k_L)}, {max(k_I, k_L)}]. "
            f"Use silhouette scores to guide final selection."
        )

    return {
        'k_inconsistency' : k_I,
        'k_linkage'       : k_L,
        'agreed'          : agreed,
        'final_k'         : final_k,
        'recommendation'  : recommendation,
        'confidence'      : confidence,
    }


# ── 6. Correlation Distance Matrix ────────────────────────────────────────────
def compute_correlation_distance_matrix(X, labels=None):
    """
    Compute the correlation distance matrix from a feature matrix.

    Correlation distance = 1 - Pearson correlation
    Range: [0, 2] where 0 = identical, 1 = uncorrelated, 2 = opposite

    Captures directional co-movement similarity — addresses Part 1 of the
    research question: COVID-19 reorganized directional co-movement of crimes.

    Parameters:
        X      : np.ndarray or pd.DataFrame — feature matrix (n_samples × n_features)
        labels : list — optional row/column labels for output DataFrame

    Returns:
        dist_matrix : pd.DataFrame — symmetric distance matrix (n × n)
    """
    if isinstance(X, pd.DataFrame):
        if labels is None:
            labels = X.index.tolist()
        X = X.values

    condensed   = pdist(X, metric='correlation')
    dist_square = squareform(condensed)

    if labels is not None:
        dist_matrix = pd.DataFrame(dist_square, index=labels, columns=labels)
    else:
        dist_matrix = pd.DataFrame(dist_square)

    return dist_matrix


# ── 7. DTW Distance Matrix (weighted combined) ────────────────────────────────
def compute_dtw_distance_matrix(
    era_data_filled,
    crime_labels,
    era_weights=DEFAULT_ERA_WEIGHTS,
    warping_pct=DEFAULT_WARPING_PCT,
    option='B'
):
    """
    Compute weighted era-separated DTW distance matrix.

    Supports two options:
        Option A — concatenated (all 300 months as one series)
        Option B — era-separated (three series per crime, weighted by era)

    Used as the primary clustering input for Stage 5b.

    Parameters:
        era_data_filled : dict — {era: pd.DataFrame} with crime_count column
        crime_labels    : list — crime names (must match fbi_code_desc values)
        era_weights     : dict — {era: weight} for Option B (must sum to 1.0)
        warping_pct     : float — Sakoe-Chiba band as fraction of series length
        option          : str — 'A' (concatenated) or 'B' (era-separated)

    Returns:
        dist_matrix : pd.DataFrame — symmetric DTW distance matrix (n × n)

    Notes:
        - Uses tslearn cdist_dtw with Sakoe-Chiba band constraint
        - Option B weights: pre_covid=0.15, covid=0.55, post_covid=0.30
        - Distances are normalized by series length for comparability
    """
    if not _TSLEARN_AVAILABLE:
        raise ImportError(
            "tslearn is required for DTW computation. "
            "Install with: pip install tslearn — "
            "https://tslearn.readthedocs.io/en/stable/installation.html"
        )

    n    = len(crime_labels)
    eras = ['pre_covid', 'covid', 'post_covid']

    # Build per-era time series — shape (n_crimes, n_months, 1)
    era_series = {}
    for era in eras:
        df   = era_data_filled[era].copy()
        wide = (
            df.pivot(index='date', columns='fbi_code_desc', values='crime_count')
            .sort_index()
            [crime_labels]
            .values
            .T
            .astype(float)
        )
        # tslearn expects shape (n_samples, n_timestamps, n_features)
        era_series[era] = wide[:, :, np.newaxis]

    if option == 'A':
        # Concatenate all eras into one series per crime
        combined       = np.concatenate(
            [era_series[era] for era in eras], axis=1
        )
        warping_window = max(1, int(warping_pct * combined.shape[1]))
        dist_square    = cdist_dtw(
            combined,
            global_constraint='sakoe_chiba',
            sakoe_chiba_radius=warping_window
        )

    elif option == 'B':
        # Validate weights sum to 1.0
        weight_sum = sum(era_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Era weights must sum to 1.0. Got {weight_sum:.4f}. "
                f"Weights: {era_weights}"
            )

        # Compute weighted sum of per-era DTW distances
        dist_square = np.zeros((n, n))
        for era in eras:
            series         = era_series[era]
            warping_window = max(1, int(warping_pct * series.shape[1]))
            era_dist       = cdist_dtw(
                series,
                global_constraint='sakoe_chiba',
                sakoe_chiba_radius=warping_window
            )
            # normalize by series length for comparability across eras
            era_dist    /= series.shape[1]
            dist_square += era_weights[era] * era_dist

    else:
        raise ValueError(f"option must be 'A' or 'B'. Got '{option}'.")

    dist_matrix = pd.DataFrame(dist_square, index=crime_labels, columns=crime_labels)
    return dist_matrix


# ── 8. DTW Single Era Distance Matrix ─────────────────────────────────────────
def compute_dtw_single_era(
    era_data_filled,
    crime_labels,
    era,
    warping_pct=DEFAULT_WARPING_PCT,
    length_normalize=True
):
    """
    Compute DTW distance matrix for a single era.

    Used for hypothesis testing — compares rhythm structure across eras:
        dtw_pre   = baseline rhythm structure
        dtw_covid = COVID-era rhythm structure
        dtw_post  = post-COVID rhythm structure

    Addresses Part 2 of the research question: temporal rhythm reorganization
    occurred later in the post-COVID period.

    Parameters:
        era_data_filled  : dict — {era: pd.DataFrame} with crime_count column
        crime_labels     : list — crime names (must match fbi_code_desc values)
        era              : str — 'pre_covid', 'covid', or 'post_covid'
        warping_pct      : float — Sakoe-Chiba band as fraction of series length
        length_normalize : bool — divide distances by series length (default True)
                           Enables fair comparison across eras of different lengths:
                           pre_covid=230 months, covid=34 months, post_covid=36 months

    Returns:
        dist_matrix : pd.DataFrame — symmetric DTW distance matrix (n × n)

    Notes:
        - Warping window = max(1, int(warping_pct × series_length))
        - Pre-COVID:  warping_window = max(1, int(0.10 × 230)) = 23 months
        - COVID:      warping_window = max(1, int(0.10 × 34))  = 3 months
        - Post-COVID: warping_window = max(1, int(0.10 × 36))  = 3 months
    """
    if not _TSLEARN_AVAILABLE:
        raise ImportError(
            "tslearn is required for DTW computation. "
            "Install with: pip install tslearn — "
            "https://tslearn.readthedocs.io/en/stable/installation.html"
        )

    valid_eras = ['pre_covid', 'covid', 'post_covid']
    if era not in valid_eras:
        raise ValueError(
            f"era must be one of {valid_eras}. Got '{era}'."
        )

    # Build time series for this era — shape (n_crimes, n_months, 1)
    df   = era_data_filled[era].copy()
    wide = (
        df.pivot(index='date', columns='fbi_code_desc', values='crime_count')
        .sort_index()
        [crime_labels]
        .values
        .T
        .astype(float)
    )
    series = wide[:, :, np.newaxis]

    n_months       = series.shape[1]
    warping_window = max(1, int(warping_pct * n_months))

    dist_square = cdist_dtw(
        series,
        global_constraint='sakoe_chiba',
        sakoe_chiba_radius=warping_window
    )

    # Normalize by series length for fair comparison across eras
    # Pre-COVID has 230 months vs 34/36 for COVID/post-COVID
    # Without normalization pre-COVID distances would be systematically larger
    if length_normalize:
        dist_square /= n_months

    dist_matrix = pd.DataFrame(
        dist_square,
        index=crime_labels,
        columns=crime_labels
    )
    return dist_matrix


# ── 9. Compare DTW Era Rhythms ────────────────────────────────────────────────
def compare_dtw_era_rhythms(dtw_pre, dtw_covid, dtw_post):
    """
    Test the rhythm reorganization hypothesis across eras.

    Hypothesis:
        COVID-19 immediately reorganized directional co-movement (correlation),
        but temporal rhythm reorganization occurred later in post-COVID.

    Tests whether post-COVID rhythms diverged MORE from baseline than
    COVID rhythms — using Spearman rank correlation between distance matrices.

    Lower rho = more divergence from baseline = more rhythm reorganization.

    Parameters:
        dtw_pre   : pd.DataFrame — pre-COVID DTW distance matrix (baseline)
        dtw_covid : pd.DataFrame — COVID-era DTW distance matrix
        dtw_post  : pd.DataFrame — post-COVID DTW distance matrix

    Returns:
        dict with keys:
            'rho_pre_covid'       : Spearman rho between pre and COVID rhythms
            'rho_pre_post'        : Spearman rho between pre and post-COVID rhythms
            'rho_covid_post'      : Spearman rho between COVID and post-COVID rhythms
            'p_pre_covid'         : p-value for pre vs COVID
            'p_pre_post'          : p-value for pre vs post-COVID
            'p_covid_post'        : p-value for COVID vs post-COVID
            'hypothesis_supported': True if rho_pre_post < rho_pre_covid
            'rhythm_shift_covid'  : degree of rhythm change during COVID (1 - rho)
            'rhythm_shift_post'   : degree of rhythm change in post-COVID (1 - rho)
            'interpretation'      : human-readable result string

    Notes:
        - Spearman rho values are valid measures of rank agreement between matrices.
        - P-values are approximate: the 325 pairwise distances (26×25/2) are not
          independent — they share row/column entries — which violates the standard
          independence assumption and makes p-values anti-conservative (too small).
          Treat p-values as indicative rather than exact.
        - The Mantel test is the correct permutation-based approach for testing
          matrix correlation significance when independence cannot be assumed.
    """
    # ── Input validation ───────────────────────────────────────────────────────
    shapes = [dtw_pre.shape, dtw_covid.shape, dtw_post.shape]
    if len(set(shapes)) != 1:
        raise ValueError(
            f"All DTW matrices must have the same shape. "
            f"Got pre={dtw_pre.shape}, covid={dtw_covid.shape}, "
            f"post={dtw_post.shape}."
        )
    if dtw_pre.shape[0] != dtw_pre.shape[1]:
        raise ValueError(
            f"DTW matrices must be square. Got shape {dtw_pre.shape}."
        )

    # ── Flatten upper triangles for pairwise comparison ───────────────────────
    n   = len(dtw_pre)
    idx = np.triu_indices(n, k=1)
    # Flattened vectors of pairwise distances (length n*(n-1)/2)
    pre_flat   = dtw_pre.values[idx]
    covid_flat = dtw_covid.values[idx]
    post_flat  = dtw_post.values[idx]
    # Compute Spearman correlations between flattened distance vectors
    rho_pre_covid,  p_pre_covid  = spearmanr(pre_flat, covid_flat)
    rho_pre_post,   p_pre_post   = spearmanr(pre_flat, post_flat)
    rho_covid_post, p_covid_post = spearmanr(covid_flat, post_flat)

    # rhythm shift = 1 - rho (higher = more divergence from baseline)
    rhythm_shift_covid = 1 - rho_pre_covid
    rhythm_shift_post  = 1 - rho_pre_post

    # hypothesis: post-COVID rhythms diverged MORE from baseline than COVID rhythms
    # lower rho = more divergence
    hypothesis_supported = rho_pre_post < rho_pre_covid
    # Interpretation will differ based on whether the hypothesis is supported or not.
    if hypothesis_supported:
        interpretation = (
            f"Hypothesis SUPPORTED — post-COVID rhythms diverged more from "
            f"baseline (ρ={rho_pre_post:.3f}) than COVID rhythms "
            f"(ρ={rho_pre_covid:.3f}). Temporal rhythm reorganization "
            f"occurred later in the post-COVID period."
        )
    else:
        interpretation = (
            f"Hypothesis NOT SUPPORTED — COVID rhythms diverged more from "
            f"baseline (ρ={rho_pre_covid:.3f}) than post-COVID rhythms "
            f"(ρ={rho_pre_post:.3f}). Rhythm reorganization was immediate, "
            f"not delayed."
        )

    return {
        'rho_pre_covid'        : round(rho_pre_covid,  4),
        'rho_pre_post'         : round(rho_pre_post,   4),
        'rho_covid_post'       : round(rho_covid_post, 4),
        'p_pre_covid'          : round(p_pre_covid,    6),
        'p_pre_post'           : round(p_pre_post,     6),
        'p_covid_post'         : round(p_covid_post,   6),
        'hypothesis_supported' : hypothesis_supported,
        'rhythm_shift_covid'   : round(rhythm_shift_covid, 4),
        'rhythm_shift_post'    : round(rhythm_shift_post,  4),
        'interpretation'       : interpretation,
    }


# ── 10. Summary Printer ───────────────────────────────────────────────────────
def print_cluster_summary(k_inconsistency, k_linkage, consensus):
    """
    Print a clean summary of cluster selection results.

    Parameters:
        k_inconsistency : dict — output from choose_clusters_from_inconsistency()
        k_linkage       : dict — output from choose_clusters_from_linkage()
        consensus       : dict — output from consensus_k()
    """
    method = k_inconsistency['method']
    metric = k_inconsistency['metric']

    print(f"\n{'='*60}")
    print(f"Cluster Selection: method={method}, metric={metric}")
    print(f"{'='*60}")

    print(f"\nInconsistency method (depth={DEFAULT_DEPTH}):")
    for t, k in k_inconsistency['threshold_k'].items():
        marker = ' <- consensus' if k == k_inconsistency['consensus_k'] else ''
        print(f"  threshold={t:<5} -> k={k}{marker}")
    print(f"  consensus_k  = {k_inconsistency['consensus_k']}")
    print(f"  agreement    = {k_inconsistency['agreement']*100:.0f}% of thresholds")

    print(f"\nLinkage gap method:")
    print(f"  k            = {k_linkage['k']}")
    print(f"  gap_magnitude= {k_linkage['gap_magnitude']:.4f}")
    print(f"  gap_location = merge step {k_linkage['gap_location']}")

    print(f"\nConsensus:")
    print(f"  agreed       = {consensus['agreed']}")
    print(f"  final_k      = {consensus['final_k']}")
    print(f"  confidence   = {consensus['confidence']}")
    print(f"  {consensus['recommendation']}")

    # If methods disagree, provide actionable guidance for next steps.
    if not consensus['agreed']:
        ki = consensus['k_inconsistency']
        kl = consensus['k_linkage']
        print(f"\n  ACTION REQUIRED: Methods disagree — inspect the dendrogram.")
        if ki is not None and kl is not None:
            lo, hi = min(ki, kl), max(ki, kl)
            print(f"  Suggested range to evaluate: k in [{lo}, {hi}]")
    print(f"  Use silhouette scores to guide final selection.")
    print(f"{'='*60}")


# ── 11. Inspect raw inconsistency matrix ───────────────────────────────────────────────
def inspect_inconsistency(Z, d=3, label=""):
    """
    Prints a formatted analysis of the inconsistency matrix to help
    calibrate fcluster thresholds.

    Footnote:
        Rows marked with '*' indicate merges whose inconsistency score
        is at or above the 75th percentile of the observed scores.
        These are relatively heterogeneous merges and may be useful
        candidates to examine when choosing a stricter threshold.
    """

    I = inconsistent(Z, d=d)
    scores = I[:, 3]

    p75 = np.percentile(scores, 75)
    p90 = np.percentile(scores, 90)

    title = f" INCONSISTENCY INSPECTOR: {label} "
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

    print(f"{'Step':<6} {'Mean':>8} {'StdDev':>8} {'Count':>6} {'Incons':>8}")
    print("-" * 45)

    for i, row in enumerate(I):
        incons = row[3]
        marker = "*" if incons >= p75 and incons > 0 else " "
        print(f"{i:<6} {row[0]:>8.4f} {row[1]:>8.4f} {row[2]:>6.0f} {incons:>8.4f} {marker}")

    non_zero = scores[scores > 0]

    print(f"\n{' STATISTICS ':-^60}")
    print(f"{'Min Score:':<25} {scores.min():.4f}")
    print(f"{'Max Score:':<25} {scores.max():.4f}")
    print(f"{'Mean Score:':<25} {scores.mean():.4f}")
    print(f"{'Median (Non-Zero):':<25} {np.median(non_zero) if len(non_zero) > 0 else 0:.4f}")
    print(f"{'75th Percentile:':<25} {p75:.4f}")
    print(f"{'90th Percentile:':<25} {p90:.4f}")
    print(f"{'='*60}")

    print("\n* Indicates merges with inconsistency at or above the 75th percentile.")
    print("  These merges are comparatively more heterogeneous and are worth checking")
    print("  when considering stricter threshold values.\n")

    return