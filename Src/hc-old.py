"""
hc.py — Hierarchical Clustering Module
=======================================
Provides hierarchical clustering utilities for crime era analysis.

Functions:
    linkage_matrix                    — computes scipy linkage matrix
    inconsistency_matrix              — computes inconsistency coefficient matrix
    choose_clusters_from_inconsistency— finds k using inconsistency threshold search
    choose_clusters_from_linkage      — finds k using largest merge distance gap
    consensus_k                       — compares both methods and reports agreement
    compute_dtw_distance_matrix       — computes weighted era-separated DTW distance matrix
    compute_correlation_distance_matrix — computes correlation distance matrix

Usage:
    import hc
    Z = hc.linkage_matrix(X, method='average', metric='correlation')
    I = hc.inconsistency_matrix(Z)
    k_I = hc.choose_clusters_from_inconsistency(Z, method='average', metric='correlation')
    k_L = hc.choose_clusters_from_linkage(Z, method='average', metric='correlation')
    k   = hc.consensus_k(k_I, k_L)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import inconsistent
from scipy.spatial.distance import pdist, squareform


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLDS   = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25]
DEFAULT_DEPTH        = 10      # inconsistency window depth
DEFAULT_WARPING_PCT  = 0.10    # DTW Sakoe-Chiba band as % of series length
DEFAULT_ERA_WEIGHTS  = {
    'pre_covid' : 0.15,
    'covid'     : 0.55,
    'post_covid': 0.30,
}


# ── 1. Linkage Matrix ──────────────────────────────────────────────────────────
def linkage_matrix(X, method='average', metric='correlation'):
    """
    Compute the hierarchical clustering linkage matrix.

    Parameters:
        X      : np.ndarray or pd.DataFrame — input data (n_samples × n_features)
                 If metric='precomputed', X must be a square distance matrix
        method : str — linkage method ('single', 'complete', 'average')
        metric : str — distance metric ('correlation', 'euclidean', 'precomputed')

    Returns:
        Z : np.ndarray — linkage matrix (n-1 × 4)
            columns: [cluster_i, cluster_j, distance, n_items]

    Notes:
        - For correlation distance, X should be the raw feature matrix (n × p)
        - For precomputed distance, X should be a condensed distance vector
          or square matrix (will be converted automatically)
        - method='ward' is not supported for non-Euclidean metrics
    """
    if isinstance(X, pd.DataFrame):
        X = X.values

    # ward linkage only works with euclidean — guard against misuse
    if method == 'ward' and metric != 'euclidean':
        raise ValueError(
            f"Ward linkage requires Euclidean metric. "
            f"Got metric='{metric}'. Use 'single', 'complete', or 'average'."
        )

    if metric == 'precomputed':
        # X is a square distance matrix — convert to condensed form
        if X.ndim == 2:
            condensed = squareform(X, checks=False)
        else:
            condensed = X
        Z = linkage(condensed, method=method)
    else:
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
    return I


# ── 3. Choose Clusters from Inconsistency ─────────────────────────────────────
def choose_clusters_from_inconsistency(
    Z, method='average', metric='correlation',
    thresholds=DEFAULT_THRESHOLDS, depth=DEFAULT_DEPTH
):
    """
    Find optimal k by searching inconsistency coefficient thresholds.

    Cuts the dendrogram where the inconsistency coefficient exceeds each
    threshold and reports k per threshold plus a consensus k where
    multiple thresholds agree.

    Parameters:
        Z          : np.ndarray — linkage matrix
        method     : str — linkage method (for labeling only)
        metric     : str — distance metric (for labeling only)
        thresholds : list — inconsistency thresholds to search
        depth      : int — inconsistency window depth

    Returns:
        dict with keys:
            'method'     : linkage method
            'metric'     : distance metric
            'threshold_k': {threshold: k} for each threshold
            'consensus_k': most common k across thresholds
            'agreement'  : fraction of thresholds that agree on consensus_k
            'I'          : inconsistency matrix
    """
    I      = inconsistency_matrix(Z, depth=depth)
    n      = len(Z) + 1   # number of original observations
    result = {}

    for t in thresholds:
        # fcluster cuts dendrogram at inconsistency threshold
        labels = fcluster(Z, t=t, criterion='inconsistent', depth=depth)
        k      = len(np.unique(labels))
        result[t] = k

    # consensus k — most common value across thresholds
    k_values     = list(result.values())
    consensus    = max(set(k_values), key=k_values.count)
    agreement    = k_values.count(consensus) / len(k_values)

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
            'method'         : linkage method
            'metric'         : distance metric
            'k'              : optimal number of clusters
            'gap_magnitude'  : size of the largest gap
            'gap_location'   : merge step where gap occurs
            'merge_distances': full sequence of merge distances
            'gaps'           : full sequence of gaps between merges
    """
    # merge distances are in column 2 of Z
    merge_distances = Z[:, 2]
    n               = len(merge_distances)

    # compute gaps between consecutive merge distances
    gaps     = np.diff(merge_distances)
    max_gap  = np.argmax(gaps)

    # k = number of clusters when we cut just before the largest gap
    # at step max_gap, there are (n - max_gap) clusters remaining
    k            = n - max_gap
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
    Compare k from inconsistency and linkage methods and report agreement.

    Parameters:
        k_inconsistency : dict — output from choose_clusters_from_inconsistency()
        k_linkage       : dict — output from choose_clusters_from_linkage()

    Returns:
        dict with keys:
            'k_inconsistency' : consensus k from inconsistency method
            'k_linkage'       : k from linkage gap method
            'agreed'          : True if both methods agree
            'final_k'         : agreed k if agreement, else None
            'recommendation'  : human-readable recommendation string
            'confidence'      : 'high', 'moderate', or 'low'
    """
    k_I = k_inconsistency['consensus_k']
    k_L = k_linkage['k']
    agreed = (k_I == k_L)

    if agreed:
        final_k       = k_I
        confidence    = 'high' if k_inconsistency['agreement'] >= 0.5 else 'moderate'
        recommendation = (
            f"Both methods agree on k={final_k}. "
            f"Inconsistency agreement: {k_inconsistency['agreement']*100:.0f}% of thresholds."
        )
    else:
        final_k       = None
        confidence    = 'low'
        recommendation = (
            f"Methods disagree — inconsistency suggests k={k_I}, "
            f"linkage gap suggests k={k_L}. "
            f"Inspect dendrogram to resolve."
        )

    return {
        'k_inconsistency' : k_I,
        'k_linkage'        : k_L,
        'agreed'           : agreed,
        'final_k'          : final_k,
        'recommendation'   : recommendation,
        'confidence'       : confidence,
    }


# ── 6. Correlation Distance Matrix ────────────────────────────────────────────
def compute_correlation_distance_matrix(X, labels=None):
    """
    Compute the correlation distance matrix from a feature matrix.

    Correlation distance = 1 - Pearson correlation
    Range: [0, 2] where 0 = identical, 1 = uncorrelated, 2 = opposite

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


# ── 7. DTW Distance Matrix ────────────────────────────────────────────────────
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
    try:
        from tslearn.metrics import cdist_dtw
    except ImportError:
        raise ImportError(
            "tslearn is required for DTW computation. "
            "Install with: pip install tslearn"
        )

    n      = len(crime_labels)
    eras   = ['pre_covid', 'covid', 'post_covid']

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
        combined = np.concatenate(
            [era_series[era] for era in eras],
            axis=1
        )
        warping_window = max(1, int(warping_pct * combined.shape[1]))
        dist_square    = cdist_dtw(combined, global_constraint='sakoe_chiba',
                                   sakoe_chiba_radius=warping_window)

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
            era_dist      /= series.shape[1]
            dist_square   += era_weights[era] * era_dist

    else:
        raise ValueError(f"option must be 'A' or 'B'. Got '{option}'.")

    dist_matrix = pd.DataFrame(dist_square, index=crime_labels, columns=crime_labels)
    return dist_matrix


# ── 8. Summary Printer ────────────────────────────────────────────────────────
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
        marker = ' ← consensus' if k == k_inconsistency['consensus_k'] else ''
        print(f"  threshold={t:<5} → k={k}{marker}")
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
    print(f"{'='*60}")