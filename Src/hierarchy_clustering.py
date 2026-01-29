import numpy as np
from scipy.cluster.hierarchy import inconsistent

def choose_clusters_from_inconsistency(Z, method, metric, depth=2):
    """
    Estimate the number of clusters in a hierarchical clustering using
    inconsistency coefficients.

    This function computes inconsistency statistics for each merge in the
    linkage matrix and identifies the largest jump between consecutive
    inconsistency coefficients (excluding the final/root merge). The intuition
    is that a large jump indicates a transition from merging similar clusters
    to merging dissimilar ones, suggesting a natural cut in the dendrogram.

    Parameters
    ----------
    Z : ndarray of shape (n_samples - 1, 4)
        Linkage matrix as produced by `scipy.cluster.hierarchy.linkage`.

    depth : int, default=2
        The maximum depth to compute inconsistency statistics, passed directly
        to `scipy.cluster.hierarchy.inconsistent`.
        depth = 0 or 1: The algorithm only considers the current node’s link height.
                        The inconsistency will always be 0 because there are no other 
                         heights to compare it to.
        depth = 2: The algorithm looks at the current node and its two immediate children 
                   (if they aren't leaves).
        depth = 3: The algorithm looks at the current node, its children, and its grandchildren.

    Returns
    -------
    n_clusters : int
        Estimated number of clusters based on the largest jump in
        inconsistency coefficients.

    jump_idx : int
        Index of the merge at which the largest jump in inconsistency
        coefficients occurs.

    inc : ndarray of shape (n_samples - 1,)
        Inconsistency coefficients for each merge in the linkage matrix.

    diffs : ndarray of shape (n_samples - 2,)
        Differences between consecutive inconsistency coefficients.

    Notes
    -----
    - The final/root merge is ignored when searching for the largest jump,
      since it always combines the two largest clusters.
    - This heuristic works best when the data exhibits a well-separated,
      hierarchical structure.
    """
    I = inconsistent(Z, depth)
    inc = I[:, -1]  # inconsistency coefficients

    # Compute differences between consecutive inconsistency values
    diffs = np.diff(inc)

    # Ignore the final merge (root)
    diffs_no_root = diffs[:-1]

    # Find the largest jump
    jump_idx = np.argmax(diffs_no_root)

    # Number of clusters = merges remaining after that jump
    n_clusters = Z.shape[0] - jump_idx

    print("----- Using Inconsistency coefficient Matrix ------")
    print(f"Method: {method}, Metric: {metric}, Depth: {depth}")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Jump occurred at merge index {jump_idx}\n")

    return n_clusters



def choose_clusters_from_linkage(Z, method, metric):
    """
    Automatically estimate the number of clusters from a hierarchical
    clustering linkage matrix by detecting the largest jump in merge heights.

    This function analyzes the distances (heights) at which clusters are merged
    in the linkage matrix and looks for the largest increase between consecutive
    merges (excluding the final/root merge). A large jump is interpreted as the
    point where the algorithm starts merging well-separated clusters, suggesting
    a natural cut in the dendrogram.

    Parameters
    ----------
    Z : ndarray of shape (n_samples - 1, 4)
        Linkage matrix produced by `scipy.cluster.hierarchy.linkage`.

    Returns
    -------
    n_clusters : int
        Estimated number of clusters corresponding to the cut just before
        the largest jump in merge heights.

    jump_idx : int
        Index of the merge at which the largest jump in height occurs.

    diffs : ndarray of shape (n_samples - 2,)
        Differences between consecutive merge heights.

    Notes
    -----
    - The final/root merge is ignored when identifying the largest jump,
      since it almost always represents a trivially large distance.
    - This heuristic works best when clusters are reasonably well-separated
      and the dendrogram shows clear height gaps.
    """
    heights = Z[:, 2]
    diffs = np.diff(heights)

    # Ignore the final merge (root merge)
    # because it always produces a huge jump
    diffs_no_root = diffs[:-1]

    # Find index of largest meaningful jump
    jump_idx = np.argmax(diffs_no_root)

    # Number of clusters = number of merges remaining after that jump
    n_clusters = Z.shape[0] - jump_idx

    print("----- Using Linkage Matrix -----")
    print(f"Method: {method}, Metric: {metric}")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Jump at merge index: {jump_idx}\n")
    return n_clusters
