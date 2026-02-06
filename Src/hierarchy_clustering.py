import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hierarchy


def linkage_matrix(data: pd.DataFrame, method: str, metric: str):
    """
    Compute the linkage matrix for hierarchical clustering.

    Parameters:
    - data: pd.DataFrame, rows as observations, columns as features
    - method: str, linkage method ('single', 'complete', 'average', 'ward', etc.)
    - metric: str, distance metric ('euclidean', 'correlation', etc.)

    Returns:
    - Z: linkage matrix
    """
    Z = hierarchy.linkage(data.values, method=method, metric=metric, optimal_ordering=True)
    return Z


def choose_clusters_from_inconsistency(Z, method, metric, depth=2):
    # Compute the inconsistency matrix
    I = hierarchy.inconsistent(Z, depth)
    inc = I[:, -1]  # inconsistency coefficients

    # Compute differences between consecutive inconsistency values
    diffs = np.diff(inc)

    # Ignore the final merge (root)
    diffs_no_root = diffs[:-1]

    # Find the largest jump (the merge where the tree “breaks” apart)
    jump_idx = np.argmax(diffs_no_root)

    if jump_idx == 0:
        print("Warning: Largest jump in inconsistency is at the first merge. "
              "This may indicate that the data does not have a clear cluster structure.")
    
    # Number of clusters BEFORE the big jump
    n_samples = Z.shape[0] + 1
    n_clusters = n_samples - (jump_idx + 1)

    print("----- Using Inconsistency coefficient Matrix ------")
    print(f"Method: {method}, Metric: {metric}, Depth: {depth}")
    print(f"Estimated number of clusters:  {n_clusters}")
    # print(f"Jump occurred at merge index {jump_idx}\n")

    return n_clusters, I


def choose_clusters_from_linkage(Z, method, metric):
    # Analyze the linkage matrix to determine the optimal number of clusters
    heights = Z[:, 2]
    diffs = np.diff(heights)

    # Ignore the final merge (root merge)
    # because it always produces a huge jump
    diffs_no_root = diffs[:-1]

    # Find index of largest meaningful jump
    jump_idx = np.argmax(diffs_no_root)

    if jump_idx == 0:
        print("Warning: Largest jump in linkage heights is at the first merge. "
              "This may indicate that the data does not have a clear cluster structure.")

    # Number of clusters BEFORE the big jump
    n_clusters = (Z.shape[0] + 1) - (jump_idx + 1)

    print("----- Using Linkage Matrix -----")
    print(f"Method: {method}, Metric: {metric}")
    print(f"Estimated number of clusters: {n_clusters}\n")
    # print(f"Jump at merge index: {jump_idx}\n")
    return n_clusters
