import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score


def build_slim_network(corr_df):
    """
    Build a maximum-parsimony pruned network:
    each node keeps only its strongest edge.
    """

    # Build a full weighted graph from the adjacency matrix
    G_full = nx.from_pandas_adjacency(corr_df)

    # Initialize pruned graph
    G_slim = nx.Graph()

    # Maximum-parsimony pruning (keep the strongest edge per node)
    for node in G_full.nodes():
        neighbors = G_full[node]

        if len(neighbors) > 0:
            # Select neighbor with the highest correlation (edge weight)
            best_buddy = max(neighbors, key=lambda x: neighbors[x]["weight"])

            # Add only the strongest connection
            G_slim.add_edge(
                node,
                best_buddy,
                weight=neighbors[best_buddy]["weight"]
            )

    return G_slim


def compute_influence(G):
    """
    Compute node influence as weighted degree (node strength).

    Returns:
    - dict: node → influence score

    Notes:
    - Handles missing weights safely
    - Explicitly includes isolated nodes
    """
    # initialize variable
    influence_dict = {}

    for node in G.nodes():

        # Sum all incident edge weights safely
        influence = sum(
            data.get("weight", 1)  # fallback = 1 if missing
            for _, _, data in G.edges(node, data=True)
        )

        influence_dict[node] = influence

    return influence_dict
    


def compute_louvain(G):
    """
    Run Louvain community detection and return:
    - partition (node → community)
    - modularity score

    Notes:
    - Partition is a dictionary (unordered)
    - Must be explicitly aligned later for comparisons (ARI, plots, etc.)
    """

    # Run Louvain clustering on weighted graph
    partition = community_louvain.best_partition(G, weight="weight")

    # Compute modularity of resulting partition
    modularity_score = community_louvain.modularity(partition, G)

    # Sanity check (ensures full coverage of nodes)
    missing_nodes = set(G.nodes()) - set(partition.keys())
    if missing_nodes:
        raise ValueError(f"Partition missing nodes: {missing_nodes}")

    return partition, modularity_score
    


def visualize_network(G, influence_dict, partition, seed):

    plt.figure(figsize=(12, 8))  # bigger canvas

    # Better spaced layout (avoid 1/sqrt(N) collapse)
    pos = nx.spring_layout(
        G,
        k=1.5,              # more separation between nodes
        iterations=300,
        seed=seed
    )

    # Slight global expansion for breathing room
    pos = {n: p * 2.2 for n, p in pos.items()}

    # Node size = influence
    node_sizes = [
        300 + (influence_dict[n] * 600)
        for n in G.nodes()
    ]

    # Edge weights (normalized safely)
    weights = [G[u][v]["weight"] for u, v in G.edges()]

    if len(weights) > 0:
        w_min, w_max = min(weights), max(weights)
        edge_widths = [
            ((w - w_min) / (w_max - w_min) * 3) + 0.8
            if w_max != w_min else 1.5
            for w in weights
        ]
    else:
        edge_widths = []

    # Community coloring
    node_colors = [partition[n] for n in G.nodes()]

    # Draw network (NO labels yet)
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        width=edge_widths,
        cmap="viridis",
        edge_color="gray",
        alpha=0.35,
        with_labels=False
    )

    # ADD LABELS SEPARATELY
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=9,
        font_color="black",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1)
    )

    plt.title("Chicago Crime: Systemic Network Architecture", fontsize=16)
    plt.axis("off")
    plt.show()
    


def validate_clusters(G, partition, original_labels, txt=''):
    """
    Compare Louvain communities vs original clustering using ARI.

    original_labels may be:
    - list/array-like aligned to G.nodes() order
    - pandas Series indexed by node name (will be reindexed to G.nodes())
    """
    nodes = list(G.nodes())

    # Ensure partition covers all nodes before scoring.
    missing_partition_nodes = [node for node in nodes if node not in partition]
    if missing_partition_nodes:
        raise ValueError(
            "Partition does not include all graph nodes. "
            f"Missing nodes: {missing_partition_nodes[:10]}"
        )

    # ALL nodes
    louvain_labels = [partition[node] for node in nodes]

    # Accept a pandas Series keyed by node names and align it safely.
    if hasattr(original_labels, "reindex") and hasattr(original_labels, "isna"):
        aligned_labels = original_labels.reindex(nodes)
        if aligned_labels.isna().any():
            missing_original = aligned_labels[aligned_labels.isna()].index.tolist()
            raise ValueError(
                "original_labels is missing labels for graph nodes. "
                f"Missing nodes: {missing_original[:10]}"
            )
        original_labels_aligned = aligned_labels.tolist()
    else:
        original_labels_aligned = list(original_labels)

    if len(original_labels_aligned) != len(nodes):
        raise ValueError(
            "Label length mismatch for ARI calculation. "
            f"Expected {len(nodes)} labels (one per graph node), "
            f"got {len(original_labels_aligned)}."
        )

    # In the original data, were these two in the same group
    ari_score = adjusted_rand_score(original_labels_aligned, louvain_labels)

    print(f"Adjusted Rand Index: {ari_score:.4f}")
    print(f"Near 0 or negative => network structure differs from original clustering {txt}")

    return ari_score



def run_pipeline(corr_df, corr_pre_clusters, seed, txt='', visualize=True):
    """
    Execute the full pipeline:
      1) Build a 1-NN network
      2) compute influence (method: 'sum', 'eigen', 'betweenness', or 'all')
      3) detect communities (Louvain or greedy)
      4) visualize 1-NN and MST (if corr_df provided)
      5) compute ARI against corr_pre_clusters (list or pd.Series)
    Inputs:
      - corr_df: pandas DataFrame of correlations (index & columns are node names)
      - corr_pre_clusters: list/array of true cluster labels aligned with corr_df.index OR pd.Series indexed by node names
      - seed: random seed for layout
      - visualize: bool, whether to call visualize_network
    Returns:
      dict with keys: G_slim, influence, partition, modularity, ari
    """
    # Build network
    G_slim = build_slim_network(corr_df)

    # Compute influence
    influence = compute_influence(G_slim)

    # Community detection
    partition, modularity_score = compute_louvain(G_slim)

    # Visualization
    if visualize:
        visualize_network(G_slim, influence, partition, seed)

    # Validation (ARI)
    ari_score = validate_clusters(G_slim, partition, corr_pre_clusters, txt)

    # Print summary
    print(f"Modularity Score: {modularity_score:.3f}")
    print(f"Adjusted Rand Index (ARI): {ari_score:.3f}")

    return {
        "G_slim": G_slim,
        "influence": influence,
        "partition": partition,
        "modularity": modularity_score,
        "ari": ari_score
    }
