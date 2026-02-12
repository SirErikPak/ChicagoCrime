from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clusters(X_normalized, labels, method_name, metric='correlation'):
    sil_score = silhouette_score(X_normalized, labels, metric=metric)
    ch_score = calinski_harabasz_score(X_normalized, labels)
    db_score = davies_bouldin_score(X_normalized, labels)
    
    print(f"--- {method_name} ---")
    print(f"Silhouette Score (Measures how well trends match): "
          f"{sil_score:.3f} (1=perfect, 0=overlapping, -1=wrong)")
    print(f"Calinski-Harabasz Index (Ratio of between-cluster vs within-cluster variance): "
          f"{ch_score:.3f}  (higher is better)")
    print(f"Davies-Bouldin Index (Average 'similarity' between clusters): "
          f"{db_score:.3f}  (lower is better)")
    print("\n")