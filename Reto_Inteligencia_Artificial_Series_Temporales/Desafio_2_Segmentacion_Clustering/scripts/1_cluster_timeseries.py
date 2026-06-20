"""Desafío 2: Clustering de series temporales con DTW, k-Shape y Louvain."""
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tslearn")

from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_synthetic_data(seed: int = 42):
    """Genera 3 clusters: seno, coseno (desfasado), señal cuadrada."""
    np.random.seed(seed)
    n, sz = 30, 50
    X, y = [], []

    for _ in range(n):
        x = np.sin(np.linspace(0, 2 * np.pi, sz)) + np.random.normal(0, 0.1, sz)
        X.append(x); y.append(0)
    for _ in range(n):
        shift = np.random.uniform(np.pi / 2, np.pi)
        x = np.sin(np.linspace(0, 2 * np.pi, sz) + shift) + np.random.normal(0, 0.1, sz)
        X.append(x); y.append(1)
    for _ in range(n):
        x = np.sign(np.sin(np.linspace(0, 3 * np.pi, sz))) + np.random.normal(0, 0.2, sz)
        X.append(x); y.append(2)

    return np.array(X), np.array(y)


def silhouette_score(dist_matrix, labels):
    from sklearn.metrics import silhouette_score as sk_silhouette
    n_unique = len(set(labels))
    if n_unique <= 1 or n_unique >= len(labels):
        return 0.0
    return sk_silhouette(dist_matrix, labels, metric='precomputed')


def main():
    print("--- Desafío 2: Clustering de Series Temporales ---")
    output_dir = os.path.dirname(__file__)

    X, y_true = generate_synthetic_data()
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    print(f"Dataset: {X.shape[0]} series, {X.shape[1]} timesteps, {len(set(y_true))} clusters reales\n")

    results = {}

    # --- DTW + k-Means ---
    print("1. DTW + k-Means (TimeSeriesKMeans)...")
    km_dtw = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=50, random_state=42)
    y_dtw = km_dtw.fit_predict(X_scaled)
    dtw_dist = cdist_dtw(X_scaled)
    sil_dtw = silhouette_score(dtw_dist, y_dtw)
    ari_dtw = adjusted_rand_score(y_true, y_dtw)
    results['dtw_kmeans'] = {'method': 'DTW + k-Means', 'ari': round(ari_dtw, 4), 'silhouette': round(sil_dtw, 4), 'inertia': round(km_dtw.inertia_, 4)}
    print(f"   ARI={ari_dtw:.4f}, Silhouette={sil_dtw:.4f}")

    # --- k-Shape ---
    print("2. k-Shape...")
    ks = KShape(n_clusters=3, max_iter=50, random_state=42)
    y_kshape = ks.fit_predict(X_scaled)
    ari_kshape = adjusted_rand_score(y_true, y_kshape)
    results['kshape'] = {'method': 'k-Shape', 'ari': round(ari_kshape, 4), 'inertia': round(ks.inertia_, 4)}
    print(f"   ARI={ari_kshape:.4f}")

    # --- Louvain ---
    print("3. Louvain Graph Segmentation...")
    dist_matrix = cdist_dtw(X_scaled)
    similarity = np.exp(-dist_matrix / dist_matrix.std())
    G = nx.from_numpy_array(similarity)
    partition = community_louvain.best_partition(G, resolution=1.0)
    y_louvain = np.array([partition[i] for i in range(len(G))])
    mod = community_louvain.modularity(partition, G)
    ari_louvain = adjusted_rand_score(y_true, y_louvain)
    results['louvain'] = {'method': 'Louvain', 'ari': round(ari_louvain, 4), 'modularity': round(mod, 4), 'clusters': len(set(partition.values()))}
    print(f"   ARI={ari_louvain:.4f}, Modularidad={mod:.4f}, Clusters detectados={len(set(partition.values()))}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [
        f'DTW + k-Means (ARI={results["dtw_kmeans"]["ari"]})',
        f'k-Shape (ARI={results["kshape"]["ari"]})',
        f'Louvain (ARI={results["louvain"]["ari"]})',
    ]
    y_preds = [y_dtw, y_kshape, y_louvain]
    for ax, y_pred, title in zip(axes, y_preds, titles):
        for cluster_id in np.unique(y_pred):
            mask = y_pred == cluster_id
            for ts in X_scaled[mask]:
                ax.plot(ts.ravel(), alpha=0.5, linewidth=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Time')
        ax.set_ylabel('Scaled Value')
        ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, 'clustering_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGráfico guardado: {plot_path}")

    print("\n--- Resumen ---")
    for name, r in results.items():
        print(f"  {r['method']}: ARI={r['ari']}")


if __name__ == "__main__":
    main()
