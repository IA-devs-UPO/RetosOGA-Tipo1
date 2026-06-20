"""Desafío 2: Evaluación detallada de métodos de clustering."""
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="tslearn")

from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import networkx as nx
import community as community_louvain
from sklearn.metrics import adjusted_rand_score, rand_score, silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_synthetic_data(seed: int = 42):
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


def evaluate_method(name: str, y_true: np.ndarray, y_pred: np.ndarray, extra: dict = None):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Adjusted Rand Index (vs ground truth): {adjusted_rand_score(y_true, y_pred):.4f}")
    print(f"  Rand Index (vs ground truth):           {rand_score(y_true, y_pred):.4f}")
    print(f"  Clusters asignados:                     {len(set(y_pred))}")
    for k, v in (extra or {}).items():
        print(f"  {k}: {v}")


def plot_centroids(centroids, method_name: str, save_path: str):
    plt.figure(figsize=(10, 4))
    for i, c in enumerate(centroids):
        plt.plot(c.ravel(), label=f'Cluster {i}', linewidth=2)
    plt.title(f'Centroides - {method_name}')
    plt.xlabel('Time')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("--- Desafío 2: Evaluación Detallada de Clustering ---")
    output_dir = os.path.dirname(__file__)

    X, y_true = generate_synthetic_data()
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    print(f"Dataset: {X.shape[0]} series x {X.shape[1]} timesteps, {len(set(y_true))} clusters reales")

    # --- DTW + k-Means ---
    km_dtw = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=50, random_state=42)
    y_dtw = km_dtw.fit_predict(X_scaled)
    dtw_dist = cdist_dtw(X_scaled)
    sil_dtw = silhouette_score(dtw_dist, y_dtw, metric='precomputed')
    evaluate_method("DTW + k-Means", y_true, y_dtw, {
        "Silhouette Score": f"{sil_dtw:.4f}",
        "Inercia": f"{km_dtw.inertia_:.4f}",
        "Iteraciones": km_dtw.n_iter_,
    })
    plot_centroids(km_dtw.cluster_centers_, "DTW + k-Means",
                   os.path.join(output_dir, "centroids_dtw.png"))

    # --- k-Shape ---
    ks = KShape(n_clusters=3, max_iter=50, random_state=42)
    y_kshape = ks.fit_predict(X_scaled)
    sbd_dist = cdist_dtw(X_scaled)  # approximate SBD via DTW for silhouette
    sil_kshape = silhouette_score(sbd_dist, y_kshape, metric='precomputed')
    evaluate_method("k-Shape", y_true, y_kshape, {
        "Silhouette Score": f"{sil_kshape:.4f}",
        "Inercia": f"{ks.inertia_:.4f}",
        "Iteraciones": ks.n_iter_,
    })
    plot_centroids(ks.cluster_centers_, "k-Shape",
                   os.path.join(output_dir, "centroids_kshape.png"))

    # --- Louvain ---
    dist_matrix = cdist_dtw(X_scaled)
    similarity = np.exp(-dist_matrix / dist_matrix.std())
    G = nx.from_numpy_array(similarity)
    partition = community_louvain.best_partition(G, resolution=1.0)
    y_louvain = np.array([partition[i] for i in range(len(G))])
    mod = community_louvain.modularity(partition, G)
    evaluate_method("Louvain Graph Segmentation", y_true, y_louvain, {
        "Modularidad": f"{mod:.4f}",
        "Clusters detectados": len(set(partition.values())),
        "Nodos en grafo": G.number_of_nodes(),
        "Aristas en grafo": G.number_of_edges(),
    })

    # --- Tabla comparativa ---
    print(f"\n{'='*60}")
    print("  TABLA COMPARATIVA")
    print(f"{'='*60}")
    print(f"  {'Método':<25} {'ARI':<10} {'Clusters':<10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for name, y_pred in [("DTW + k-Means", y_dtw), ("k-Shape", y_kshape), ("Louvain", y_louvain)]:
        ari = adjusted_rand_score(y_true, y_pred)
        nc = len(set(y_pred))
        print(f"  {name:<25} {ari:<10.4f} {nc:<10}")

    print(f"\nGráficos de centroides guardados en: {output_dir}/")


if __name__ == "__main__":
    main()
