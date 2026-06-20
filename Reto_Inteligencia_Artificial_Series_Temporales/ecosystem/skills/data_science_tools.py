import numpy as np
import os


def evaluate_custom_loss(loss_name: str, y_true: list[float], y_pred: list[float]) -> float:
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    if loss_name.lower() == 'mae':
        return float(np.mean(np.abs(y_t - y_p)))
    elif loss_name.lower() == 'huber':
        delta = 1.0
        error = np.abs(y_t - y_p)
        loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
        return float(np.mean(loss))
    elif loss_name.lower() == 'log-cosh':
        loss = np.log(np.cosh(y_p - y_t))
        return float(np.mean(loss))
    else:
        raise ValueError(f"Pérdida desconocida: {loss_name}")


def run_clustering_analysis(n_series_per_cluster: int = 30, random_seed: int = 42) -> dict:
    """
    Ejecuta análisis completo de clustering de series temporales usando
    DTW, k-Shape y Louvain. Genera datos sintéticos y devuelve métricas comparativas.
    """
    from tslearn.clustering import TimeSeriesKMeans, KShape
    from tslearn.metrics import cdist_dtw
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance
    import networkx as nx
    import community as community_louvain
    from sklearn.metrics import adjusted_rand_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    np.random.seed(random_seed)
    n = n_series_per_cluster
    sz = 50

    X, y_true = [], []
    for _ in range(n):
        x = np.sin(np.linspace(0, 2 * np.pi, sz)) + np.random.normal(0, 0.1, sz)
        X.append(x)
        y_true.append(0)
    for _ in range(n):
        shift = np.random.uniform(np.pi / 2, np.pi)
        x = np.sin(np.linspace(0, 2 * np.pi, sz) + shift) + np.random.normal(0, 0.1, sz)
        X.append(x)
        y_true.append(1)
    for _ in range(n):
        x = np.sign(np.sin(np.linspace(0, 3 * np.pi, sz))) + np.random.normal(0, 0.2, sz)
        X.append(x)
        y_true.append(2)

    X = np.array(X)
    y_true = np.array(y_true)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)

    results = {}
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(plot_dir, exist_ok=True)

    # --- DTW + k-Means ---
    km_dtw = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=50, random_state=random_seed)
    y_dtw = km_dtw.fit_predict(X_scaled)
    dtw_dist = cdist_dtw(X_scaled)
    silhouette_dtw = _silhouette_score(dtw_dist, y_dtw)
    results['dtw_kmeans'] = {
        'method': 'DTW + k-Means (TimeSeriesKMeans)',
        'ari_vs_ground_truth': float(round(adjusted_rand_score(y_true, y_dtw), 4)),
        'silhouette_score': float(round(silhouette_dtw, 4)),
        'n_iterations': int(km_dtw.n_iter_),
        'inertia': float(round(km_dtw.inertia_, 4)),
    }

    # --- k-Shape ---
    ks = KShape(n_clusters=3, max_iter=50, random_state=random_seed)
    y_kshape = ks.fit_predict(X_scaled)
    results['kshape'] = {
        'method': 'k-Shape',
        'ari_vs_ground_truth': float(round(adjusted_rand_score(y_true, y_kshape), 4)),
        'n_iterations': int(ks.n_iter_),
        'inertia': float(round(ks.inertia_, 4)),
    }

    # --- Louvain Graph Segmentation ---
    dist_matrix = cdist_dtw(X_scaled)
    similarity = np.exp(-dist_matrix / dist_matrix.std())
    G = nx.from_numpy_array(similarity)
    partition = community_louvain.best_partition(G, resolution=1.0)
    y_louvain = np.array([partition[i] for i in range(len(G))])
    n_clusters_louvain = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G)
    results['louvain'] = {
        'method': 'Louvain Graph Segmentation',
        'n_clusters_detected': int(n_clusters_louvain),
        'modularity': float(round(modularity, 4)),
        'ari_vs_ground_truth': float(round(adjusted_rand_score(y_true, y_louvain), 4)),
    }

    # --- Comparison plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [f'DTW + k-Means (ARI={results["dtw_kmeans"]["ari_vs_ground_truth"]})',
              f'k-Shape (ARI={results["kshape"]["ari_vs_ground_truth"]})',
              f'Louvain (ARI={results["louvain"]["ari_vs_ground_truth"]})']
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

    plot_path = os.path.join(plot_dir, 'clustering_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'results': results,
        'plot_path': plot_path,
        'dataset_shape': list(X_scaled.shape),
        'n_clusters_ground_truth': 3,
    }


def _silhouette_score(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score as sk_silhouette
    n_unique = len(set(labels))
    if n_unique <= 1 or n_unique >= len(labels):
        return 0.0
    return float(sk_silhouette(dist_matrix, labels, metric='precomputed'))


def save_script(filename: str, content: str, challenge_dir: str = "Desafio_2_Segmentacion_Clustering") -> str:
    """
    Guarda un script Python en el directorio scripts/ del desafío correspondiente.
    ÚSALA para crear los archivos de solución del reto.

    Args:
        filename: Nombre del archivo (ej. '1_cluster_timeseries.py')
        content: Código Python completo del script
        challenge_dir: Directorio del desafío (default: Desafio_2_Segmentacion_Clustering)

    Returns:
        Ruta absoluta del archivo guardado
    """
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', challenge_dir))
    scripts_dir = os.path.join(base, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    filepath = os.path.join(scripts_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    return filepath
