#!/usr/bin/env python3
"""
Desafío 2: Segmentación Estructural mediante Clustering de Series Temporales
Script 1: Generación de datos sintéticos y clustering con DTW, k-Shape y Louvain

Genera 3 patrones (seno, coseno desfasado, señal cuadrada) y aplica:
  1. TimeSeriesKMeans con métrica DTW (y centroide DBA)
  2. k-Shape (métrica SBD)
  3. Louvain sobre grafo de afinidad DTW
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# ── Generación de datos sintéticos ────────────────────────────────────────────

def generate_synthetic_data(n_series_per_cluster=15, random_seed=42, length=100):
    """Genera series temporales sintéticas con 3 patrones distintos."""
    rng = np.random.RandomState(random_seed)
    t = np.linspace(0, 4 * np.pi, length)

    series_list = []
    labels_true = []

    # Patrón 1: Seno puro
    for i in range(n_series_per_cluster):
        phase = rng.uniform(-0.2, 0.2)
        noise = rng.normal(0, 0.1, size=length)
        s = np.sin(t + phase) + noise
        series_list.append(s)
        labels_true.append(0)

    # Patrón 2: Coseno desfasado (fase ≈ π/2 respecto al seno)
    for i in range(n_series_per_cluster):
        phase = rng.uniform(-0.2, 0.2)
        noise = rng.normal(0, 0.1, size=length)
        s = np.cos(t + phase) + noise
        series_list.append(s)
        labels_true.append(1)

    # Patrón 3: Señal cuadrada
    for i in range(n_series_per_cluster):
        noise = rng.normal(0, 0.12, size=length)
        base = np.sign(np.sin(t))
        s = base + noise
        series_list.append(s)
        labels_true.append(2)

    return np.array(series_list), np.array(labels_true), t


# ── Método 1: DTW + k-Means (TimeSeriesKMeans) ──────────────────────────────

def cluster_dtw_kmeans(data, n_clusters=3, random_seed=42):
    """Clustering usando TimeSeriesKMeans con métrica DTW."""
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.metrics import dtw
    except ImportError:
        raise ImportError("tslearn es necesario. Instala con: pip install tslearn")

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        max_iter=50,
        random_state=random_seed,
        n_init=3,
        verbose=0
    )
    labels = model.fit_predict(data)
    return labels, model


# ── Método 2: k-Shape ────────────────────────────────────────────────────────

def cluster_kshape(data, n_clusters=3, random_seed=42):
    """Clustering usando k-Shape de tslearn."""
    try:
        from tslearn.clustering import KShape
    except ImportError:
        raise ImportError("tslearn es necesario. Instala con: pip install tslearn")

    model = KShape(
        n_clusters=n_clusters,
        max_iter=50,
        random_state=random_seed,
        n_init=3,
        verbose=0
    )
    labels = model.fit_predict(data)
    return labels, model


# ── Método 3: Louvain sobre grafo de afinidad DTW ────────────────────────────

def build_dtw_affinity_graph(data, k_neighbors=5):
    """Construye grafo de afinidad basado en distancia DTW.
    
    La afinidad se define como: A[i,j] = exp(-DTW(i,j)² / (2*σ²))
    Solo se conservan las k aristas más fuertes por nodo para mantener
    un grafo disperso.
    """
    try:
        from tslearn.metrics import dtw
    except ImportError:
        raise ImportError("tslearn es necesario. Instala con: pip install tslearn")

    n = data.shape[0]
    # Calcular matriz de distancias DTW (triangular superior)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(data[i], data[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Convertir distancias a afinidades (similitud gaussiana)
    sigma = np.median(dist_matrix[dist_matrix > 0]) if np.any(dist_matrix > 0) else 1.0
    affinity = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(affinity, 0.0)  # sin self-loops

    # Dispersar: solo k vecinos más cercanos (mayor afinidad)
    affinity_sparse = np.zeros_like(affinity)
    for i in range(n):
        idx = np.argsort(affinity[i])[::-1][:k_neighbors]
        affinity_sparse[i, idx] = affinity[i, idx]

    return affinity_sparse, dist_matrix


def cluster_louvain(affinity_matrix, random_seed=42):
    """Detección de comunidades con Louvain sobre grafo de afinidad.
    
    Construye un grafo NetworkX a partir de la matriz de afinidad dispersa
    y aplica el algoritmo de Louvain (community.best_partition).
    """
    try:
        import networkx as nx
        import community as community_louvain
    except ImportError:
        raise ImportError(
            "networkx y python-louvain son necesarios. "
            "Instala con: pip install networkx python-louvain"
        )

    G = nx.Graph()
    n = affinity_matrix.shape[0]
    # Añadir nodos
    G.add_nodes_from(range(n))
    # Añadir aristas con peso = afinidad
    for i in range(n):
        for j in range(i + 1, n):
            if affinity_matrix[i, j] > 0:
                G.add_edge(i, j, weight=affinity_matrix[i, j])

    # Si el grafo está vacío o desconectado, asignar comunidades dummy
    if G.number_of_edges() == 0:
        return np.zeros(n, dtype=int), 0.0, G

    # Aplicar Louvain
    partition = community_louvain.best_partition(G, random_state=random_seed)
    labels = np.array([partition[i] for i in range(n)])

    # Calcular modularidad
    modularity = community_louvain.modularity(partition, G)

    return labels, modularity, G


def run_louvain_clustering(data, n_clusters_expected=3, k_neighbors=5, random_seed=42):
    """Wrapper completo: construye grafo, ejecuta Louvain, post-procesa etiquetas."""
    affinity, dist_matrix = build_dtw_affinity_graph(data, k_neighbors=k_neighbors)
    labels, modularity, G = cluster_louvain(affinity, random_seed=random_seed)

    # Re-mapear etiquetas a rango [0, n_comunas) para comparación limpia
    unique_labels = np.unique(labels)
    mapping = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels_mapped = np.array([mapping[l] for l in labels])

    return labels_mapped, modularity, G, affinity


# ── Visualización ─────────────────────────────────────────────────────────────

def plot_clustering_results(data, labels_true, labels_dtw, labels_kshape,
                            labels_louvain, t, save_path="clustering_comparison.png"):
    """Genera gráfico comparativo de los 3 métodos de clustering."""
    n_clusters = len(np.unique(labels_true))
    fig, axes = plt.subplots(4, n_clusters, figsize=(4 * n_clusters, 10),
                             sharex='col', sharey='row')

    method_names = [
        ("Ground Truth", labels_true),
        ("DTW + k-Means", labels_dtw),
        ("k-Shape", labels_kshape),
        ("Louvain", labels_louvain),
    ]

    for row, (name, labels) in enumerate(method_names):
        for col in range(n_clusters):
            ax = axes[row, col]
            mask = labels == col
            if np.any(mask):
                cluster_data = data[mask]
                for serie in cluster_data:
                    ax.plot(t, serie, alpha=0.4, linewidth=0.8)
                # Centroide (media)
                centroid = cluster_data.mean(axis=0)
                ax.plot(t, centroid, color='black', linewidth=2.5, linestyle='--',
                        label='Centroide')
            ax.set_title(f'{name} - Cluster {col}', fontsize=9)
            if row == 3:
                ax.set_xlabel('Tiempo', fontsize=8)
            if col == 0:
                ax.set_ylabel('Amplitud', fontsize=8)
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Gráfico guardado en: {save_path}")


# ── Flujo principal ───────────────────────────────────────────────────────────

def main(n_series_per_cluster=15, random_seed=42, length=100):
    print("=" * 70)
    print(" DESAFÍO 2: Clustering de Series Temporales")
    print("=" * 70)

    # 1. Generar datos
    print("\n[1] Generando datos sintéticos...")
    data, labels_true, t = generate_synthetic_data(
        n_series_per_cluster=n_series_per_cluster,
        random_seed=random_seed,
        length=length
    )
    n_total = data.shape[0]
    print(f"    → {n_total} series, {len(np.unique(labels_true))} clases reales, "
          f"longitud={length}")

    # 2. DTW + k-Means
    print("\n[2] Ejecutando DTW + k-Means (TimeSeriesKMeans)...")
    labels_dtw, model_dtw = cluster_dtw_kmeans(data, n_clusters=3,
                                                random_seed=random_seed)
    print(f"    → Clusters encontrados: {len(np.unique(labels_dtw))}")

    # 3. k-Shape
    print("\n[3] Ejecutando k-Shape...")
    labels_kshape, model_kshape = cluster_kshape(data, n_clusters=3,
                                                  random_seed=random_seed)
    print(f"    → Clusters encontrados: {len(np.unique(labels_kshape))}")

    # 4. Louvain
    print("\n[4] Ejecutando Louvain sobre grafo de afinidad DTW...")
    labels_louvain, modularity, G, affinity = run_louvain_clustering(
        data, n_clusters_expected=3, k_neighbors=8, random_seed=random_seed
    )
    print(f"    → Comunidades encontradas: {len(np.unique(labels_louvain))}")
    print(f"    → Modularidad: {modularity:.4f}")
    print(f"    → Aristas en grafo: {G.number_of_edges()}")

    # 5. Visualización
    print("\n[5] Generando gráfico comparativo...")
    plot_clustering_results(data, labels_true, labels_dtw, labels_kshape,
                            labels_louvain, t)

    # 6. Métricas de evaluación
    from sklearn.metrics import adjusted_rand_score

    ari_dtw = adjusted_rand_score(labels_true, labels_dtw)
    ari_kshape = adjusted_rand_score(labels_true, labels_kshape)
    ari_louvain = adjusted_rand_score(labels_true, labels_louvain)

    print("\n" + "=" * 70)
    print(" MÉTRICAS DE EVALUACIÓN (Adjusted Rand Index vs Ground Truth)")
    print("=" * 70)
    print(f"  DTW + k-Means  : ARI = {ari_dtw:.4f}")
    print(f"  k-Shape        : ARI = {ari_kshape:.4f}")
    print(f"  Louvain        : ARI = {ari_louvain:.4f}")
    print("=" * 70)

    # Devolver resultados para análisis posterior
    return {
        'data': data,
        'labels_true': labels_true,
        'labels_dtw': labels_dtw,
        'labels_kshape': labels_kshape,
        'labels_louvain': labels_louvain,
        'modularity': modularity,
        'ari_dtw': ari_dtw,
        'ari_kshape': ari_kshape,
        'ari_louvain': ari_louvain,
        't': t
    }


if __name__ == "__main__":
    results = main(n_series_per_cluster=15, random_seed=42)
