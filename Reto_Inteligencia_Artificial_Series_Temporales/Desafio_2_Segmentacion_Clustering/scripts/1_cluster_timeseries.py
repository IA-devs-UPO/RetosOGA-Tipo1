import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from sklearn.metrics import adjusted_rand_score

# Tslearn imports
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Python-louvain import
import community as community_louvain

def generate_synthetic_data():
    """Genera 3 clusters de series temporales: Seno, Coseno (desfasado), Ruido Estructurado."""
    np.random.seed(42)
    n_ts_per_cluster = 30
    sz = 50
    X = []
    y = []

    # Cluster 0: Sine wave
    for _ in range(n_ts_per_cluster):
        x = np.sin(np.linspace(0, 2 * np.pi, sz)) + np.random.normal(0, 0.1, sz)
        X.append(x)
        y.append(0)

    # Cluster 1: Phase-shifted Sine wave (Cosine)
    # Misma forma morfológica, pero desfasada.
    for _ in range(n_ts_per_cluster):
        # Desfase aleatorio para complicarlo
        shift = np.random.uniform(np.pi/2, np.pi)
        x = np.sin(np.linspace(0, 2 * np.pi, sz) + shift) + np.random.normal(0, 0.1, sz)
        X.append(x)
        y.append(1)

    # Cluster 2: Señal cuadrada / pulsos
    for _ in range(n_ts_per_cluster):
        x = np.sign(np.sin(np.linspace(0, 3 * np.pi, sz))) + np.random.normal(0, 0.2, sz)
        X.append(x)
        y.append(2)

    X = np.array(X)[..., np.newaxis] # (90, 50, 1) format required by tslearn
    y = np.array(y)
    
    # KShape requiere series normalizadas a media 0 y varianza 1
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y

def main():
    print("--- Desafío 2: Segmentación de Series Temporales ---")
    
    # 1. Generar Datos
    X, y_true = generate_synthetic_data()
    print(f"Dataset sintético generado: {X.shape[0]} series de longitud {X.shape[1]}.")
    
    n_clusters = 3
    
    # 2. KMeans Euclidiano
    print("1. Ejecutando TimeSeriesKMeans (Euclidean)...")
    km_eucl = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=42)
    y_pred_eucl = km_eucl.fit_predict(X)
    ari_eucl = adjusted_rand_score(y_true, y_pred_eucl)
    
    # 3. KMeans DTW
    print("2. Ejecutando TimeSeriesKMeans (DTW + DBA)...")
    km_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter_barycenter=10, random_state=42)
    y_pred_dtw = km_dtw.fit_predict(X)
    ari_dtw = adjusted_rand_score(y_true, y_pred_dtw)
    
    # 4. k-Shape
    print("3. Ejecutando k-Shape (Shape-based Distance)...")
    kshape = KShape(n_clusters=n_clusters, random_state=42)
    y_pred_kshape = kshape.fit_predict(X)
    ari_kshape = adjusted_rand_score(y_true, y_pred_kshape)
    
    # 5. Grafo Semántico y Louvain
    print("4. Construyendo Grafo de Afinidad y ejecutando Louvain...")
    # Calcular matriz de distancias DTW
    dist_matrix = cdist_dtw(X)
    
    # Convertir distancias a matriz de afinidad (RBF Kernel)
    gamma = 0.05
    affinity_matrix = np.exp(-gamma * dist_matrix**2)
    
    # Construir el grafo limitando aristas débiles para mantener topología significativa
    threshold = np.percentile(affinity_matrix, 85) # Top 15% de conexiones
    G = nx.Graph()
    n_nodes = X.shape[0]
    G.add_nodes_from(range(n_nodes))
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if affinity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=affinity_matrix[i, j])
                
    # Detección de comunidades con Louvain
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    y_pred_louvain = [partition[i] for i in range(n_nodes)]
    ari_louvain = adjusted_rand_score(y_true, y_pred_louvain)
    
    # Mostrar resultados
    print("\n--- Resultados (Adjusted Rand Index) ---")
    print("Mide la similitud entre el clustering y el Ground Truth (1.0 = Perfecto).")
    print(f"Euclidean K-Means: {ari_eucl:.4f}")
    print(f"DTW K-Means:       {ari_dtw:.4f}")
    print(f"k-Shape:           {ari_kshape:.4f}")
    print(f"Grafo + Louvain:   {ari_louvain:.4f}")
    
    # 6. Guardar Gráfico Comparativo de los Baricentros DTW y clusters
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    # Plot K-Means DTW
    plt.subplot(1, 3, 1)
    for i in range(n_clusters):
        plt.plot(km_dtw.cluster_centers_[i].ravel(), label=f'Cluster {i}')
    plt.title(f'Centroides DTW (DBA)\nARI: {ari_dtw:.2f}')
    plt.legend()
    
    # Plot k-Shape
    plt.subplot(1, 3, 2)
    for i in range(n_clusters):
        plt.plot(kshape.cluster_centers_[i].ravel(), label=f'Cluster {i}')
    plt.title(f'Centroides k-Shape\nARI: {ari_kshape:.2f}')
    
    # Plot Louvain Graph Communities
    plt.subplot(1, 3, 3)
    pos = nx.spring_layout(G, seed=42)
    cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    plt.title(f'Comunidades Louvain\nARI: {ari_louvain:.2f}')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "clustering_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nVisualización guardada en {plot_path}")

if __name__ == "__main__":
    main()
