#!/usr/bin/env python3
"""
Desafio 2: Segmentacion Estructural mediante Clustering de Series Temporales
Script 2: Evaluacion completa de los 3 metodos de clustering

Calcula metricas cuantitativas (ARI, NMI, silhouette, modularidad)
y genera graficos de evaluacion comparativa.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')


def compute_silhouette_dtw(data, labels):
    """Silhouette score usando distancia DTW como metrica."""
    try:
        from tslearn.metrics import dtw
    except ImportError:
        print("[WARN] tslearn no disponible, usando silhouette con distancia euclidea")
        return silhouette_score(data, labels)

    n = data.shape[0]
    if len(np.unique(labels)) < 2:
        return 0.0

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw(data[i], data[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return silhouette_score(dist_matrix, labels, metric='precomputed')


def compute_intra_inter_distances(data, labels, metric='dtw'):
    """Calcula distancias intra-cluster e inter-cluster promedio."""
    try:
        from tslearn.metrics import dtw
    except ImportError:
        print("[WARN] tslearn no disponible, usando distancia euclidea")
        from scipy.spatial.distance import euclidean
        def dtw(x, y): return np.linalg.norm(x - y)

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    intra_dists = {}
    inter_dists = {}

    for lab in unique_labels:
        mask = labels == lab
        idx = np.where(mask)[0]
        if len(idx) < 2:
            intra_dists[lab] = 0.0
        else:
            dists = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    dists.append(dtw(data[idx[i]], data[idx[j]]))
            intra_dists[lab] = np.mean(dists) if dists else 0.0

        # Distancias inter-cluster: promedio hacia otros clusters
        other_idx = np.where(~mask)[0]
        if len(other_idx) > 0 and len(idx) > 0:
            dists = []
            for i in idx:
                for j in other_idx:
                    dists.append(dtw(data[i], data[j]))
            inter_dists[lab] = np.mean(dists) if dists else 0.0
        else:
            inter_dists[lab] = 0.0

    return intra_dists, inter_dists


def plot_metric_comparison(results_dict, save_path="metric_comparison.png"):
    """Genera grafico de barras comparativo de metricas."""
    methods = list(results_dict.keys())
    metrics_names = ['ARI', 'NMI', 'Silhouette', 'Modularidad']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx]
        values = [results_dict[m][metric_name] for m in methods]
        colors = ['#2196F3', '#FF9800', '#4CAF50']

        bars = ax.bar(methods, values, color=colors, edgecolor='black',
                      linewidth=0.8, alpha=0.85)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Puntaje', fontsize=10)

        # Añadir valor sobre cada barra
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')

        ax.set_ylim(0, max(1.0, max(values) * 1.15))
        ax.tick_params(axis='x', rotation=15, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Comparacion de Metodos de Clustering de Series Temporales',
                 fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico de metricas guardado en: {save_path}")


def plot_intra_inter_distances(intra_dists_all, inter_dists_all, methods,
                               save_path="intra_inter_distances.png"):
    """Grafico de distancias intra-cluster e inter-cluster."""
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4.5))

    if n_methods == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        intra = intra_dists_all[method]
        inter = inter_dists_all[method]
        clusters = list(intra.keys())

        x = np.arange(len(clusters))
        width = 0.35

        bars1 = ax.bar(x - width/2, [intra[c] for c in clusters], width,
                       label='Intra-cluster', color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, [inter[c] for c in clusters], width,
                       label='Inter-cluster', color='#FF5722', alpha=0.8)

        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('Distancia DTW promedio', fontsize=10)
        ax.set_title(f'{method}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'C{c}' for c in clusters], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Distancias Intra-cluster vs Inter-cluster (DTW)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico de distancias guardado en: {save_path}")


def plot_confusion_matrix_ari(ari_values, methods, save_path="ari_heatmap.png"):
    """Heatmap comparativo de ARI entre pares de metodos."""
    n = len(methods)
    matrix = np.ones((n, n))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i != j and (m1, m2) in ari_values:
                matrix[i, j] = ari_values[(m1, m2)]
            elif i != j and (m2, m1) in ari_values:
                matrix[i, j] = ari_values[(m2, m1)]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=methods, yticklabels=methods,
                vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title('Acuerdo entre Metodos (Adjusted Rand Index)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Heatmap ARI guardado en: {save_path}")


def plot_cluster_size_distribution(all_labels, methods, save_path="cluster_sizes.png"):
    """Grafico de distribucion de tamanos de cluster."""
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))

    if len(methods) == 1:
        axes = [axes]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        labels = all_labels[method]
        unique, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))

        bars = ax.bar(unique, counts, color=colors, edgecolor='black',
                      linewidth=0.8)
        ax.set_xlabel('Cluster', fontsize=10)
        ax.set_ylabel('# Series', fontsize=10)
        ax.set_title(f'{method}', fontsize=11, fontweight='bold')
        ax.set_xticks(unique)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=9,
                    fontweight='bold')

        ax.set_ylim(0, max(counts) * 1.2)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Distribucion de Tamanos de Cluster',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico de tamanos guardado en: {save_path}")


def evaluate_all(data, labels_true, labels_dtw, labels_kshape, labels_louvain,
                 modularity_louvain):
    """Evalua todos los metodos y retorna diccionario de metricas."""
    methods = {
        'DTW+k-Means': labels_dtw,
        'k-Shape': labels_kshape,
        'Louvain': labels_louvain
    }

    results = {}
    all_labels_dict = {}
    ari_pairs = {}

    for method_name, labels in methods.items():
        # ARI vs ground truth
        ari = adjusted_rand_score(labels_true, labels)
        # NMI vs ground truth
        nmi = normalized_mutual_info_score(labels_true, labels)
        # Silhouette con DTW
        sil = compute_silhouette_dtw(data, labels)

        # Modularidad (solo Louvain tiene sentido)
        mod = modularity_louvain if method_name == 'Louvain' else 0.0

        results[method_name] = {
            'ARI': ari,
            'NMI': nmi,
            'Silhouette': sil,
            'Modularidad': mod
        }
        all_labels_dict[method_name] = labels

    # ARI entre pares de metodos
    methods_list = list(methods.keys())
    for i in range(len(methods_list)):
        for j in range(i + 1, len(methods_list)):
            m1, m2 = methods_list[i], methods_list[j]
            ari = adjusted_rand_score(all_labels_dict[m1], all_labels_dict[m2])
            ari_pairs[(m1, m2)] = ari
            print(f"  ARI({m1} vs {m2}) = {ari:.4f}")

    return results, all_labels_dict, ari_pairs


def plot_all_metrics(results, all_labels_dict, ari_pairs, data):
    """Genera todos los graficos de evaluacion."""
    methods = list(results.keys())

    # 1. Comparacion de metricas
    plot_metric_comparison(results)

    # 2. ARI heatmap entre metodos
    plot_confusion_matrix_ari(ari_pairs, methods)

    # 3. Distribucion de tamanos
    plot_cluster_size_distribution(all_labels_dict, methods)

    # 4. Distancias intra/inter
    intra_dists_all = {}
    inter_dists_all = {}
    for method in methods:
        labels = all_labels_dict[method]
        intra, inter = compute_intra_inter_distances(data, labels)
        intra_dists_all[method] = intra
        inter_dists_all[method] = inter

    plot_intra_inter_distances(intra_dists_all, inter_dists_all, methods)


def main(data, labels_true, labels_dtw, labels_kshape, labels_louvain,
         modularity_louvain):
    """Funcion principal de evaluacion."""
    print("=" * 70)
    print(" EVALUACION COMPLETA DE CLUSTERING DE SERIES TEMPORALES")
    print("=" * 70)

    # Calcular metricas
    print("\n[1] Calculando metricas cuantitativas...")
    results, all_labels_dict, ari_pairs = evaluate_all(
        data, labels_true, labels_dtw, labels_kshape, labels_louvain,
        modularity_louvain
    )

    # Mostrar tabla resumen
    print("\n" + "-" * 70)
    print(f"{'Metodo':<18} {'ARI':<10} {'NMI':<10} {'Silhouette':<12} {'Modularidad':<12}")
    print("-" * 70)
    for method, metrics in results.items():
        print(f"{method:<18} {metrics['ARI']:<10.4f} {metrics['NMI']:<10.4f} "
              f"{metrics['Silhouette']:<12.4f} {metrics['Modularidad']:<12.4f}")
    print("-" * 70)

    # Generar graficos
    print("\n[2] Generando graficos de evaluacion...")
    plot_all_metrics(results, all_labels_dict, ari_pairs, data)

    # Conclusion
    print("\n[3] ANALISIS:")
    print("-" * 70)

    # Mejor metodo segun ARI
    best_ari_method = max(results, key=lambda m: results[m]['ARI'])
    best_ari_val = results[best_ari_method]['ARI']

    print(f"  - Mejor ARI vs Ground Truth: {best_ari_method} ({best_ari_val:.4f})")

    # Cohesión intra-cluster vs separacion
    for method in results:
        labels = all_labels_dict[method]
        intra, inter = compute_intra_inter_distances(data, labels)
        avg_intra = np.mean(list(intra.values()))
        avg_inter = np.mean(list(inter.values()))
        ratio = avg_intra / avg_inter if avg_inter > 0 else float('inf')
        print(f"  - {method:<15}: intra={avg_intra:.3f}, inter={avg_inter:.3f}, "
              f"ratio={ratio:.3f}")

    print("\n[OK] Evaluacion completada exitosamente.")
    return results


if __name__ == "__main__":
    # Ejemplo de uso: importar resultados del script 1
    import sys
    sys.path.insert(0, '.')
    
    # Si se ejecuta standalone, generar datos de ejemplo
    print("Ejecutando evaluacion...")
    print("Usa este script importando desde 1_cluster_timeseries.py")
