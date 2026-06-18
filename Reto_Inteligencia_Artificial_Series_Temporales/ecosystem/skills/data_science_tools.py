import numpy as np

def evaluate_custom_loss(loss_name: str, y_true: list[float], y_pred: list[float]) -> float:
    """
    Simula la evaluación de una función de pérdida personalizada (MAE, Huber, Log-Cosh).
    Esta función es usada por los agentes para ilustrar conceptos de optimización GBDT.
    """
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    if loss_name.lower() == 'mae':
        return float(np.mean(np.abs(y_t - y_p)))
    elif loss_name.lower() == 'huber':
        delta = 1.0 # default delta
        error = np.abs(y_t - y_p)
        loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
        return float(np.mean(loss))
    elif loss_name.lower() == 'log-cosh':
        loss = np.log(np.cosh(y_p - y_t))
        return float(np.mean(loss))
    else:
        raise ValueError(f"Pérdida desconocida: {loss_name}")

def cluster_time_series(method: str, num_series: int = 5) -> dict:
    """
    Simula la ejecución de un algoritmo de clustering de series temporales.
    Devuelve métricas simuladas del desempeño del método seleccionado.
    """
    metrics = {}
    if method.lower() == 'dtw':
        metrics = {
            'method': 'Dynamic Time Warping (DTW)',
            'time_complexity': 'O(n*m)',
            'robustness_to_shift': 'High',
            'simulated_silhouette_score': 0.72
        }
    elif method.lower() == 'k-shape':
        metrics = {
            'method': 'k-Shape',
            'time_complexity': 'O(n log n)',
            'robustness_to_shift': 'High (Scale & Shift Invariant)',
            'simulated_silhouette_score': 0.78
        }
    elif method.lower() == 'louvain':
        metrics = {
            'method': 'Louvain Graph Segmentation',
            'time_complexity': 'O(V log V)',
            'robustness_to_shift': 'Depends on graph affinity metric',
            'simulated_modularity': 0.85
        }
    else:
        raise ValueError(f"Método desconocido: {method}")
        
    return metrics
