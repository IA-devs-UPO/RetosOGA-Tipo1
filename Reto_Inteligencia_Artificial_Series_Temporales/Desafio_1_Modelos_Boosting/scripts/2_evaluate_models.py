import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def log_cosh_objective(y_true, y_pred):
    residual = y_pred - y_true
    grad = np.tanh(residual)
    hess = 1.0 - grad**2
    return grad, hess

models_dir = os.path.join(os.path.dirname(__file__), "models")

def main():
    print("--- Evaluando Modelos de Boosting ---")
    
    # 1. Cargar dataset
    X_train, X_test, y_train, y_test = joblib.load(os.path.join(models_dir, "dataset.pkl"))
    
    # 2. Cargar modelos
    model_mae = joblib.load(os.path.join(models_dir, "model_mae.pkl"))
    model_huber = joblib.load(os.path.join(models_dir, "model_huber.pkl"))
    model_logcosh = joblib.load(os.path.join(models_dir, "model_logcosh.pkl"))
    
    # 3. Predicciones
    y_pred_mae = model_mae.predict(X_test)
    y_pred_huber = model_huber.predict(X_test)
    y_pred_logcosh = model_logcosh.predict(X_test)
    
    # 4. Métricas sobre el set de test
    # Evaluamos usando MAE puro para comprobar qué tan bien generalizan las aproximaciones
    mae_score = mean_absolute_error(y_test, y_pred_mae)
    huber_score = mean_absolute_error(y_test, y_pred_huber)
    logcosh_score = mean_absolute_error(y_test, y_pred_logcosh)
    
    print("\nResultados en conjunto de Prueba (Métrica de evaluación: MAE):")
    print(f"Test MAE (Modelo MAE Nativo): {mae_score:.4f}")
    print(f"Test MAE (Modelo Huber):      {huber_score:.4f}")
    print(f"Test MAE (Modelo Log-Cosh):   {logcosh_score:.4f}")
    
    # 5. Graficar reconstrucción de la señal
    # Generamos un dominio uniforme para ver cómo ajusta la curva cada modelo
    X_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    y_true_plot = np.sin(X_plot).ravel()
    
    plt.figure(figsize=(12, 6))
    plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Datos de Entrenamiento (con Outliers)', s=15)
    plt.plot(X_plot, y_true_plot, 'k--', label='Señal Real (sin(x))', linewidth=2)
    
    plt.plot(X_plot, model_mae.predict(X_plot), color='blue', label=f'LGBM MAE (Test MAE: {mae_score:.2f})', alpha=0.8, linewidth=2)
    plt.plot(X_plot, model_huber.predict(X_plot), color='red', label=f'LGBM Huber (Test MAE: {huber_score:.2f})', alpha=0.8, linewidth=2)
    plt.plot(X_plot, model_logcosh.predict(X_plot), color='green', label=f'LGBM Log-Cosh (Test MAE: {logcosh_score:.2f})', alpha=0.8, linewidth=2)
    
    plt.title("Comparativa: MAE vs Huber vs Log-Cosh ante Outliers (Desafío 1)", fontsize=14)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(os.path.dirname(__file__), "comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico comparativo exportado en: {plot_path}")

if __name__ == "__main__":
    main()
