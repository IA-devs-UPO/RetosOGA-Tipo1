"""Desafío 1: Evaluar y comparar modelos de boosting."""
import os
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def main():
    print("--- Evaluando Modelos de Boosting ---")
    models_dir = os.path.join(os.path.dirname(__file__), "models")

    X_train, X_test, y_train, y_test = joblib.load(os.path.join(models_dir, "dataset.pkl"))

    model_mae = joblib.load(os.path.join(models_dir, "model_mae.pkl"))
    model_huber = joblib.load(os.path.join(models_dir, "model_huber.pkl"))
    model_logcosh = joblib.load(os.path.join(models_dir, "model_logcosh.pkl"))

    y_pred_mae = model_mae.predict(X_test)
    y_pred_huber = model_huber.predict(X_test)
    y_pred_logcosh = model_logcosh.predict(X_test)

    mae_score = mean_absolute_error(y_test, y_pred_mae)
    huber_score = mean_absolute_error(y_test, y_pred_huber)
    logcosh_score = mean_absolute_error(y_test, y_pred_logcosh)

    print("\nResultados en conjunto de Prueba (Métrica: MAE):")
    print(f"  MAE Nativo:  {mae_score:.4f}")
    print(f"  Huber:       {huber_score:.4f}")
    print(f"  Log-Cosh:    {logcosh_score:.4f}")

    X_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    y_true_plot = np.sin(X_plot).ravel()

    plt.figure(figsize=(12, 6))
    plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Train (con outliers)', s=15)
    plt.plot(X_plot, y_true_plot, 'k--', label='Señal real sin(x)', linewidth=2)
    plt.plot(X_plot, model_mae.predict(X_plot), color='blue', label=f'MAE (Test MAE: {mae_score:.2f})', alpha=0.8, linewidth=2)
    plt.plot(X_plot, model_huber.predict(X_plot), color='red', label=f'Huber (Test MAE: {huber_score:.2f})', alpha=0.8, linewidth=2)
    plt.plot(X_plot, model_logcosh.predict(X_plot), color='green', label=f'Log-Cosh (Test MAE: {logcosh_score:.2f})', alpha=0.8, linewidth=2)

    plt.title("Comparativa: MAE vs Huber vs Log-Cosh ante Outliers")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(os.path.dirname(__file__), "comparison_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nGráfico guardado: {plot_path}")


if __name__ == "__main__":
    main()
