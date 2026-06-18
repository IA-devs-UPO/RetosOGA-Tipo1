import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split

# Directorio de modelos
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

# 2. Definir Pérdida Personalizada Log-Cosh
# La pérdida es L(y, y_pred) = log(cosh(y_pred - y))
# Gradiente: tanh(y_pred - y)
# Hessiano: 1 - tanh^2(y_pred - y)
def log_cosh_objective(y_true, y_pred):
    residual = y_pred - y_true
    grad = np.tanh(residual)
    hess = 1.0 - grad**2
    return grad, hess

def main():
    print("--- Desafío 1: Entrenando Modelos de Boosting con diferentes Loss ---")
    
    # 1. Generar Dataset Sintético con Outliers
    np.random.seed(42)
    X = np.linspace(0, 10, 500).reshape(-1, 1)
    
    # Función verdadera: sin(x) + ruido gaussiano
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

    # Insertar outliers severos para simular anomalías que MAE/Huber/Log-Cosh deben manejar mejor que MSE
    outliers_idx = np.random.choice(len(X), size=20, replace=False)
    y[outliers_idx] += np.random.normal(0, 5, 20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Guardar dataset para la evaluación posterior
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(models_dir, "dataset.pkl"))
    print(f"Dataset sintético generado con {len(X)} muestras (20 outliers).")

    # 3. Entrenamiento de Modelos
    
    # Modelo A: MAE (L1) nativo de LightGBM
    # LightGBM implementa aproximaciones de cuantiles en las hojas para resolver H=0
    print("\nEntrenando modelo con MAE (regression_l1)...")
    model_mae = lgb.LGBMRegressor(objective='regression_l1', n_estimators=100, random_state=42)
    model_mae.fit(X_train, y_train)

    # Modelo B: Pérdida Huber nativa
    # Huber es diferenciable en el origen (cuadrática para errores pequeños, lineal para grandes)
    print("Entrenando modelo con pérdida Huber...")
    model_huber = lgb.LGBMRegressor(objective='huber', alpha=1.0, n_estimators=100, random_state=42)
    model_huber.fit(X_train, y_train)

    # Modelo C: Pérdida Log-Cosh personalizada
    # A diferencia de MAE, Log-Cosh tiene gradiente y hessiano definidos en todos los puntos
    print("Entrenando modelo con pérdida Log-Cosh (Objetivo personalizado)...")
    model_logcosh = lgb.LGBMRegressor(objective=log_cosh_objective, n_estimators=100, random_state=42)
    model_logcosh.fit(X_train, y_train)

    # 4. Guardar Modelos
    joblib.dump(model_mae, os.path.join(models_dir, "model_mae.pkl"))
    joblib.dump(model_huber, os.path.join(models_dir, "model_huber.pkl"))
    joblib.dump(model_logcosh, os.path.join(models_dir, "model_logcosh.pkl"))
    
    print("\nModelos entrenados y guardados exitosamente en scripts/models/.")

if __name__ == "__main__":
    main()
