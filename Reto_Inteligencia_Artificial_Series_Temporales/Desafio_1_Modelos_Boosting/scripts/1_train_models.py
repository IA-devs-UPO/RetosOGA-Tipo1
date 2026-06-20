"""Desafío 1: Entrenar modelos GBDT con diferentes funciones de pérdida."""
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.model_selection import train_test_split


def log_cosh_objective(y_true, y_pred):
    residual = y_pred - y_true
    grad = np.tanh(residual)
    hess = 1.0 - grad**2
    return grad, hess


def huber_objective(y_true, y_pred):
    delta = 1.0
    residual = y_pred - y_true
    abs_res = np.abs(residual)
    grad = np.where(abs_res <= delta, residual, delta * np.sign(residual))
    hess = np.where(abs_res <= delta, 1.0, 0.0)
    return grad, hess


def main():
    print("--- Desafío 1: Entrenando Modelos de Boosting ---")
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    np.random.seed(42)
    X = np.linspace(0, 10, 500).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])
    outliers_idx = np.random.choice(len(X), size=20, replace=False)
    y[outliers_idx] += np.random.normal(0, 5, 20)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    joblib.dump((X_train, X_test, y_train, y_test), os.path.join(models_dir, "dataset.pkl"))
    print(f"Dataset: {len(X)} muestras (20 outliers).")

    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "verbose": -1,
        "random_state": 42,
    }

    print("\nEntrenando LGBM con pérdida MAE (nativa)...")
    model_mae = lgb.LGBMRegressor(objective="mae", **params)
    model_mae.fit(X_train, y_train)
    joblib.dump(model_mae, os.path.join(models_dir, "model_mae.pkl"))
    print(f"  Train MAE: {np.mean(np.abs(y_train - model_mae.predict(X_train))):.4f}")

    print("\nEntrenando LGBM con pérdida Huber (nativa)...")
    model_huber = lgb.LGBMRegressor(objective="huber", **params)
    model_huber.fit(X_train, y_train)
    joblib.dump(model_huber, os.path.join(models_dir, "model_huber.pkl"))
    print(f"  Train MAE: {np.mean(np.abs(y_train - model_huber.predict(X_train))):.4f}")

    print("\nEntrenando LGBM con pérdida Log-Cosh (personalizada)...")
    model_logcosh = lgb.LGBMRegressor(objective=log_cosh_objective, **params)
    model_logcosh.fit(X_train, y_train)
    joblib.dump(model_logcosh, os.path.join(models_dir, "model_logcosh.pkl"))
    print(f"  Train MAE: {np.mean(np.abs(y_train - model_logcosh.predict(X_train))):.4f}")

    print("\n¡Modelos entrenados y guardados!")


if __name__ == "__main__":
    main()
