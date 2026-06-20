#!/usr/bin/env python3
"""
=============================================================================
Desafío 3: Predicción de Demanda Energética Horaria por Código Postal
Script 1: Preprocesado, feature engineering y entrenamiento (LightGBM)
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "modelos")
os.makedirs(MODEL_DIR, exist_ok=True)

CP_COLS = ["cp_41001", "cp_41003", "cp_41005", "cp_41010", "cp_41020"]
N_LAGS = [24, 48, 168]  # 1 día, 2 días, 1 semana
TARGET_HORIZON = 743  # horas en marzo 2023

# ============================================================================
# CARGA DE DATOS
# ============================================================================
def load_data():
    """Carga los 4 datasets y parsea fechas."""
    demanda = pd.read_csv(
        os.path.join(DATA_DIR, "demanda_energia_entrenamiento.csv"),
        parse_dates=["fechaHora"]
    )
    clima = pd.read_csv(
        os.path.join(DATA_DIR, "clima.csv"),
        parse_dates=["fechaHora"]
    )
    calendario = pd.read_csv(
        os.path.join(DATA_DIR, "calendario.csv"),
        parse_dates=["fechaHora"]
    )
    cp_desc = pd.read_csv(
        os.path.join(DATA_DIR, "cp_descripcion.csv")
    )
    return demanda, clima, calendario, cp_desc

# ============================================================================
# MERGE DE DATASETS (TRAIN)
# ============================================================================
def merge_datasets(demanda, clima, calendario):
    """Merge demanda + clima + calendario por fechaHora."""
    df = demanda.merge(clima, on="fechaHora", how="left")
    df = df.merge(calendario, on="fechaHora", how="left")
    df = df.sort_values("fechaHora").reset_index(drop=True)
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def extract_time_features(df):
    """Extrae features temporales del datetime."""
    df = df.copy()
    df["hora"] = df["fechaHora"].dt.hour
    df["dia"] = df["fechaHora"].dt.day
    df["dia_semana"] = df["fechaHora"].dt.dayofweek  # 0=lunes
    df["mes"] = df["fechaHora"].dt.month
    df["ano"] = df["fechaHora"].dt.year
    df["dia_ano"] = df["fechaHora"].dt.dayofyear
    df["fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["hora_punta"] = df["hora"].isin(range(8, 22)).astype(int)  # 8h-21h
    # Features cíclicas para hora y día_semana
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    # Interacción hora × fin_de_semana
    df["hora_fin_semana"] = df["hora"] * df["fin_semana"]
    return df

def impute_missing(df):
    """Imputa valores ausentes en clima."""
    df = df.copy()
    # Clima: forward-fill, luego backward-fill, luego mediana
    for col in ["humedad", "velocidadViento", "temperatura", "lluvia"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(df[col].median())
    # Demanda: interpolar linealmente
    for cp in CP_COLS:
        if cp in df.columns:
            df[cp] = df[cp].interpolate(method="linear", limit_direction="both")
            df[cp] = df[cp].bfill().ffill()  # por si quedan bordes
    return df

def add_lag_features(df, cp_cols, lags):
    """Añade lags de la variable objetivo para cada CP."""
    df = df.copy()
    for cp in cp_cols:
        for lag in lags:
            df[f"{cp}_lag_{lag}h"] = df[cp].shift(lag)
    return df

def add_rolling_features(df, cp_cols, windows=[6, 12, 24]):
    """Añade medias móviles de la variable objetivo."""
    df = df.copy()
    for cp in cp_cols:
        for w in windows:
            df[f"{cp}_rolling_mean_{w}h"] = df[cp].rolling(w, min_periods=1).mean()
            df[f"{cp}_rolling_std_{w}h"] = df[cp].rolling(w, min_periods=1).std().fillna(0)
    return df

def add_clima_lags(df, lags=[6, 12, 24]):
    """Añade lags de variables climáticas."""
    df = df.copy()
    for col in ["temperatura", "humedad", "lluvia", "velocidadViento"]:
        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df

# ============================================================================
# PREPARACIÓN DE FEATURES + TARGET
# ============================================================================
def prepare_features(df, cp_cols):
    """Construye matriz X (features) e Y (targets) para entrenamiento."""
    # Columnas base (excluir demanda y fecha)
    base_features = [
        "hora", "dia", "dia_semana", "mes", "ano", "dia_ano",
        "fin_semana", "hora_punta",
        "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
        "hora_fin_semana",
        "cest", "es_festivo_o_domingo",
        "lluvia", "temperatura", "humedad", "velocidadViento"
    ]
    # Añadir lags y rolling
    lag_cols = []
    for cp in cp_cols:
        for lag in [24, 48, 168]:
            lag_cols.append(f"{cp}_lag_{lag}h")
        for w in [6, 12, 24]:
            lag_cols.append(f"{cp}_rolling_mean_{w}h")
            lag_cols.append(f"{cp}_rolling_std_{w}h")
    for col in ["temperatura", "humedad", "lluvia", "velocidadViento"]:
        for lag in [6, 12, 24]:
            lag_cols.append(f"{col}_lag_{lag}h")

    all_features = base_features + [c for c in lag_cols if c in df.columns]
    # Solo filas sin NA en features
    df_clean = df[all_features + cp_cols].dropna()
    X = df_clean[all_features]
    Y = df_clean[cp_cols]
    return X, Y, all_features

# ============================================================================
# ENTRENAMIENTO POR CP CON OPTUNA + TIMESERIES CV
# ============================================================================
def train_model_cp(X, y_cp, cp_name, n_trials=30):
    """Entrena un LightGBM para un CP usando Optuna + TimeSeriesSplit."""
    tscv = TimeSeriesSplit(n_splits=3, test_size=670)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "verbosity": -1,
            "random_state": 42,
        }
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_cp.iloc[train_idx], y_cp.iloc[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="mape",
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            preds = model.predict(X_val)
            # sMAPE
            denom = (np.abs(y_val) + np.abs(preds)) / 2
            smape = np.mean(2 * np.abs(y_val - preds) / (np.abs(y_val) + np.abs(preds))) * 100
            scores.append(smape)
        return np.mean(scores)

    study = optuna.create_study(direction="minimize", study_name=f"lgb_{cp_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["verbosity"] = -1
    best_params["random_state"] = 42

    # Entrenar con todos los datos con los mejores hiperparámetros
    model = lgb.LGBMRegressor(**best_params, n_estimators=best_params["n_estimators"])
    model.fit(X, y_cp)

    return model, study.best_value, best_params

# ============================================================================
# MAIN: ENTRENAMIENTO COMPLETO
# ============================================================================
def main():
    print("=" * 70)
    print("DESAFÍO 3 — PREPROCESADO Y ENTRENAMIENTO (LightGBM + Optuna)")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1] Cargando datasets...")
    demanda, clima, calendario, cp_desc = load_data()
    print(f"  Demanda: {demanda.shape}")
    print(f"  Clima: {clima.shape}")
    print(f"  Calendario: {calendario.shape}")
    print(f"  CP desc: {cp_desc.shape}")

    # 2. Merge
    print("\n[2] Fusionando datasets...")
    df = merge_datasets(demanda, clima, calendario)
    print(f"  Dataset fusionado: {df.shape}")

    # 3. Feature engineering
    print("\n[3] Extrayendo features temporales...")
    df = extract_time_features(df)
    print(f"  Columnas tras time features: {len(df.columns)}")

    # 4. Imputación
    print("\n[4] Imputando valores ausentes...")
    df = impute_missing(df)
    print("  Nulos restantes:", df.isnull().sum().sum())

    # 5. Lags y rolling
    print("\n[5] Añadiendo lags y rolling features...")
    df = add_lag_features(df, CP_COLS, N_LAGS)
    df = add_rolling_features(df, CP_COLS)
    df = add_clima_lags(df)
    print(f"  Columnas totales: {len(df.columns)}")

    # 6. Preparar X, Y
    print("\n[6] Preparando matrices X/Y...")
    X, Y, feature_names = prepare_features(df, CP_COLS)
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  Features: {len(feature_names)}")

    # 7. Entrenar modelo para cada CP
    print("\n[7] Entrenando modelos por CP (Optuna + TimeSeries CV)...")
    models = {}
    results = []
    for cp in CP_COLS:
        print(f"\n  --- {cp} ---")
        model, best_smape, best_params = train_model_cp(X, Y[cp], cp, n_trials=30)
        models[cp] = model
        results.append({"cp": cp, "best_smape_val": round(best_smape, 3)})
        print(f"  ✓ Mejor sMAPE val: {best_smape:.3f}%")
        # Guardar modelo
        model_path = os.path.join(MODEL_DIR, f"lgb_{cp}.txt")
        model.booster_.save_model(model_path)
        print(f"  ✓ Modelo guardado: {model_path}")

    # 8. Guardar metadata
    print("\n[8] Guardando metadata...")
    meta = {
        "feature_names": feature_names,
        "cp_models": {cp: f"lgb_{cp}.txt" for cp in CP_COLS},
        "results": results,
        "overall_smape": round(np.mean([r["best_smape_val"] for r in results]), 3),
    }
    import json
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata guardada en {MODEL_DIR}/metadata.json")

    # 9. Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS (validación cruzada):")
    for r in results:
        print(f"  {r['cp']}: {r['best_smape_val']:.3f}%")
    print(f"  OVERALL: {meta['overall_smape']:.3f}%")
    print("=" * 70)
    print("¡Entrenamiento completado con éxito!")

if __name__ == "__main__":
    main()