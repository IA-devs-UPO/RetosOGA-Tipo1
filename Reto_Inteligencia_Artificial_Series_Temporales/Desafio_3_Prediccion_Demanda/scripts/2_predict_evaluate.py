#!/usr/bin/env python3
"""
=============================================================================
Desafío 3: Predicción de Demanda Energética Horaria por Código Postal
Script 2: Predicción de Marzo 2023 y Evaluación con sMAPE
=============================================================================
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "modelos")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "resultados")
os.makedirs(RESULTS_DIR, exist_ok=True)

CP_COLS = ["cp_41001", "cp_41003", "cp_41005", "cp_41010", "cp_41020"]
N_LAGS = [24, 48, 168]

# ============================================================================
# CARGA DE DATOS (incluye marzo 2023 para predictores)
# ============================================================================
def load_prediction_data():
    """Carga clima y calendario de marzo 2023 para predecir."""
    clima = pd.read_csv(
        os.path.join(DATA_DIR, "clima.csv"),
        parse_dates=["fechaHora"]
    )
    calendario = pd.read_csv(
        os.path.join(DATA_DIR, "calendario.csv"),
        parse_dates=["fechaHora"]
    )
    demanda = pd.read_csv(
        os.path.join(DATA_DIR, "demanda_energia_entrenamiento.csv"),
        parse_dates=["fechaHora"]
    )
    return demanda, clima, calendario

def load_metadata():
    """Carga la metadata del entrenamiento."""
    with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as f:
        return json.load(f)

# ============================================================================
# FEATURE ENGINEERING (idéntico al de entrenamiento)
# ============================================================================
def extract_time_features(df):
    """Extrae features temporales del datetime."""
    df = df.copy()
    df["hora"] = df["fechaHora"].dt.hour
    df["dia"] = df["fechaHora"].dt.day
    df["dia_semana"] = df["fechaHora"].dt.dayofweek
    df["mes"] = df["fechaHora"].dt.month
    df["ano"] = df["fechaHora"].dt.year
    df["dia_ano"] = df["fechaHora"].dt.dayofyear
    df["fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["hora_punta"] = df["hora"].isin(range(8, 22)).astype(int)
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["hora_fin_semana"] = df["hora"] * df["fin_semana"]
    return df

def impute_missing(df):
    """Imputa valores ausentes en clima."""
    df = df.copy()
    for col in ["humedad", "velocidadViento", "temperatura", "lluvia"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(df[col].median())
    for cp in CP_COLS:
        if cp in df.columns:
            df[cp] = df[cp].interpolate(method="linear", limit_direction="both")
            df[cp] = df[cp].bfill().ffill()
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
# CONSTRUCCIÓN DEL DATASET PARA PREDICCIÓN (Marzo 2023)
# ============================================================================
def build_prediction_dataset(demanda, clima, calendario, feature_names):
    """
    Construye el dataset completo (histórico + marzo 2023) con features,
    para poder generar los lags necesarios antes de predecir.
    """
    # Merge histórico + clima + calendario
    df_pred = demanda.merge(clima, on="fechaHora", how="left")
    df_pred = df_pred.merge(calendario, on="fechaHora", how="left")
    df_pred = df_pred.sort_values("fechaHora").reset_index(drop=True)

    # Feature engineering
    df_pred = extract_time_features(df_pred)
    df_pred = impute_missing(df_pred)

    # Lags y rolling (necesitan datos históricos completos)
    df_pred = add_lag_features(df_pred, CP_COLS, N_LAGS)
    df_pred = add_rolling_features(df_pred, CP_COLS)
    df_pred = add_clima_lags(df_pred)

    # Filtrar solo marzo 2023
    mask_marzo = (df_pred["fechaHora"] >= "2023-03-01") & (df_pred["fechaHora"] < "2023-04-01")
    df_marzo = df_pred[mask_marzo].reset_index(drop=True)

    # Verificar que tenemos todas las features necesarias
    missing_features = [f for f in feature_names if f not in df_marzo.columns]
    if missing_features:
        print(f"  ⚠ Features faltantes: {missing_features}")
        # Añadirlas con NaN
        for f in missing_features:
            df_marzo[f] = np.nan

    X_pred = df_marzo[feature_names]
    return X_pred, df_marzo

# ============================================================================
# MÉTRICA sMAPE
# ============================================================================
def smape(y_true, y_pred):
    """Calcula sMAPE entre arrays real y predicho."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Evitar división por cero donde ambos sean 0
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return 100.0 * np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask])

# ============================================================================
# PREDICCIÓN
# ============================================================================
def predict_all_cps(X_pred, df_marzo, metadata):
    """Predice la demanda para cada CP usando los modelos guardados."""
    predictions = {}
    models_loaded = {}

    for cp in CP_COLS:
        model_path = os.path.join(MODEL_DIR, metadata["cp_models"][cp])
        model = lgb.Booster(model_file=model_path)
        models_loaded[cp] = model
        preds = model.predict(X_pred)
        predictions[cp] = preds
        print(f"  ✓ {cp}: {len(preds)} predicciones generadas")

    return predictions, models_loaded

# ============================================================================
# EVALUACIÓN VS REALES
# ============================================================================
def evaluate_predictions(predictions, df_marzo):
    """Evalúa las predicciones contra los valores reales usando sMAPE."""
    print("\n" + "-" * 50)
    print("EVALUACIÓN sMAPE vs VALORES REALES")
    print("-" * 50)

    results = []
    all_true = []
    all_pred = []

    for cp in CP_COLS:
        y_true = df_marzo[cp].values
        y_pred = predictions[cp]
        s = smape(y_true, y_pred)
        results.append({"cp": cp, "sMAPE": round(s, 3)})
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())
        print(f"  {cp}: {s:.3f}%")

    overall = smape(np.array(all_true), np.array(all_pred))
    print(f"  OVERALL sMAPE: {overall:.3f}%")
    print("-" * 50)

    return results, overall

# ============================================================================
# GUARDAR PREDICCIONES (formato wide)
# ============================================================================
def save_predictions(predictions, df_marzo):
    """Guarda las predicciones en formato wide (como el CSV de entrada)."""
    fecha_hora = df_marzo["fechaHora"].values
    pred_df = pd.DataFrame({"fechaHora": fecha_hora})
    for cp in CP_COLS:
        pred_df[cp] = predictions[cp]

    output_path = os.path.join(RESULTS_DIR, "predicciones_marzo_2023.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Predicciones guardadas: {output_path}")
    print(f"  Forma: {pred_df.shape}")
    return pred_df

def save_evaluation_results(results, overall):
    """Guarda los resultados de evaluación."""
    output_path = os.path.join(RESULTS_DIR, "evaluacion_smape.json")
    eval_data = {
        "resultados_por_cp": results,
        "overall_smape": round(overall, 3),
        "total_predicciones": 743 * 5
    }
    with open(output_path, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"  ✓ Evaluación guardada: {output_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("DESAFÍO 3 — PREDICCIÓN MARZO 2023 Y EVALUACIÓN")
    print("=" * 70)

    # 1. Cargar datos y metadata
    print("\n[1] Cargando datos y metadatos...")
    demanda, clima, calendario = load_prediction_data()
    metadata = load_metadata()
    feature_names = metadata["feature_names"]
    print(f"  Features esperadas: {len(feature_names)}")

    # 2. Construir dataset de predicción
    print("\n[2] Construyendo dataset de predicción (Marzo 2023)...")
    X_pred, df_marzo = build_prediction_dataset(demanda, clima, calendario, feature_names)
    print(f"  X_pred: {X_pred.shape}")
    print(f"  df_marzo (filas): {len(df_marzo)}")

    # Verificar que no haya nulos en X_pred
    nulos = X_pred.isnull().sum().sum()
    if nulos > 0:
        print(f"  ⚠ {nulos} valores nulos en features — imputando...")
        X_pred = X_pred.ffill().bfill().fillna(0)
        nulos_post = X_pred.isnull().sum().sum()
        print(f"  Nulos tras imputación: {nulos_post}")

    # 3. Predecir
    print("\n[3] Generando predicciones para cada CP...")
    predictions, models_loaded = predict_all_cps(X_pred, df_marzo, metadata)

    # 4. Evaluar
    print("\n[4] Evaluando contra valores reales...")
    results, overall = evaluate_predictions(predictions, df_marzo)

    # 5. Guardar
    print("\n[5] Guardando resultados...")
    pred_df = save_predictions(predictions, df_marzo)
    save_evaluation_results(results, overall)

    # 6. Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL:")
    for r in results:
        print(f"  {r['cp']}: {r['sMAPE']:.3f}%")
    print(f"  OVERALL sMAPE: {overall:.3f}%")
    print(f"  Predicciones totales: {743 * 5}")
    print("=" * 70)
    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    main()