import numpy as np
import os


def evaluate_custom_loss(loss_name: str, y_true: list[float], y_pred: list[float]) -> float:
    y_t, y_p = np.array(y_true), np.array(y_pred)
    if loss_name.lower() == 'mae':
        return float(np.mean(np.abs(y_t - y_p)))
    elif loss_name.lower() == 'huber':
        d = 1.0; e = np.abs(y_t - y_p)
        return float(np.mean(np.where(e <= d, 0.5 * e**2, d * (e - 0.5 * d))))
    elif loss_name.lower() == 'log-cosh':
        return float(np.mean(np.log(np.cosh(y_p - y_t))))
    raise ValueError(f"Pérdida desconocida: {loss_name}")


def save_script(filename: str, content: str, challenge_dir: str = "Desafio_3_Prediccion_Demanda") -> str:
    """Guarda un script Python en scripts/ del desafío. ÚSALA para crear los archivos de solución."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', challenge_dir))
    scripts_dir = os.path.join(base, 'scripts')
    os.makedirs(scripts_dir, exist_ok=True)
    p = os.path.join(scripts_dir, filename)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
    return p


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Desafio_3_Prediccion_Demanda', 'dataset'))


def explore_demand_data() -> dict:
    """Carga los 4 CSV y devuelve resumen: forma, columnas, nulos, estadísticas."""
    import pandas as pd
    result = {}
    files = {
        "demanda": "demanda_energia_entrenamiento.csv",
        "clima": "clima.csv",
        "calendario": "calendario.csv",
        "cp_desc": "cp_descripcion.csv",
    }
    for key, fname in files.items():
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "nulos": {c: int(df[c].isna().sum()) for c in df.columns if c != "fechaHora"},
        }
        if key == "demanda":
            info["cp_stats"] = {c: {"media": float(round(df[c].mean(), 2)), "min": float(round(df[c].min(), 2)), "max": float(round(df[c].max(), 2)), "nulos_pct": round(df[c].isna().mean() * 100, 1)} for c in df.columns if c != "fechaHora"}
        result[key] = info
    return result


def run_demand_pipeline(use_lag_hours: bool = True, max_evals: int = 30) -> dict:
    """Pipeline completo: merge, features, train LightGBM, predict Marzo 2023, calcula sMAPE."""
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_percentage_error
    import warnings
    warnings.filterwarnings("ignore")

    target = pd.read_csv(os.path.join(DATA_DIR, "demanda_energia_entrenamiento.csv"))
    clima = pd.read_csv(os.path.join(DATA_DIR, "clima.csv"))
    cal = pd.read_csv(os.path.join(DATA_DIR, "calendario.csv"))

    for d in [target, clima, cal]:
        d["fechaHora"] = pd.to_datetime(d["fechaHora"], utc=True)

    df = target.merge(clima, on="fechaHora", how="left").merge(cal, on="fechaHora", how="left")
    df = df.sort_values("fechaHora").reset_index(drop=True)

    cp_cols = [c for c in target.columns if c.startswith("cp_")]
    features = ["hora", "dia_semana", "mes", "finde", "cest", "es_festivo_o_domingo",
                "lluvia", "temperatura", "humedad", "velocidadViento"]
    if use_lag_hours:
        features += ["sin24", "cos24"]

    def featurize(df):
        df = df.copy()
        df["hora"] = df["fechaHora"].dt.hour.fillna(0).astype(int)
        df["dia_semana"] = df["fechaHora"].dt.dayofweek.fillna(0).astype(int)
        df["mes"] = df["fechaHora"].dt.month.fillna(1).astype(int)
        df["finde"] = df["dia_semana"].isin([5, 6]).astype(int)
        if use_lag_hours:
            ang = 2 * np.pi * df["hora"] / 24
            df["sin24"] = np.sin(ang)
            df["cos24"] = np.cos(ang)
        for c in ["cest", "es_festivo_o_domingo"]:
            if c in df.columns:
                df[c] = df[c].fillna(False).astype(int)
        for c in ["lluvia", "temperatura", "humedad", "velocidadViento"]:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())
        return df

    df = featurize(df)
    split_date = pd.Timestamp("2023-02-01", tz="UTC")
    train = df[df["fechaHora"] < split_date].copy()
    val = df[(df["fechaHora"] >= split_date)].copy()

    results = {}
    all_actuals, all_preds = [], []

    for cp in cp_cols:
        tr = train.dropna(subset=[cp]).copy()
        va = val.copy()
        X_tr, y_tr = tr[features], tr[cp]
        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=42)
        model.fit(X_tr, y_tr)
        va["pred"] = model.predict(va[features])
        mask = va[cp].notna()
        if mask.sum() > 0:
            y_real, y_pred = va.loc[mask, cp], va.loc[mask, "pred"]
            smape = 100 * (np.abs(y_real - y_pred) / ((np.abs(y_real) + np.abs(y_pred)) / 2)).mean()
            results[cp] = {"sMAPE_val": float(round(smape, 3)), "train_size": len(X_tr), "val_size": int(mask.sum())}
            all_actuals.extend(y_real.tolist())
            all_preds.extend(y_pred.tolist())

    if all_actuals:
        overall = 100 * (np.abs(np.array(all_actuals) - np.array(all_preds)) / ((np.abs(np.array(all_actuals)) + np.abs(np.array(all_preds))) / 2)).mean()
        results["overall"] = {"sMAPE_val": float(round(overall, 3)), "total": len(all_actuals)}

    return results