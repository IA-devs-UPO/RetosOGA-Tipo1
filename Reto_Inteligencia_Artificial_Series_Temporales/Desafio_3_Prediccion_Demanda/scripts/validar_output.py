#!/usr/bin/env python3
"""
Valida el CSV de predicciones contra los requisitos del enunciado (Sección 4).
"""

import sys
import numpy as np
import pandas as pd

REQUIRED_COLS = ["fechaHora", "cp_41001", "cp_41003", "cp_41005", "cp_41010", "cp_41020"]
CP_COLS = REQUIRED_COLS[1:]
EXPECTED_ROWS = 743
EXPECTED_TOTAL = 3715

def validar(path):
    print(f"Validando: {path}")
    print("=" * 60)

    errores = []

    # 1. Cargar
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  ❌ No se pudo leer el CSV: {e}")
        sys.exit(1)

    # 2. Columnas exactas
    if list(df.columns) != REQUIRED_COLS:
        errores.append(f"Columnas: esperadas {REQUIRED_COLS}, obtenidas {list(df.columns)}")
    else:
        print(f"  ✅ Columnas exactas: {REQUIRED_COLS}")

    # 3. Filas
    if len(df) != EXPECTED_ROWS:
        errores.append(f"Filas: esperadas {EXPECTED_ROWS}, obtenidas {len(df)}")
    else:
        print(f"  ✅ Filas: {EXPECTED_ROWS}")

    # 4. CPs válidos
    for cp in CP_COLS:
        if cp not in df.columns:
            errores.append(f"Falta columna {cp}")

    # 5. Duplicados fechaHora
    dups = df["fechaHora"].duplicated().sum()
    if dups > 0:
        errores.append(f"Duplicados en fechaHora: {dups}")
    else:
        print(f"  ✅ Sin duplicados en fechaHora")

    # 6. Valores nulos
    nulos = df[CP_COLS].isnull().sum().sum()
    if nulos > 0:
        errores.append(f"Valores nulos en CPs: {nulos}")
    else:
        print(f"  ✅ Sin valores nulos")

    # 7. Negativos
    negativos = (df[CP_COLS] < 0).sum().sum()
    if negativos > 0:
        errores.append(f"Valores negativos: {negativos}")
    else:
        print(f"  ✅ Sin valores negativos")

    # 8. Infinitos
    infinitos = np.isinf(df[CP_COLS].values).sum()
    if infinitos > 0:
        errores.append(f"Valores infinitos: {infinitos}")
    else:
        print(f"  ✅ Sin valores infinitos")

    # 9. Texto / no numérico
    for cp in CP_COLS:
        if not np.issubdtype(df[cp].dtype, np.number):
            errores.append(f"Columna {cp} no es numérica: {df[cp].dtype}")

    # 10. Escala UNE — sanity check contra el target histórico
    try:
        data_dir = path.parent if hasattr(path, "parent") else None
        if data_dir and (data_dir / ".." / "dataset" / "demanda_energia_entrenamiento.csv").exists():
            hist = pd.read_csv(data_dir.parent / "dataset" / "demanda_energia_entrenamiento.csv")
            for cp in CP_COLS:
                if cp in hist.columns:
                        hist_min, hist_max = hist[cp].min(), hist[cp].max()
                        pred_min, pred_max = df[cp].min(), df[cp].max()
                        fuera_rango = ((df[cp] < hist_min * 0.1) | (df[cp] > hist_max * 2)).sum()
                        if fuera_rango > 0:
                            errores.append(
                                f"{cp}: {fuera_rango} predicciones fuera de rango "
                                f"(hist: [{hist_min:.1f}, {hist_max:.1f}], pred: [{pred_min:.1f}, {pred_max:.1f}])"
                            )
    except Exception:
        pass  # No podemos validar escala sin el histórico

    # 11. Formato fechaHora ISO 8601 con timezone
    try:
        fechas = df["fechaHora"]
        for f in fechas:
            pd.Timestamp(f)
        print(f"  ✅ Fechas parseables como Timestamp")
        # Verificar que el primer y último timestamp están en el rango correcto
        primera = pd.Timestamp(fechas.iloc[0])
        ultima = pd.Timestamp(fechas.iloc[-1])
        mar = pd.Timestamp("2023-03-01T00:00:00+01:00")
        abr = pd.Timestamp("2023-04-01T00:00:00+02:00")
        if primera < mar:
            errores.append(f"Primera fecha anterior a marzo: {primera}")
        if ultima >= abr:
            errores.append(f"Última fecha >= abril: {ultima}")
        print(f"  ✅ Rango temporal: {primera} → {ultima}")
    except Exception as e:
        errores.append(f"Formato fecha inválido: {e}")

    # 12. Total predicciones
    total_pred = len(df) * len(CP_COLS)
    if total_pred != EXPECTED_TOTAL:
        errores.append(f"Total predicciones: esperadas {EXPECTED_TOTAL}, obtenidas {total_pred}")
    else:
        print(f"  ✅ Total predicciones: {EXPECTED_TOTAL} ({len(df)} × {len(CP_COLS)})")

    # Resumen
    print("=" * 60)
    if errores:
        print(f"\n❌ VALIDACIÓN FALLIDA — {len(errores)} error(es):")
        for e in errores:
            print(f"   • {e}")
        sys.exit(1)
    else:
        print("\n✅ VALIDACIÓN SUPERADA — El CSV cumple todos los requisitos del enunciado.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "../resultados/predicciones_marzo_2023.csv"
    else:
        path = sys.argv[1]
    validar(path)
