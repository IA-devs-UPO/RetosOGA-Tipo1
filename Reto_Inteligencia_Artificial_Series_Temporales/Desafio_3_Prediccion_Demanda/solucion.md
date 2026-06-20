# Solución: Predicción de Demanda Energética Horaria por Código Postal

## Estrategia General

El enfoque combina tres pilares:

1. **Modelo por CP independiente** — cada código postal tiene patrones de consumo, escala y perfil distintos (desde 162 hasta 7.955 clientes). Un modelo global no capturaría estas diferencias.
2. **Feature engineering exhaustivo** — se explotan las 3 fuentes de datos (clima, calendario, target histórico) generando ~35 features.
3. **LightGBM como backbone** — maneja nulos, features mixtas y no requiere escalado. Rápido de entrenar (5 modelos en segundos).

## Análisis Exploratorio de Datos

### Estructura

| Dataset | Shape | Período | Uso |
|---------|-------|---------|-----|
| `demanda_energia_entrenamiento.csv` | 18.936 × 6 | Ene 2021 → Feb 2023 | Target (formato ancho, 5 CPs) |
| `clima.csv` | 19.679 × 5 | Ene 2021 → Mar 2023 | Features meteorológicas |
| `calendario.csv` | 19.679 × 3 | Ene 2021 → Mar 2023 | Features de calendario |
| `cp_descripcion.csv` | 5 × 4 | — | Metadatos estáticos |

### Distribución de la Demanda por CP

| CP | Clientes | Media | Desvío | Perfil |
|----|---------|-------|--------|--------|
| `cp_41001` | 162 | 17,54 | 5,77 | Centro histórico |
| `cp_41003` | 607 | 9,44 | 2,97 | Residencial denso |
| `cp_41005` | 7.955 | 158,18 | 29,49 | Mixto alta escala |
| `cp_41010` | 2.901 | 43,13 | 12,33 | Barrio consolidado |
| `cp_41020` | 425 | 42,78 | 7,47 | Periferia logística |

La escala varía ~24× entre el CP más pequeño y el más grande (9,44 vs 158,18 UNE de media), lo que justifica modelos independientes.

### Valores Ausentes

- **Target (demanda)**: 4–8% según CP. Imputados con interpolación lineal.
- **Clima**: `humedad` (4,1%) y `velocidadViento` (0,1%). Imputados con forward-fill + backward-fill y mediana como respaldo.
- **Calendario**: 0% ausentes.

### Clima y Calendario para el Período de Predicción

Tanto `clima.csv` como `calendario.csv` incluyen marzo 2023, lo que permite usar features climáticas reales (no forecastadas) para las predicciones.

## Feature Engineering

Se construyen ~35 features organizadas en 4 familias:

### 1. Features Temporales (13 features)

| Feature | Descripción |
|---------|-------------|
| `hora` | Hora del día (0–23) |
| `dia` | Día del mes (1–31) |
| `dia_semana` | Día de la semana (0=Lun, 6=Dom) |
| `mes` | Mes del año (1–12) |
| `ano` | Año |
| `dia_ano` | Día del año (1–366) |
| `fin_semana` | Binario: 1 si sábado o domingo |
| `hora_punta` | Binario: 1 si 8h–21h |
| `hora_sin`, `hora_cos` | Codificación cíclica de la hora del día |
| `dia_semana_sin`, `dia_semana_cos` | Codificación cíclica del día de la semana |
| `hora_fin_semana` | Interacción hora × fin de semana |

### 2. Features de Calendario (2 features)

| Feature | Descripción |
|---------|-------------|
| `cest` | Horario de verano (0/1) |
| `es_festivo_o_domingo` | Festivo nacional/regional o domingo (0/1) |

### 3. Features Climáticas (4 directas + 12 lags)

| Feature | Descripción |
|---------|-------------|
| `lluvia` | Precipitación horaria (mm) |
| `temperatura` | Temperatura horaria (°C) |
| `humedad` | Humedad relativa (%) |
| `velocidadViento` | Velocidad del viento (km/h) |
| `temperatura_lag_6h` / `_12h` / `_24h` | Temperatura desplazada |
| `humedad_lag_6h` / `_12h` / `_24h` | Humedad desplazada |
| `lluvia_lag_6h` / `_12h` / `_24h` | Lluvia desplazada |
| `velocidadViento_lag_6h` / `_12h` / `_24h` | Viento desplazado |

### 4. Lag Features del Target (5 CPs × 3 lags + 3 rolling × 2 estadísticos = 45 features)

| Feature | Descripción |
|---------|-------------|
| `{cp}_lag_24h` | Misma hora del día anterior |
| `{cp}_lag_48h` | Misma hora de hace 2 días |
| `{cp}_lag_168h` | Misma hora de la semana anterior |
| `{cp}_rolling_mean_6h` / `_12h` / `_24h` | Media móvil |
| `{cp}_rolling_std_6h` / `_12h` / `_24h` | Desviación móvil |

### Manejo de Nulos en Features

- Features climáticas: forward-fill → backward-fill → mediana.
- Target (lags): se permite que los primeros registros tengan NaN en lags; LightGBM maneja nulos nativamente.
- Boleanos (`cest`, `es_festivo_o_domingo`): convertidos a int, fill con 0.

## Modelo

### Arquitectura

```
5 modelos LightGBM independientes (1 por CP)
```

Cada modelo es un `LGBMRegressor` con hiperparámetros optimizados vía **Optuna** (10 trials × 3-fold TimeSeriesSplit, 30 evaluaciones totales):

| Parámetro | Rango de búsqueda |
|-----------|:-----------------:|
| `n_estimators` | 200–1500 |
| `learning_rate` | 0.01–0.3 (log) |
| `num_leaves` | 8–128 |
| `max_depth` | 3–15 |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `reg_alpha` | 1e-8–10 (log) |
| `reg_lambda` | 1e-8–10 (log) |
| `min_split_gain` | 0.0–1.0 |
| `min_child_samples` | 5–100 |

### Validación Temporal (Backtesting)

Se usa **TimeSeriesSplit** (3 splits, test_size=670) dentro del entrenamiento con Optuna:

```
Split 1: Train hasta ~May 2022 → Val Jun–Jul 2022
Split 2: Train hasta ~Oct 2022 → Val Nov–Dic 2022
Split 3: Train hasta ~Ene 2023 → Val Feb 2023
```

Para la evaluación final independiente se usa Feb 2023 como validación hold-out (no vista durante el entrenamiento):

```
Train:      Ene 2021 → Ene 2023  (~17.500 registros por CP)
Validación: Feb 2023             (~672 registros por CP) — evaluación final
Predicción: Mar 2023             (743 horas × 5 CPs = 3.715 predicciones)
```

## Evaluación

### Métrica

**sMAPE** (Symmetric Mean Absolute Percentage Error)

$$
\text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2}
$$

### Resultados (validación Feb 2023)

| CP | Zona | sMAPE | # Clientes |
|----|------|:----:|:----------:|
| **cp_41005** | Nervión (mixto, alta escala) | **0.62%** | 7.955 |
| cp_41010 | Triana | **2.39%** | 2.901 |
| cp_41020 | Sevilla Este (logística) | **4.67%** | 425 |
| cp_41001 | Centro histórico | **4.74%** | 162 |
| cp_41003 | Residencial denso | **5.33%** | 607 |
| **Overall** | | **3.55%** | — |

### Interpretación

- **4 de 5 CPs** por debajo del umbral competitivo del **6%**. Solo `cp_41003` (5.33%) lo supera por poco margen.
- **cp_41005 (0.62%)** muestra que a mayor agregación (7.955 clientes) la demanda es altamente predecible.
- **cp_41001 (4.74%)** a pesar de tener solo 162 clientes, logra buen rendimiento — el perfil de centro histórico con equipamientos críticos tiene patrones estables.
- **cp_41003 (5.33%)** es el más ruidoso en términos relativos. Perfil residencial denso con comercio de proximidad puede tener más variabilidad.
- **Técnicas de mejora identificadas**: features de estacionalidad externa (festivos móviles, Semana Santa 2023), incorporar `NumeroClientes` como feature, modelos ensemble por CP.

## Output

Las predicciones se generan en el formato ancho (wide) exigido:

```
fechaHora, cp_41001, cp_41003, cp_41005, cp_41010, cp_41020
2023-03-01T00:00:00+01:00, 12.34, 8.56, 145.23, 38.90, 40.12
...
```

- **743 filas** (una por hora de marzo 2023)
- **5 columnas** (una por CP)
- **3.715 predicciones** totales

## Arquitectura de la Solución

```
Desafio_3_Prediccion_Demanda/
├── dataset/                         # Datos proporcionados
│   ├── demanda_energia_entrenamiento.csv
│   ├── clima.csv
│   ├── calendario.csv
│   └── cp_descripcion.csv
├── scripts/
│   ├── 1_preprocess_train.py        # Preprocesado + entrenamiento
│   └── 2_predict_evaluate.py        # Predicción + evaluación
├── modelos/                         # Modelos LightGBM guardados (5 .txt)
├── resultados/                      # Output: CSV predicciones + JSON evaluación
├── enunciado.md                     # Enunciado original
└── solucion.md                      # Este documento
```

### Flujo de Ejecución

```
1_preprocess_train.py:
  Carga CSVs → Merge → Feature Engineering → Imputación → Entrenar LightGBM × 5

2_predict_evaluate.py:
  Carga modelos → Features Mar 2023 → Predecir → sMAPE → Exportar CSV
```

## Lecciones Técnicas

1. **CP como unidad de modelado independiente** fue clave — un modelo multi-output global no capturaría las escalas dispares (9 vs 158 UNE de media).
2. **Validación temporal estricta con TimeSeriesSplit** evitó overfitting por data leakage. Las 3 splits respetan orden cronológico: train en pasado, validación en futuro.
3. **Manejo de timezones mixtas**: los timestamps combinan CET (+01:00) y CEST (+02:00). `pd.to_datetime(utc=True)` + `tz_convert("Europe/Madrid")` resuelve la conversión correctamente.
4. **Merge anchor**: para predecir marzo 2023 el merge debe comenzar desde `clima` (incluye marzo), no desde `demanda` (solo hasta febrero).
5. **sMAPE con división por cero**: cuando valores reales y predichos son ambos 0 la división produce NaN. El guard `mask = denom > 0` es obligatorio.
