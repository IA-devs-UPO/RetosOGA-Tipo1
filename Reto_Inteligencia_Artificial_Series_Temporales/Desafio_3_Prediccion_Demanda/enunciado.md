### **Título del Desafío:** Predicción de Demanda Energética Horaria por Código Postal

#### **1. Contexto del Problema**

La empresa municipal de energía de Sevilla necesita predecir la demanda energética horaria agregada, expresada en **UNE** (unidad normalizada de energía), para cinco códigos postales urbanos (CP). Se deben generar predicciones horarias para **marzo de 2023** (31 días × 24h − 1h = 743 horas; el 26 de marzo a las 02:00 cambia a CEST, eliminando una hora), con **5 CPs** por hora, totalizando **3.715 predicciones** (743 × 5).

La métrica de evaluación es la **sMAPE** (Symmetric Mean Absolute Percentage Error). Valores competitivos: por debajo del **6%**.

#### **2. Datasets**

Los 4 archivos están en `dataset/`. El feature temporal común es `fechaHora` en formato ISO 8601 con timezone (`+01:00` CET / `+02:00` CEST).

---

##### **`dataset/demanda_energia_entrenamiento.csv`** — Variable objetivo (target)

- **Formato**: ancho (wide): una fila por hora, una columna por CP.
- **Columnas**: `fechaHora`, `cp_41001`, `cp_41003`, `cp_41005`, `cp_41010`, `cp_41020`
- **Rango temporal**: `2021-01-01T00:00:00+01:00` → `2023-02-28T23:00:00+01:00`
- **Total**: 18.936 registros horarios
- **Valores ausentes**: sí — distribución por CP:

  | Columna | % ausentes | Valores | Mínimo | Máximo | Media |
  |---|---|---|---|---|---|
  | `cp_41001` | 4,1% | 18.158 | 3,80 | 60,49 | 17,54 |
  | `cp_41003` | 0,6% | 18.831 | 3,98 | 24,35 | 9,44 |
  | `cp_41005` | 4,0% | 18.178 | 89,70 | 228,27 | 158,18 |
  | `cp_41010` | 5,9% | 17.824 | 19,72 | 167,26 | 43,13 |
  | `cp_41020` | 8,0% | 17.425 | 27,95 | 84,49 | 42,78 |

---

##### **`dataset/cp_descripcion.csv`** — Metadatos estáticos de cada CP

| Columna | Descripción |
|---|---|
| `CodifoPostal` | Identificador del CP (ej. `cp_41005`) |
| `Area` | Nombre del barrio o zona |
| `Descripcion` | Perfil de consumo: tipo de zona (residencial, comercial, mixto, equipamientos críticos) |
| `NumeroClientes` | Número de clientes conectados en ese CP |

- **Nota**: El nombre de columna `CodifoPostal` contiene una errata (`f` por `g`). Úsalo tal cual.
- Datos de los 5 CPs:

  | CP | Zona | Clientes | Perfil |
  |---|---|---|---|
  | `cp_41001` | Arenal - Santa Cruz - Alfalfa | 162 | Centro histórico, equipamientos críticos, actividad terciaria |
  | `cp_41003` | Santa Catalina - San Julián - Feria | 607 | Residencial urbano denso, comercio proximidad |
  | `cp_41005` | Nervión - La Buhaira - Ciudad Jardín | 7.955 | Mixto residencial-comercial, alta escala de carga |
  | `cp_41010` | Triana | 2.901 | Barrio consolidado, patrón residencial y servicios |
  | `cp_41020` | Sevilla Este - Santa Clara - Este-Alcosa | 425 | Periferia expansión, actividad logística/terciaria |

---

##### **`dataset/clima.csv`** — Variables meteorológicas horarias

- **Rango temporal**: `2021-01-01T00:00:00+01:00` → `2023-03-31T23:00:00+02:00` **(incluye marzo 2023)**
- **Total**: 19.679 registros horarios
- **Estación**: única, representativa y común a todos los CP.

  | Columna | Descripción | Unidad | % ausentes | Mínimo | Máximo | Media |
  |---|---|---|---|---|---|---|
  | `lluvia` | Precipitación | mm | 0% | 0,00 | 45,70 | 0,08 |
  | `temperatura` | Temperatura | °C | 0% | -2,10 | 35,40 | 15,21 |
  | `humedad` | Humedad relativa | % | 4,1% | 17,00 | 100,00 | 65,38 |
  | `velocidadViento` | Velocidad del viento | km/h | 0,1% | 1,00 | 77,00 | 14,13 |

- **Valores ausentes**: permitida la imputación (en `humedad` y `velocidadViento`).

---

##### **`dataset/calendario.csv`** — Variables de calendario

- **Rango temporal**: `2021-01-01T00:00:00+01:00` → `2023-03-31T23:00:00+02:00` **(incluye marzo 2023)**
- **Total**: 19.679 registros horarios

  | Columna | Tipo | Descripción |
  |---|---|---|
  | `cest` | `True` / `False` | `True` si el registro está en horario de verano (CEST, UTC+2); `False` en horario estándar (CET, UTC+1) |
  | `es_festivo_o_domingo` | `True` / `False` | `True` si es festivo nacional/regional o domingo |

---

#### **3. Evaluación**

La métrica oficial es la **sMAPE**:

$$ \text{sMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|) / 2} $$

- **$n$ = 3.715** (743 horas × 5 CPs)
- Las predicciones se comparan contra los valores reales observados en marzo de 2023.
- Gana el equipo con **menor sMAPE**. Umbral competitivo: **< 6%**.

---

#### **4. Formato de Entrega**

El archivo de resultados debe ser **CSV** con **3.716 filas** (cabecera + 3.715 predicciones).

**Columnas exactas** (mismo orden):
```
fechaHora,cp_41001,cp_41003,cp_41005,cp_41010,cp_41020
```

**Horizonte temporal**: todas las marcas horarias locales de Sevilla (`Europe/Madrid`) desde `2023-03-01T00:00:00+01:00` hasta `2023-03-31T23:00:00+02:00`. No existe la hora `2023-03-26T02:00:00` (cambio a CEST) → **743 registros** de `fechaHora`.

**Reglas**:
- `cp` debe ser uno de: `cp_41001`, `cp_41003`, `cp_41005`, `cp_41010`, `cp_41020`.
- **Sin duplicados** por combinación `fechaHora–cp`.
- **Sin valores** faltantes, infinitos, texto ni negativos.
- Predicciones en **UNE**, misma escala que el target histórico.
- El nombre del archivo es libre, extensión `.csv`.

**Ejemplo**:

```
fechaHora,cp_41001,cp_41003,cp_41005,cp_41010,cp_41020
2023-03-01T00:00:00+01:00,11.200,4.800,130.125,36.450,40.900
2023-03-01T01:00:00+01:00,9.750,4.300,121.500,34.210,46.005
```
