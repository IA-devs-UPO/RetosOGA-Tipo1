### **Título del Desafío:** Predicción de Demanda Intermitente de Repuestos Críticos para Aerogeneradores

#### **1. Contexto del Problema**

Un gran operador de parques eólicos terrestres cuenta con una flota de $N$ aerogeneradores distribuidos geográficamente. El mantenimiento correctivo de estos equipos requiere de componentes altamente especializados (como tarjetas de control del multiplicador, sensores de guiñada y actuadores de paso de pala). Las series temporales de uso de estos componentes son extremadamente esporádicas e intermitentes: hay meses consecutivos con cero solicitudes, interrumpidos ocasionalmente por picos de demanda debido a fallas estructurales por desgaste o tormentas.

La empresa sufre un dilema: un exceso de almacenamiento de estas piezas inmoviliza millones de euros en inventario, pero la falta de stock ante una falla crítica detiene el aerogenerador, lo que cuesta miles de euros diarios en energía no producida.

#### **2. Conjunto de Datos Provistos**

Los participantes recibirán las siguientes bases de datos:

* `historial_mantenimiento.csv`: Registros semanales de sustitución de piezas de los últimos 5 años por aerogenerador y tipo de componente (con una tasa de observaciones con valor de $0$ superior al 75%).


* `climatologia_historica.csv`: Series diarias de velocidad media del viento, ráfagas máximas, humedad relativa, temperatura extrema y número de impactos de rayo registrados por parque eólico.
* `especificaciones_turbinas.csv`: Identificador del aerogenerador, modelo, fecha de instalación, fabricante del componente crítico y horas totales de operación acumuladas.

#### **3. Objetivo Predictivo y Trampa de la Métrica**

El objetivo es predecir la demanda agregada de cada componente para las próximas 4 semanas en cada parque eólico.

* **La trampa del MAE:** El hackatón se evalúa oficialmente mediante el Error Absoluto Medio ($MAE$) debido a su fácil traducción a costes financieros directos. Sin embargo, dado que la mediana de una serie con más del 50% de ceros es exactamente cero, los modelos estándar de Machine Learning (como XGBoost o redes LSTM tradicionales) entrenados por defecto para optimizar el $MAE$ convergerán rápidamente a una predicción plana de cero. Este modelo será inútil para la operación, ya que jamás recomendará reponer stock.


* **Desafío técnico:** Los participantes deberán esquivar esta trampa matemática mediante técnicas como:
1. Estrategias híbridas de clasificación-regresión (modelando primero la probabilidad de que la demanda sea mayor que cero y, posteriormente, estimando su magnitud).


2. Implementación de funciones de pérdida suavizadas y parametrizadas (como la pérdida Huber o aproximaciones personalizadas).


3. Uso de agrupamientos (clustering) morfológicos previos mediante distancias elásticas (como DTW o k-Shape) para identificar patrones de fallas comunes antes de entrenar los modelos.
