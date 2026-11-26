# Módulo 5 — Análisis de la evaluación con Ragas

Juez: `qwen2.5:7b-instruct` en local (Ollama). Métricas: `faithfulness`,
`answer_relevancy`, `llm_context_precision_with_reference`, `context_recall`.
Datos completos en `resultados_ragas_sample.csv` y `resultados_ragas_rag.csv`.

## 1. Dataset de ejemplo del enunciado

Salida de `python evaluate_rag.py`, tal cual:

```
[1/3] Construyendo dataset 'sample'...
[2/3] Evaluando 2 preguntas con juez 'qwen2.5:7b-instruct'...

--- Puntuaciones por pregunta ---
                                user_input  faithfulness  answer_relevancy  llm_context_precision_with_reference  context_recall
What is the company policy on remote work?           1.0          0.792495                                   1.0             1.0
 When was the company founded and by whom?           1.0          0.975903                                   0.5             1.0

--- Medias ---
{'faithfulness': 1.0000, 'answer_relevancy': 0.8842, 'llm_context_precision_with_reference': 0.7500, 'context_recall': 1.0000}

[3/3] Detalle completo guardado en resultados_ragas_sample.csv
```

La precisión de contexto baja a 0.5 en la segunda pregunta (*"When was the
company founded and by whom?"*) porque de los dos fragmentos recuperados solo
uno es útil para la parte de "by whom": es exactamente lo que mide la métrica,
la proporción de ruido en el contexto.

Detalle revelador: la respuesta de esa pregunta omite a Alice Johnson y aun así
`context_recall` sale 1.0. No es un error. `context_recall` evalúa el
**contexto recuperado**, no la respuesta: el contexto sí traía el nombre. El
fallo es del generador, y quien debería haberlo detectado es
`answer_relevancy`, que se queda en 0.976 — demasiado benévola. Un juez de 7B
parámetros no es lo bastante fino aquí; con un modelo mayor esa nota bajaría.

## 2. Pipeline real del ejercicio 1

Línea base: extracción con Marker sobre GPU, 21 chunks indexados, `TOP_K = 4` y
**solo búsqueda densa**. Salida de `evaluate_rag.py --dataset rag` con
`RETRIEVAL_MODE=dense`:

```
--- Puntuaciones por pregunta ---
                                   user_input  faithfulness  answer_relevancy  llm_context_precision_with_reference  context_recall
What SQL Server version is required to use...      0.000000          0.000000                                   1.0             0.0
Which Unity Catalog permissions does the u...      1.000000          0.908982                                   1.0             1.0
Which SQL Server database variations does ...      1.000000          0.996317                                   1.0             1.0
Which SQL Server edition is required to us...      0.333333          0.922106                                   1.0             1.0
How is the traffic between the client inte...      1.000000          0.809464                                   1.0             1.0

--- Medias ---
{'faithfulness': 0.6667, 'answer_relevancy': 0.7274, 'llm_context_precision_with_reference': 1.0000, 'context_recall': 0.8000}
```

Las medias esconden que **4 de 5 preguntas salen bien y una se hunde**. Esa es
la pregunta 1:

> *What SQL Server version is required to use change tracking with Lakeflow Connect?*

El dato está en la página 49 del PDF ("SQL Server 2012 or a later version must
be enabled to use change tracking"), el sistema recuperó las páginas 47 y 50-51,
y el modelo respondió correctamente que no aparece en la documentación. Con
`context_recall = 0` el diagnóstico es inequívoco: **el cuello de botella es el
retriever, no el generador**.

Ampliando la búsqueda a 8 resultados se ve exactamente dónde queda el chunk bueno:

| Rango | Página | Score | ¿Contiene la respuesta? |
|---|---|---|---|
| 1 | 47 | 0.845 | no |
| 2 | 47 | 0.791 | no |
| 3 | 50 | 0.735 | no |
| 4 | 51 | 0.700 | no |
| 5 | 52 | 0.687 | no |
| **6** | **49** | **0.687** | **sí** |

No es que el chunk no exista ni que esté mal troceado: está en el puesto 6, a
0.013 puntos del quinto, y `TOP_K = 4` lo deja fuera por dos posiciones.

### Por qué falla la recuperación

La pregunta contiene "change tracking" y "SQL Server version", dos términos casi
literales del documento. Un embedding denso los promedia con el resto de la
frase y los diluye; los chunks ganadores hablan de compatibilidad de versiones
*en general* y ganan por similitud temática. Es el caso de libro donde la
búsqueda densa pierde frente a la léxica.

### Efecto del extractor y del troceado

Se evaluaron tres configuraciones sobre las mismas 5 preguntas:

| Configuración | chunks | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|---|
| pymupdf4llm | 19 | 0.667 | 0.735 | 0.800 | 0.800 |
| Marker | 23 | 0.667 | 0.728 | **1.000** | 0.800 |
| Marker + tablas compactadas | 21 | 0.667 | 0.727 | **1.000** | 0.800 |

**pymupdf → Marker** sube la precisión de contexto de 0.800 a 1.000. pymupdf
parte cada celda multilínea en una fila suelta de la tabla; Marker mantiene la
fila entera. Esos fragmentos rotos entraban en el contexto como ruido.

**Marker → Marker con tablas compactadas** viene de un fallo que las métricas
casi no vieron pero que sí destrozaba las respuestas. Marker alinea las columnas
rellenando con espacios, y ese relleno triplica los tokens de la tabla: la de
permisos de Unity Catalog se pasaba del `chunk_size` de 500 y el chunker la
partía en dos. La segunda mitad quedaba como filas sueltas **sin fila de
cabecera**, así que el modelo leía la columna *Documentation* como si fueran
nombres de permisos e inventaba respuestas del tipo "necesita GRANT PRIVILEGES y
MANAGE COMPUTE POLICIES". Al preguntar en español, la respuesta era directamente
incorrecta.

Lo interesante es que **ninguna de las cuatro métricas lo detectó**: la fila del
arreglo es idéntica a la anterior. Y es coherente. Las filas mal interpretadas
*sí* están literalmente en el contexto, así que la respuesta es "fiel" a lo
recuperado; el error es de interpretación de la estructura de la tabla, no de
alucinación, y ninguna de las cuatro métricas mira si el chunk es
sintácticamente válido.

La evidencia del arreglo, por tanto, no es cuantitativa sino la respuesta misma.
Antes, a *"¿qué permisos de Unity Catalog necesita Lakeflow Connect?"*:

> "GRANT PRIVILEGES", "MANAGE COMPUTE POLICIES" ← valores de la columna *Documentation*

Después, las cuatro filas reales de la tabla, con `CREATE CONNECTION on the
metastore` incluida. La conclusión práctica es que **Ragas no sustituye a mirar
los chunks**: el fallo se encontró leyendo el `chunks.jsonl`, no la tabla de
puntuaciones.

El arreglo son dos funciones en `ingest_pdf.py`: `compact_tables()` quita el
relleno de espacios antes de trocear, y `repair_table_header()` devuelve la fila
de cabecera a cualquier chunk que empiece en mitad de una tabla, por si aun así
alguna no cabe.

### Faithfulness baja en una respuesta correcta

La pregunta 4 se responde bien y aun así puntúa 0.333. Al desglosar las
afirmaciones se ve el motivo: el juez cuenta como afirmación no respaldada el
metacomentario del propio modelo ("*esto se menciona en el fragmento [1]*").
Esa frase no está en el contexto, así que la marca como no fundamentada. Es un
artefacto de combinar un prompt que pide citas con un juez pequeño, no una
alucinación. Quitar la instrucción de citar subiría la métrica y empeoraría el
producto, que es un buen recordatorio de que la métrica no es el objetivo.

## 3. Intervención: búsqueda híbrida

El diagnóstico anterior señalaba al retriever, así que se implementó la primera
tarea opcional del enunciado: indexar también vectores dispersos BM25 y fusionar
las dos búsquedas con Reciprocal Rank Fusion (`RETRIEVAL_MODE=hybrid`).

RRF puntúa por **posición**, no por score. Es lo que hace falta aquí: una
distancia coseno y un peso BM25 están en escalas distintas y no se pueden sumar,
pero sus rankings sí se pueden combinar.

### Efecto sobre la pregunta que fallaba

| | Rango 1 | Rango 2 | Rango 3 | Rango 4 |
|---|---|---|---|---|
| **dense** | p.47 | p.47 | p.50 | p.51 |
| **hybrid** | p.47 | **p.49 ✅** | p.50 | p.47 |

El chunk con la respuesta pasa del puesto 6, fuera del top-4, al puesto 2. BM25
lo encuentra sin dificultad porque "change tracking" aparece literalmente en él,
que es exactamente la señal que el embedding denso promediaba y perdía.

### Efecto sobre las métricas

Misma extracción, mismos 21 chunks, mismo `TOP_K = 4`. Solo cambia `retrieve()`:

| Métrica | dense | hybrid | |
|---|---|---|---|
| faithfulness | 0.667 | **0.933** | +0.267 |
| answer_relevancy | 0.727 | **0.931** | +0.204 |
| llm_context_precision_with_reference | 1.000 | 1.000 | = |
| context_recall | 0.800 | **1.000** | +0.200 |

`context_recall = 1.000`: las cinco preguntas recuperan ya todo lo necesario.
`faithfulness` y `answer_relevancy` suben de rebote, no porque el generador haya
mejorado, sino porque la pregunta 1 pasa de 0.000 a puntuar como las demás. Es
la confirmación del diagnóstico: **el generador nunca fue el problema**.

Se esperaba que `context_precision` bajase, porque BM25 mete resultados por
coincidencia léxica que no siempre son relevantes. No ocurrió: se mantiene en
1.000. Con `TOP_K = 4` sobre un corpus de 21 chunks hay poco margen para que
entre ruido; en un corpus grande esa caída sí sería el precio a pagar.

Los datos por pregunta de ambos modos están en `resultados_ragas_rag_dense.csv`
y `resultados_ragas_rag_hybrid.csv`.

## 4. Qué haría con estos números

**Si `context_recall` es bajo** — era el caso antes de la intervención. Por
orden de coste:

1. **Búsqueda híbrida**: implementada, ver sección 3. Subió el recall de 0.800 a
   1.000 sin coste en precisión.
2. **Subir `TOP_K`** de 4 a 8. Arreglo inmediato de una línea que también habría
   bastado aquí, porque el chunk bueno estaba en el puesto 6. Es peor solución:
   mete cuatro chunks de ruido en todas las consultas para arreglar una.
3. **Chunks más pequeños con solape** (`--chunker token`): la página 49 mezcla
   varios requisitos en un chunk de 500 tokens, lo que diluye su embedding.
4. **Reranker** sobre un top-20 inicial: recupera ancho y luego afina. Sube la
   precisión sin sacrificar recall, a cambio de latencia.

**Si `faithfulness` es bajo** (y no es un artefacto como el de arriba):
bajar la temperatura a 0 (ya está), endurecer la instrucción de no salir del
contexto, y sobre todo **comprobar antes `context_recall`**: si el contexto no
traía la respuesta, el problema no es que el modelo alucine, es que no le hemos
dado con qué responder. Arreglar el generador cuando el fallo es del retriever
es el error de diagnóstico clásico.

**Si `answer_relevancy` es bajo**: el contexto es bueno pero la respuesta no
aterriza en la pregunta. Ahí sí toca el prompt o un modelo generador mayor.

## 5. Limitaciones de esta evaluación

- 5 preguntas es una muestra demasiado pequeña para leer las medias como algo
  estable; sirve para diagnosticar, no para comparar configuraciones finas.
- **El juez no es determinista aun con `temperature=0`.** Dos ejecuciones
  seguidas de la misma configuración híbrida dieron `faithfulness` 0.900 y
  0.933, y la densa 0.667 y 0.733. Las diferencias de ±0.03 entre configuraciones
  no significan nada; el salto de +0.267 de la sección 3 sí, porque es un orden
  de magnitud mayor que ese ruido. Lo riguroso sería promediar varias corridas.
- Juez y generador son el **mismo modelo**, lo que introduce sesgo a favor de
  sus propias respuestas. Lo correcto sería un juez distinto y más grande.
- El generador no siempre respeta el idioma de la pregunta (la pregunta 4 es en
  inglés y responde en español), algo que ninguna de las cuatro métricas capta.
