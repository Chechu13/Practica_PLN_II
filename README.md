# NLP II — Grounded Question Answering

> Práctica de **Procesamiento del Lenguaje Natural II** del Grado en Inteligencia Artificial.
>
> Proyecto experimental de respuesta a preguntas factuales en inglés mediante recuperación de contexto (*retrieval-augmented generation*, RAG), modelos Transformer y evaluación cuantitativa.

## Descripción

El objetivo es generar respuestas fundamentadas en una colección local de contextos procedentes de Wikipedia. El proyecto explora distintas estrategias de recuperación y generación, desde arquitecturas Transformer implementadas con PyTorch hasta modelos de Hugging Face, incluido **Qwen2.5-7B-Instruct ajustado mediante LoRA**.

La evaluación utiliza Exact Match (EM) y F1 a nivel de tokens. En el experimento de prompting también se analizan por separado las preguntas respondibles y no respondibles, la latencia y el consumo de memoria.

## Enfoques explorados

- Preprocesado y análisis exploratorio del corpus.
- Recuperación semántica mediante embeddings con `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`.
- Recuperación híbrida que combina similitud semántica y señales léxicas.
- Recuperación con *fallback* externo en experimentos específicos.
- Decoder Transformer y Encoder-Decoder Transformer implementados con PyTorch.
- Modelos de Hugging Face como referencia.
- Fine-tuning de `Qwen/Qwen2.5-7B-Instruct` mediante adaptadores LoRA.
- Comparación entre modelo base y modelos ajustados.
- Estudio de estrategias de prompting **zero-shot, one-shot y few-shot**, con adaptadores entrenados por separado.

## Resultados

### Experimento previo: recuperación con y sin fallback

Esta tabla corresponde a un experimento anterior sobre 70 preguntas de test. Se conserva por separado del estudio de prompting porque las configuraciones no son idénticas.

| Configuración | EM (%) | F1 (%) | N |
|---|---:|---:|---:|
| Qwen Base + Context | 15.71 | 30.85 | 70 |
| Qwen Base + Fallback | 18.57 | 36.87 | 70 |
| Qwen Fine-tuned + Context | **71.43** | **75.35** | 70 |
| Qwen Fine-tuned + Fallback | 68.57 | 73.96 | 70 |

### Comparación de prompting y adaptadores LoRA

El conjunto contiene 350 ejemplos. Con `train_test_split(test_size=0.2, random_state=42)`, se utilizan 280 para entrenamiento y 70 para test. La evaluación del notebook compara el modelo base (`BASE`) con tres adaptadores: `FT_zero`, `FT_one` y `FT_few`.

Resultados registrados en la ejecución más reciente:

| Modelo | EM (%) | F1 (%) | F1 respondibles (%) | F1 no respondibles (%) | Tiempo medio de generación (s) |
|---|---:|---:|---:|---:|---:|
| BASE | 15.71 | 31.11 | 40.33 | 0.00 | 4.651 |
| FT_zero | **78.57** | **81.86** | 78.34 | 93.75 | 1.800 |
| FT_one | 58.57 | 67.28 | 65.00 | 75.00 | 0.417 |
| FT_few | 75.71 | 79.86 | 75.75 | 93.75 | **0.329** |

Cada condición obtuvo 70 predicciones correctas en el sentido de que no hubo errores de ejecución; esto no significa que todas las respuestas fueran correctas. Los valores de EM y F1 evalúan la calidad de las predicciones.

En esta ejecución, `FT_zero` obtuvo el mejor EM y F1 global. `FT_few` quedó aproximadamente **2 puntos de F1** por debajo y tuvo la menor latencia media de generación. `FT_one` quedó por debajo de las otras dos variantes ajustadas. La mejora de los adaptadores frente a BASE es notable en este conjunto de test.

Estas observaciones son descriptivas y corresponden a una ejecución concreta: no demuestran por sí solas significación estadística ni que una estrategia sea universalmente superior. La diferencia pequeña entre `FT_zero` y `FT_few` debe interpretarse con prudencia.

### Eficiencia y análisis de errores

Además de las métricas globales, el experimento guarda tablas específicas para comparar latencia, resultados por tipo de pregunta y diferencias entre estrategias:

- `efficiency_table.csv`: F1, latencia media y percentil 95, velocidad de generación y VRAM máxima.
- `quality_table.csv`: EM, F1 global y F1 por tipo de pregunta.
- `improvement_table.csv`: mejora de cada adaptador respecto a BASE.
- `strategy_table.csv`: diferencias de F1, EM y tiempo entre estrategias.
- `error_table.csv`: análisis de respuestas vacías y errores de ejecución.

Los resultados completos y las predicciones de esta ejecución se guardan en `results/rag_prompt_comparison/`, incluyendo `summary_metrics.csv`, `all_predictions.csv` y las predicciones individuales por modelo. Los CSV parciales permiten inspeccionar el progreso de una ejecución interrumpida.

## Datos y métricas

Las métricas utilizadas son:

- **Exact Match (EM):** indica si la predicción normalizada coincide exactamente con la respuesta de referencia.
- **F1 token-level:** mide el solapamiento entre los tokens de la predicción y la referencia.
- **Answerable F1:** F1 calculado para las preguntas respondibles.
- **Unanswerable F1:** F1 calculado para las preguntas no respondibles.
- **Latencia de generación:** tiempo dedicado a generar la respuesta, separado del tiempo de recuperación cuando corresponde.
- **Tokens/s y VRAM máxima:** medidas complementarias de eficiencia.

La implementación general de las métricas está en `src/evaluate.py`. La interpretación de las métricas por tipo de pregunta depende de cómo estén definidas las referencias y las predicciones vacías en el experimento.

## Estructura del proyecto

La estructura puede evolucionar entre experimentos. Las carpetas principales son:

```text
.
├── data/                 # Datos y contextos
├── models/               # Embeddings y adaptadores/modelos
├── notebooks/            # Análisis y notebooks experimentales
├── prompts/              # Prompts de las distintas estrategias
├── reports/              # Informes
├── results/              # Métricas y predicciones generadas
├── scripts/
│   ├── create_context_embeddings.py
│   └── experiments/      # Scripts de experimentación y entrenamiento
└── src/
    ├── modelos/
    ├── preprocess/
    ├── retrieval/
    └── evaluate.py
```

### Artefactos del experimento de prompting

- `scripts/experiments/train_qwen_prompt.py`: entrenamiento con selección de estrategia mediante `--prompt zero|one|few`.
- `models/prompt_training/qwen_lora_zero/`: adaptador zero-shot.
- `models/prompt_training/qwen_lora_one/`: adaptador one-shot.
- `models/prompt_training/qwen_lora_few/`: adaptador few-shot.
- `results/prompt_training/`: prompts guardados durante el entrenamiento, si están disponibles localmente.
- `notebooks/comparacion_finetuning_prompts.ipynb`: notebook de comparación.
- `results/rag_prompt_comparison/`: métricas, tablas, contextos recuperados y predicciones generadas.

Los adaptadores LoRA y algunos resultados son artefactos locales; su presencia en la estructura descrita no implica que todos los modelos estén versionados en Git.

## Instalación

El proyecto utiliza **Python 3.12** y gestiona las dependencias mediante `uv`.

```bash
git clone https://github.com/Chechu13/Practica_PLN_II.git
cd Practica_PLN_II
uv sync
```

El proyecto está configurado para utilizar versiones de PyTorch compatibles con **CUDA 12.8** en Windows.

Para generar o regenerar los embeddings de los contextos:

```bash
uv run python scripts/create_context_embeddings.py
```

## Reproducibilidad

Para que las comparaciones sean interpretables:

1. Mantén fija la partición de entrenamiento y test.
2. Utiliza el mismo recuperador local `ContextRetriever` y los mismos contextos recuperados para todos los modelos comparados.
3. No actives el *fallback* web en el experimento de prompting, ya que introduciría información externa.
4. Comprueba que cada adaptador se evalúe con la estrategia de prompt que le corresponde y, cuando estén guardados, con los prompts usados durante el entrenamiento.
5. Conserva las predicciones individuales y las métricas junto con la configuración de generación.

El tiempo medido depende del hardware, la carga de la GPU y la configuración de generación. Los resultados publicados aquí deben entenderse como los de la ejecución registrada, no como garantías de rendimiento en otro equipo.

## Estado del proyecto

🚧 **En desarrollo.**

El proyecto reúne experimentos de recuperación, generación y fine-tuning con LoRA. La comparación más reciente sugiere que el ajuste mejora notablemente el rendimiento frente al modelo base y que `FT_zero` y `FT_few` ofrecen resultados globales cercanos, con distintas latencias. Conviene validar los prompts exactos y repetir las evaluaciones antes de formular conclusiones definitivas.

---

**Universidad Rey Juan Carlos — Grado en Inteligencia Artificial**  
**Procesamiento del Lenguaje Natural II**
