# NLP II — Grounded Question Answering

> Práctica de **Procesamiento del Lenguaje Natural II** del Grado en Inteligencia Artificial.
>
> Proyecto experimental de respuesta a preguntas factuales en inglés con recuperación de contexto (*retrieval-augmented generation*, RAG), modelos Transformer y evaluación cuantitativa.

## Descripción

El objetivo es generar respuestas fundamentadas en una colección interna de contextos procedentes de Wikipedia. El proyecto compara distintas estrategias de recuperación y generación, desde arquitecturas Transformer implementadas con PyTorch hasta modelos de Hugging Face, incluido **Qwen2.5-7B-Instruct con fine-tuning mediante LoRA**.

La evaluación considera la coincidencia exacta de las respuestas, el solapamiento de tokens y, en algunos experimentos, el comportamiento por separado en preguntas respondibles y no respondibles.

## Enfoques explorados

- Preprocesado y análisis exploratorio del corpus.
- Recuperación semántica mediante embeddings con `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`.
- Recuperación híbrida que combina similitud semántica y señales léxicas.
- Recuperación con *fallback* externo en experimentos específicos.
- Decoder Transformer y Encoder-Decoder Transformer implementados desde cero con PyTorch.
- Modelos de Hugging Face como referencia.
- Fine-tuning de `Qwen/Qwen2.5-7B-Instruct` mediante LoRA.
- Comparación entre modelo base y modelo ajustado, y entre variantes de recuperación.
- Estudio de estrategias de prompting: **zero-shot, one-shot y few-shot**, tanto con el modelo base como con adaptadores LoRA entrenados por separado.

## Resultados

### Comparación de recuperación: modelo base frente a fine-tuning

La siguiente tabla recoge un experimento previo de comparación sobre **70 preguntas de test**. Las puntuaciones se expresan en porcentaje.

| Configuración | EM (%) | F1 (%) | N |
|---|---:|---:|---:|
| Qwen Base + Context | 15.71 | 30.85 | 70 |
| Qwen Base + Fallback | 18.57 | 36.87 | 70 |
| Qwen Fine-tuned + Context | **71.43** | **75.35** | 70 |
| Qwen Fine-tuned + Fallback | 68.57 | 73.96 | 70 |

Estos valores pertenecen a ese experimento concreto y no deben combinarse directamente con los resultados del estudio de prompting descrito a continuación.

### Estudio de prompting y adaptadores LoRA

Se entrenaron adaptadores LoRA independientes para las estrategias zero-shot, one-shot y few-shot, usando una partición de entrenamiento/test de 280/70 ejemplos. El objetivo es medir si los ejemplos incluidos en el prompt aportan una mejora adicional y cómo interactúan con el fine-tuning.

Resultados preliminares observados en la gráfica de comparación:

| Configuración | F1 global aproximado (%) | Interpretación provisional |
|---|---:|---|
| Base-Zero | 31 | Mejor que Base-One en esta ejecución |
| Base-One | 12 | Rendimiento bajo |
| Base-Few | 0 | Resultado que requiere validación |
| FT-Zero | **80** | Mejor resultado observado en esta ejecución |
| FT-One | 21 | Muy inferior a FT-Zero |
| FT-Few | 0 | Resultado que requiere validación |

Los valores de esta tabla son **lecturas aproximadas de la gráfica**, no una transcripción del CSV de resultados. Deben sustituirse por las cifras exactas de `results/prompt_training_comparison/summary.csv` cuando estén disponibles.

La evaluación preliminar favorece al adaptador entrenado con zero-shot. Sin embargo, la caída extrema de las variantes few-shot —y la diferencia entre adaptadores— hace necesario validar antes de extraer conclusiones definitivas:

- que cada adaptador se evalúa con el prompt empleado durante su entrenamiento;
- que los ejemplos one-shot/few-shot se insertan correctamente y sus marcadores se resuelven;
- que la generación y la extracción de respuestas son idénticas entre condiciones;
- que todas las condiciones usan el mismo conjunto de test y el mismo contexto recuperado.

No se considera demostrado que one-shot o few-shot sean intrínsecamente peores hasta completar estas comprobaciones. Las métricas completas y las predicciones deben consultarse en los archivos de resultados generados por los experimentos.

## Datos y evaluación

El conjunto usado en el estudio de prompting contiene 350 ejemplos; con `train_test_split(test_size=0.2, random_state=42)` se obtienen **280 ejemplos de entrenamiento y 70 de test**.

Las métricas utilizadas incluyen:

- **Exact Match (EM):** comprueba si la predicción normalizada coincide exactamente con la respuesta de referencia.
- **F1 token-level:** mide el solapamiento entre los tokens de la predicción y la referencia.
- **Answerable F1:** F1 calculado para las preguntas respondibles.
- **Unanswerable F1:** F1 calculado para las preguntas no respondibles.

La implementación de las métricas generales está en `src/evaluate.py`. Las métricas por tipo de pregunta se usan en el experimento de prompting.

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
- `results/prompt_training/`: prompts usados durante el entrenamiento.
- `results/prompt_training_comparison/`: resumen y predicciones de la comparación, cuando se ha ejecutado el notebook correspondiente.

Los adaptadores LoRA y los resultados son artefactos locales; su presencia en la estructura descrita no implica que estén versionados en Git.

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

## Reproducibilidad y uso del recuperador

En los experimentos de prompting, la evaluación debe emplear el **recuperador local `ContextRetriever`** y los embeddings ya generados. No se debe activar un recuperador con *fallback* web, ya que introduciría fuentes externas y rompería la comparación con contexto exclusivamente local.

Para reproducir las comparaciones, conserva la misma partición de test, el mismo recuperador, los mismos contextos y la misma configuración de generación entre condiciones. Registra las métricas exactas y guarda las predicciones junto con los prompts empleados.

## Estado del proyecto

🚧 **En desarrollo.**

El proyecto incluye varios enfoques de recuperación y generación, fine-tuning con LoRA y experimentos de prompting. La prioridad actual es consolidar la evaluación comparativa, verificar la construcción de los prompts —especialmente en few-shot— y documentar los resultados finales sin mezclar experimentos con configuraciones distintas.

---

**Universidad Rey Juan Carlos — Grado en Inteligencia Artificial**  
**Procesamiento del Lenguaje Natural II**
