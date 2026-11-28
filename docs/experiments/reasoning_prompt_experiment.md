# Experimento de prompts orientados al razonamiento

## Hipótesis

Una instrucción que guíe al modelo a identificar la intención de la pregunta,
localizar evidencia y comprobar si esta permite responder puede mejorar el
question answering factual. No se presupone que mejoren EM o F1: se medirá.

## Diseño

Se comparan, por separado, las condiciones zero-shot, one-shot y few-shot
existentes con sus equivalentes de razonamiento:

| Baseline | Razonamiento |
|---|---|
| `zero` | `zero_reasoning` |
| `one` | `one_reasoning` |
| `few` | `few_reasoning` |

Las plantillas nuevas preservan el formato de respuesta corta y abstención de
las plantillas controladas. Solo añaden una lista de comprobaciones internas
y prohíben mostrar análisis intermedio. El objetivo supervisado durante el
fine-tuning sigue siendo exclusivamente la respuesta de referencia: no se
fabrican trazas de razonamiento ni se entrena al modelo a emitirlas.

Se reutilizan Qwen2.5-7B-Instruct, LoRA, semilla 42, partición 80/20,
recuperación local y los hiperparámetros del entrenamiento controlado. Las
demostraciones one/few-shot se seleccionan exclusivamente del split de train.
Los adaptadores y resultados se almacenan en directorios separados para no
sobrescribir los experimentos previos.

## Ejecución

Entrenar las tres variantes de razonamiento:

```bash
uv run python scripts/experiments/train_qwen_reasoning_prompt.py --prompt zero_reasoning
uv run python scripts/experiments/train_qwen_reasoning_prompt.py --prompt one_reasoning
uv run python scripts/experiments/train_qwen_reasoning_prompt.py --prompt few_reasoning
```

Antes de evaluar, asegúrate de que los rankings congelados se han construido
sobre exactamente la partición reservada por `train_test_split` (20%, seed 42).
El evaluador compara preguntas, respuestas de referencia y etiquetas de
respondibilidad y aborta si difieren.

Comparación emparejada con el mismo contexto congelado:

```bash
uv run python scripts/experiments/compare_reasoning_prompts.py --conditions zero zero_reasoning
uv run python scripts/experiments/compare_reasoning_prompts.py --conditions one one_reasoning
uv run python scripts/experiments/compare_reasoning_prompts.py --conditions few few_reasoning
```

Se puede añadir `--limit 8` para un piloto técnico rápido. Los resultados del
piloto no deben presentarse como evaluación final.

## Métricas y análisis

El evaluador guarda predicciones por pregunta y EM/F1 globales, además de F1
para preguntas respondibles y no respondibles, latencia, longitud generada y
número de respuestas vacías. La unidad de comparación es la misma pregunta
reservada y el mismo primer contexto semántico congelado para ambas condiciones.

Los artefactos se guardan en `results/reasoning_prompt_comparison/`:
`predictions.csv`, `metrics.json` y `experiment_config.json`.

## Limitaciones

- La instrucción de razonamiento es una guía de análisis latente, no una
  cadena de pensamiento supervisada. No permite concluir que el modelo haya
  aprendido razonamiento explícito.
- Un solo seed y un conjunto de evaluación no permiten inferir significación
  estadística ni generalización.
- El coste, tiempo y memoria pueden variar con el hardware.

No se incluyen puntuaciones hasta ejecutar entrenamiento y evaluación. Nunca
se deben copiar métricas de experimentos anteriores como si fueran resultados
de esta comparación.
