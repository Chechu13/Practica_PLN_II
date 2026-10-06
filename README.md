# NLP II — Grounded Question Answering

> **Proyecto en desarrollo — README temporal**
>
> Práctica obligatoria de **Procesamiento del Lenguaje Natural II** (Grado en Inteligencia Artificial).

## Descripción

Este proyecto aborda la **generación de respuestas fundamentadas** para preguntas factuales formuladas en inglés.

El sistema trabaja sobre una colección interna de contextos procedentes de Wikipedia y combina diferentes estrategias de **recuperación y generación**, incluyendo modelos Transformer implementados con PyTorch, modelos de Hugging Face y un mecanismo de *fallback* mediante búsqueda web.

La práctica requiere comparar distintas propuestas y analizar tanto la calidad de las respuestas como la recuperación de contexto y el comportamiento global del sistema. 
## Enfoque actual

La implementación se encuentra todavía en desarrollo. Actualmente se han explorado, entre otras, las siguientes líneas:

- **Preprocesado y análisis exploratorio (EDA)** del corpus.
- **Recuperación semántica de contextos** mediante embeddings con `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`.
- **Recuperación híbrida**, combinando similitud semántica con un pequeño componente léxico.
- **Fallback de recuperación**, recurriendo a fuentes externas cuando la colección interna no resulta suficiente.
- **Decoder Transformer implementado desde cero** con PyTorch.
- **Encoder-Decoder Transformer implementado desde cero** con PyTorch.
- **Modelos de Hugging Face** para comparación.
- **Fine-tuning de Qwen mediante LoRA**.
- Comparación **2×2** entre modelo base/fine-tuned y recuperación interna/fallback externo.
- Evaluación mediante **Exact Match (EM)** y **token-level F1**.

## Resultados preliminares

Los siguientes resultados corresponden a la evaluación actual disponible en el repositorio sobre **70 preguntas**:

| Enfoque | EM | F1 | N |
|---|---:|---:|---:|
| Qwen Base + Context | 15.71 | 30.85 | 70 |
| Qwen Base + Fallback | 18.57 | 36.87 | 70 |
| Qwen Fine-tuned + Context | **71.43** | **75.35** | 70 |
| Qwen Fine-tuned + Fallback | 68.57 | 73.96 | 70 |

También se han evaluado por separado otras aproximaciones, entre ellas un Transformer Encoder-Decoder y un modelo de Hugging Face.

Los resultados completos y las predicciones se almacenan en `results/`.

## Estructura del proyecto

```text
.
├── data/
│   ├── contexts.json
│   └── questions_train.json
├── models/
│   ├── context_embeddings.pt
│   └── comparison_2x2/
├── notebooks/
│   ├── EDA.ipynb
│   └── qwen_retrieval_comparison_2x2_inference_retrieval.ipynb
├── prompts/
│   ├── primer.md
│   ├── segundo.md
│   └── tercer.md
├── reports/
├── results/
├── scripts/
│   ├── create_context_embeddings.py
│   └── experiments/
└── src/
    ├── modelos/
    ├── preprocess/
    ├── retrieval/
    └── evaluate.py
```

## Instalación

El proyecto utiliza **Python 3.12** y gestiona las dependencias mediante `uv`.

### Clonado

```bash
git clone https://github.com/Chechu13/Practica_PLN_II.git
cd Practica_PLN_II
```

### Entorno

```bash
uv sync
```

El proyecto está configurado para utilizar las versiones de PyTorch compatibles con **CUDA 12.8** en Windows.

### Generación del índice de embeddings

Si es necesario regenerar los embeddings de los contextos:

```bash
uv run python scripts/create_context_embeddings.py
```

## Evaluación

Las métricas implementadas actualmente son:

- **Exact Match (EM)**: comprueba si la respuesta normalizada coincide exactamente con la referencia.
- **F1 token-level**: mide el solapamiento entre los tokens de la predicción y la referencia.

La implementación de estas métricas se encuentra en:

```text
src/evaluate.py
```

El enunciado de la práctica establece además la necesidad de justificar las métricas utilizadas y considerar la calidad de recuperación, generación y coste del sistema completo. 

## Estado actual

🚧 **En desarrollo**

La estructura experimental, los primeros recuperadores y varias propuestas de generación ya están implementados. La siguiente fase consiste en ampliar y consolidar la comparación experimental, documentar las decisiones de diseño y completar el análisis de resultados.

---

**Universidad Rey Juan Carlos — Grado en Inteligencia Artificial**  
**Procesamiento del Lenguaje Natural II**
