import gc
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

sys.path.append(PROJECT_ROOT)

from src.evaluate import evaluate_predictions
from src.preprocess.preprocess import prepare_dataset
from src.retrieval.context_retriever import ContextRetriever


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MAX_LENGTH = 384
MAX_NEW_TOKENS = 32

CONTEXT_INDEX_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "context_embeddings.pt"
)

ADAPTER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "comparison_2x2",
    "qwen_lora_context"
)

PROMPTS_DIR = os.path.join(
    PROJECT_ROOT,
    "prompts"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "prompt_comparison"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PROMPTS
# ============================================================

def load_prompt(name):
    path = os.path.join(
        PROMPTS_DIR,
        name
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read().strip()


def build_system_prompt(prompt_name, examples):
    """
    Construye el prompt de sistema.

    Zero-shot:
        no necesita ejemplos.

    One-shot:
        usa examples[0].

    Few-shot:
        usa examples[0:4].
    """

    template = load_prompt(prompt_name)

    if prompt_name == "primer.md":
        return template

    if prompt_name == "one_shot.md":
        example = examples[0]

        return template.format(
            example_context=example["context"],
            example_question=example["question"],
            example_answer=example["answer"],
        )

    if prompt_name == "few_shot.md":
        selected = examples[:4]

        values = {}

        for i, example in enumerate(
            selected,
            start=1
        ):
            values[f"example_{i}_context"] = example["context"]
            values[f"example_{i}_question"] = example["question"]
            values[f"example_{i}_answer"] = example["answer"]

        return template.format(**values)

    raise ValueError(
        f"Prompt desconocido: {prompt_name}"
    )


# ============================================================
# EXAMPLES
# ============================================================

def select_examples(train_data):
    """
    Selecciona ejemplos exclusivamente del conjunto
    de entrenamiento.

    Intentamos obtener ejemplos variados:
      - respuesta corta
      - persona
      - respuesta larga/lista
      - pregunta sin respuesta
    """

    answerable = [
        item
        for item in train_data
        if item["has_answer"] and item["context"]
    ]

    unanswerable = [
        item
        for item in train_data
        if not item["has_answer"]
    ]

    # Respuesta corta
    short = min(
        answerable,
        key=lambda x: len(
            x["answer"].split()
        )
    )

    # Persona / respuesta de longitud media
    medium_candidates = [
        x
        for x in answerable
        if 1 <= len(x["answer"].split()) <= 5
        and x != short
    ]

    medium = (
        medium_candidates[0]
        if medium_candidates
        else answerable[1]
    )

    # Respuesta larga
    long_answer = max(
        [
            x
            for x in answerable
            if x != short and x != medium
        ],
        key=lambda x: len(
            x["answer"].split()
        )
    )

    examples = [
        short,
        medium,
        long_answer,
    ]

    # Añadimos una pregunta sin respuesta si existe.
    if unanswerable:
        examples.append(
            unanswerable[0]
        )

    return examples


# ============================================================
# MODEL
# ============================================================

def load_base_model(tokenizer):
    print(
        f"\nCargando modelo base: {MODEL_ID}"
    )

    dtype = (
        torch.bfloat16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    model = model.to(DEVICE)
    model.eval()

    return model


def load_finetuned_model(tokenizer):
    print(
        "\nCargando modelo fine-tuned..."
    )

    dtype = (
        torch.bfloat16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# GENERATION
# ============================================================

def make_messages(
    item,
    system_prompt
):
    context = (
        item["context"]
        or "No context is available."
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {item['question']}"
            ),
        },
    ]


def generate_answer(
    model,
    tokenizer,
    item,
    system_prompt,
):
    messages = make_messages(
        item,
        system_prompt,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[
        :,
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return answer.strip()


# ============================================================
# RETRIEVAL
# ============================================================

def resolve_test_contexts(
    test_data,
    retriever
):
    """
    Recuperamos los contextos exclusivamente del
    índice local.

    No se utiliza búsqueda web.
    Todos los prompts reciben exactamente los
    mismos contextos recuperados.
    """

    resolved = []

    for item in test_data:
        retrieved = retriever.retrieve(
            item["question"]
        )

        resolved.append({
            **item,
            "context": retrieved["context"],
            "retrieved_context_id": retrieved["context_id"],
            "retrieval_score": retrieved["score"],
        })

    return resolved


# ============================================================
# EVALUATION
# ============================================================

def evaluate_condition(
    model,
    tokenizer,
    test_data,
    prompt_name,
    examples,
    model_name,
):
    print(
        f"\n[{model_name}] "
        f"Prompt: {prompt_name}"
    )

    system_prompt = build_system_prompt(
        prompt_name,
        examples,
    )

    results = []

    for i, item in enumerate(
        test_data,
        start=1
    ):
        prediction = generate_answer(
            model,
            tokenizer,
            item,
            system_prompt,
        )

        results.append({
            "question": item["question"],
            "prediction": prediction,
            "reference": item["answer"],
            "has_answer": item["has_answer"],
            "retrieved_context_id": item.get(
                "retrieved_context_id"
            ),
            "retrieval_score": item.get(
                "retrieval_score"
            ),
        })

        if i % 10 == 0:
            print(
                f"  {i}/{len(test_data)}"
            )

    answerable = [
        x
        for x in results
        if x["has_answer"]
    ]

    unanswerable = [
        x
        for x in results
        if not x["has_answer"]
    ]

    metrics = {
        "overall": evaluate_predictions(
            results
        ),
        "answerable": evaluate_predictions(
            answerable
        ),
        "unanswerable": evaluate_predictions(
            unanswerable
        ),
    }

    print(
        f"  EM: "
        f"{metrics['overall']['exact_match']}% | "
        f"F1: "
        f"{metrics['overall']['f1_score']}%"
    )

    print(
        f"  Answerable F1: "
        f"{metrics['answerable']['f1_score']}%"
    )

    print(
        f"  Unanswerable F1: "
        f"{metrics['unanswerable']['f1_score']}%"
    )

    return {
        "model": model_name,
        "prompt": prompt_name.replace(
            ".md",
            ""
        ),
        "metrics": metrics,
        "predictions": results,
    }


# ============================================================
# CLEANUP
# ============================================================

def unload_model(model):
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print(
        f"Dispositivo: {DEVICE}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataset = prepare_dataset(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "questions_train.json",
        ),
        os.path.join(
            PROJECT_ROOT,
            "data",
            "contexts.json",
        ),
    )

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=42,
    )

    print(
        f"Train: {len(train_data)} | "
        f"Test: {len(test_data)}"
    )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    examples = select_examples(
        train_data
    )

    print(
        "\nEjemplos utilizados:"
    )

    for i, example in enumerate(
        examples,
        start=1
    ):
        print(
            f"\nExample {i}:"
        )

        print(
            f"Q: {example['question']}"
        )

        print(
            f"A: {example['answer']}"
        )

    # --------------------------------------------------------
    # RETRIEVER LOCAL
    # --------------------------------------------------------

    retriever = ContextRetriever(
        CONTEXT_INDEX_PATH,
        device=DEVICE,
    )

    print(
        "\nResolviendo contextos del test "
        "usando únicamente el índice local..."
    )

    resolved_test_data = resolve_test_contexts(
        test_data,
        retriever,
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # PROMPTS
    # --------------------------------------------------------

    prompts = [
        "primer.md",
        "one_shot.md",
        "few_shot.md",
    ]

    all_results = []

    # ========================================================
    # BASE MODEL
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "MODELO BASE"
    )

    print(
        "=" * 60
    )

    base_model = load_base_model(
        tokenizer
    )

    for prompt_name in prompts:

        result = evaluate_condition(
            base_model,
            tokenizer,
            resolved_test_data,
            prompt_name,
            examples,
            "base",
        )

        all_results.append(
            result
        )

    unload_model(
        base_model
    )

    # ========================================================
    # FINE-TUNED MODEL
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "MODELO FINE-TUNED"
    )

    print(
        "=" * 60
    )

    print(
        f"Adapter: {ADAPTER_PATH}"
    )

    finetuned_model = load_finetuned_model(
        tokenizer
    )

    for prompt_name in prompts:

        result = evaluate_condition(
            finetuned_model,
            tokenizer,
            resolved_test_data,
            prompt_name,
            examples,
            "fine_tuned",
        )

        all_results.append(
            result
        )

    unload_model(
        finetuned_model
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results_path = os.path.join(
        RESULTS_DIR,
        "results.json",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = []

    for result in all_results:

        metrics = result["metrics"]

        summary.append({
            "model": result["model"],
            "prompt": result["prompt"],
            "EM": metrics[
                "overall"
            ]["exact_match"],
            "F1": metrics[
                "overall"
            ]["f1_score"],
            "Answerable_F1": metrics[
                "answerable"
            ]["f1_score"],
            "Unanswerable_F1": metrics[
                "unanswerable"
            ]["f1_score"],
        })

    summary_path = os.path.join(
        RESULTS_DIR,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "RESULTADOS"
    )

    print(
        "=" * 60
    )

    for row in summary:

        print(
            f"{row['model']:12} | "
            f"{row['prompt']:10} | "
            f"EM={row['EM']:6.2f} | "
            f"F1={row['F1']:6.2f} | "
            f"Ans={row['Answerable_F1']:6.2f} | "
            f"Unans={row['Unanswerable_F1']:6.2f}"
        )

    print(
        f"\nResultados guardados en:\n"
        f"{results_path}"
    )

    print(
        f"\nResumen guardado en:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()