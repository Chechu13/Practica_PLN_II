"""Paired evaluation of baseline and reasoning-oriented prompt adapters.

The script reuses the frozen DEV rankings, retrieved contexts and generation
settings from the controlled Qwen2.5 experiment. It does not evaluate TEST.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate import exact_match_score, f1_score
from src.preprocess.preprocess import prepare_dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 512
MAX_NEW_TOKENS = 96
RANKINGS_PATH = ROOT / "results" / "rag_audit_2026-10-29" / "rankings.jsonl"
INDEX_PATH = ROOT / "models" / "context_embeddings.pt"
PROMPT_FILES = {
    "zero": "controlled_zero.md",
    "one": "controlled_one.md",
    "few": "controlled_few.md",
    "zero_reasoning": "controlled_zero_reasoning.md",
    "one_reasoning": "controlled_one_reasoning.md",
    "few_reasoning": "controlled_few_reasoning.md",
}
ADAPTER_NAMES = {
    "zero": "qwen_lora_controlled_zero",
    "one": "qwen_lora_controlled_one",
    "few": "qwen_lora_controlled_few",
    "zero_reasoning": "qwen_lora_reasoning_zero",
    "one_reasoning": "qwen_lora_reasoning_one",
    "few_reasoning": "qwen_lora_reasoning_few",
}
OUTPUT_DIR = ROOT / "results" / "reasoning_prompt_comparison"


def load_records() -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not RANKINGS_PATH.exists():
        raise FileNotFoundError(
            f"No existen los rankings congelados: {RANKINGS_PATH}. "
            "Genera primero el audit de recuperación."
        )
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe el índice de contextos: {INDEX_PATH}")
    with RANKINGS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    index = torch.load(INDEX_PATH, map_location="cpu", weights_only=True)
    return records, dict(zip(index["context_ids"], index["contexts"]))


def load_prompt(prompt_name: str, result_root: Path) -> str:
    saved_prompt = result_root / prompt_name / "training_prompt.txt"
    # Training prompts for one/few-shot contain examples selected from train.
    if saved_prompt.exists():
        return saved_prompt.read_text(encoding="utf-8").strip()
    if prompt_name.endswith("_reasoning") and prompt_name == "zero_reasoning":
        return (ROOT / "prompts" / PROMPT_FILES[prompt_name]).read_text(
            encoding="utf-8"
        ).strip()
    raise FileNotFoundError(
        f"No se encuentra el prompt exacto guardado por el entrenamiento: {saved_prompt}"
    )


def load_model(with_adapter: bool, adapter_path: Path, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device.type == "cuda" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    if with_adapter:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def generate(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    context: str,
    system_prompt: str,
    device: torch.device,
) -> tuple[str, float, int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context or 'No context is available.'}\n\n"
                f"Question: {record['question']}"
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH
    )
    input_tokens = int(inputs["input_ids"].shape[1])
    inputs = {key: value.to(device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    new_tokens = output[:, input_tokens:]
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return text, elapsed, int(new_tokens.shape[1])


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        answerable = [row for row in group if row["has_answer"]]
        unanswerable = [row for row in group if not row["has_answer"]]
        summaries.append({
            "condition": condition,
            "n": len(group),
            "em_percent": 100 * float(np.mean([row["em"] for row in group])),
            "f1_percent": 100 * float(np.mean([row["f1"] for row in group])),
            "answerable_f1_percent": (
                100 * float(np.mean([row["f1"] for row in answerable]))
                if answerable else 0.0
            ),
            "unanswerable_f1_percent": (
                100 * float(np.mean([row["f1"] for row in unanswerable]))
                if unanswerable else 0.0
            ),
            "mean_generation_seconds": float(
                np.mean([row["generation_seconds"] for row in group])
            ),
            "mean_generated_tokens": float(
                np.mean([row["generated_tokens"] for row in group])
            ),
            "empty_predictions": sum(not row["prediction"] for row in group),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara prompts controlados estándar y orientados al razonamiento."
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(PROMPT_FILES),
        default=["zero", "zero_reasoning"],
    )
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "prompt_training_controlled")
    parser.add_argument("--reasoning-results-root", type=Path, default=ROOT / "results" / "prompt_training_reasoning")
    parser.add_argument("--limit", type=int, default=None, help="Límite opcional para un piloto; no usar para resultados finales.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit debe ser mayor que cero.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA fue solicitado, pero no está disponible.")
    records, contexts = load_records()
    records = [record for record in records if record["split"] == args.split]
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError(f"No hay registros para el split {args.split}.")

    # Validate that the frozen DEV rankings correspond to the same dataset
    # split used for training before reporting paired metrics.
    source_dataset = prepare_dataset(
        str(ROOT / "data" / "questions_train.json"),
        str(ROOT / "data" / "contexts.json"),
    )
    _, expected_test = train_test_split(
        source_dataset, test_size=0.2, random_state=42
    )
    expected_questions = {item["question"] for item in expected_test}
    ranking_questions = {record["question"] for record in records}
    if ranking_questions != expected_questions:
        raise ValueError(
            "Los rankings DEV no coinciden con la partición test (20%, seed=42) "
            "del entrenamiento. No se puede presentar esta evaluación como "
            "una comparación emparejada válida. Genera rankings sobre la "
            "partición reservada exacta o evalúa ambos adaptadores con el "
            "mismo conjunto congelado derivado del split."
        )

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[dict[str, Any]] = []

    for condition in args.conditions:
        reasoning = condition.endswith("_reasoning")
        adapter_path = ROOT / "models" / "prompt_training_reasoning" / ADAPTER_NAMES[condition] if reasoning else (
            ROOT / "models" / "prompt_training_controlled" / ADAPTER_NAMES[condition]
        )
        if not adapter_path.exists():
            raise FileNotFoundError(
                f"Falta el adaptador para '{condition}': {adapter_path}. "
                "Entrena primero las condiciones indicadas."
            )
        prompt_root = (
            args.reasoning_results_root if reasoning else args.results_root
        )
        system_prompt = load_prompt(condition, prompt_root)
        model = None
        try:
            model = load_model(True, adapter_path, device)
            for record in records:
                ranking = record.get("semantic_ids", [])
                context_id = ranking[0] if ranking else "none"
                context = contexts.get(context_id, "")
                prediction, generation_seconds, generated_tokens = generate(
                    model, tokenizer, record, context, system_prompt, device
                )
                reference = record.get("answer") or ""
                rows.append({
                    "split": record["split"],
                    "index": record["index"],
                    "condition": condition,
                    "question": record["question"],
                    "reference": reference,
                    "prediction": prediction,
                    "has_answer": bool(record["has_answer"]),
                    "context_id": context_id,
                    "generation_seconds": generation_seconds,
                    "generated_tokens": generated_tokens,
                    "em": exact_match_score(prediction, reference),
                    "f1": f1_score(prediction, reference),
                })
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions_path = args.output / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = summarize(rows)
    (args.output / "metrics.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison = {
        "model_id": MODEL_ID,
        "split": args.split,
        "n_questions": len(records),
        "conditions": args.conditions,
        "same_frozen_contexts": True,
        "context_source": str(RANKINGS_PATH),
        "test_used_for_selection": False,
        "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False},
    }
    (args.output / "experiment_config.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
