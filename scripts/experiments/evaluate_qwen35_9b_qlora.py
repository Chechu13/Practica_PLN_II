"""Paired DEV evaluation for the Qwen3.5-9B QLoRA adapters.

The base model and adapter are loaded with the same 4-bit configuration and
are evaluated on the frozen DEV rankings used by generation_audit.py.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluate import exact_match_score, f1_score

MODEL_ID = "Qwen/Qwen3.5-9B"
SEED = 42
MAX_LENGTH = 512
MAX_NEW_TOKENS = 96
RANKINGS_PATH = ROOT / "results" / "rag_audit_2026-10-29" / "rankings.jsonl"
INDEX_PATH = ROOT / "models" / "context_embeddings.pt"
PROMPTS_DIR = ROOT / "prompts"
ADAPTER_ROOT = ROOT / "models" / "prompt_training_controlled_qwen35_9b"
OUTPUT_ROOT = ROOT / "results" / "prompt_training_controlled_qwen35_9b"
PROMPT_FILES = {
    "zero": "controlled_zero.md",
    "one": "controlled_one.md",
    "few": "controlled_few.md",
}
ADAPTER_NAMES = {
    "zero": "qwen35_9b_lora_controlled_zero",
    "one": "qwen35_9b_lora_controlled_one",
    "few": "qwen35_9b_lora_controlled_few",
}


def load_records() -> tuple[list[dict], dict[str, str]]:
    with RANKINGS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    index = torch.load(INDEX_PATH, map_location="cpu", weights_only=True)
    return records, dict(zip(index["context_ids"], index["contexts"]))


def load_quantized_model() -> tuple[Any, Any, float]:
    try:
        import bitsandbytes as bnb
        if not getattr(bnb, "__version__", None):
            raise ImportError("bitsandbytes no expone una versión válida")
    except ImportError as exc:
        raise RuntimeError(
            "La evaluación QLoRA requiere bitsandbytes. Ejecuta `uv sync`."
        ) from exc
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization,
        torch_dtype=dtype,
        device_map={"": 0},
    )
    model.eval()
    return model, tokenizer, time.perf_counter() - started


def generate(
    model: Any,
    tokenizer: Any,
    record: dict,
    context: str,
    system_prompt: str,
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
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    input_tokens = int(inputs["input_ids"].shape[1])
    inputs = {key: value.cuda() for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    generated = outputs[:, input_tokens:]
    return (
        tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip(),
        elapsed,
        int(generated.shape[1]),
    )


def evaluate_model(
    model_name: str,
    model: Any,
    tokenizer: Any,
    records: list[dict],
    contexts: dict[str, str],
    system_prompt: str,
    prompt_strategy: str,
    load_seconds: float,
) -> list[dict]:
    rows = []
    for record in records:
        ranking = record.get("semantic_ids", [])
        context_id = ranking[0] if ranking else "none"
        prediction, generation_seconds, generated_tokens = generate(
            model, tokenizer, record, contexts.get(context_id, ""), system_prompt
        )
        reference = record.get("answer") or ""
        rows.append(
            {
                "split": record["split"],
                "index": record["index"],
                "model": model_name,
                "prompt_strategy": prompt_strategy,
                "question": record["question"],
                "reference": reference,
                "prediction": prediction,
                "has_answer": record["has_answer"],
                "empty_prediction": not bool(prediction),
                "retrieved_context_id": context_id,
                "generation_seconds": generation_seconds,
                "generated_tokens": generated_tokens,
                "tokens_per_second": (
                    generated_tokens / generation_seconds
                    if generation_seconds
                    else 0.0
                ),
                "model_load_seconds": load_seconds,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "em": exact_match_score(prediction, reference),
                "f1": f1_score(prediction, reference),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for model_name in ("base", "qlora"):
        model_rows = [row for row in rows if row["model"] == model_name]
        if not model_rows:
            continue
        latencies = np.array([row["generation_seconds"] for row in model_rows])
        summaries.append(
            {
                "model": model_name,
                "em_percent": 100 * np.mean([row["em"] for row in model_rows]),
                "f1_percent": 100 * np.mean([row["f1"] for row in model_rows]),
                "latency_mean_seconds": float(np.mean(latencies)),
                "latency_median_seconds": float(np.median(latencies)),
                "tokens_per_second": float(
                    np.mean([row["tokens_per_second"] for row in model_rows])
                ),
                "empty_predictions": sum(
                    row["empty_prediction"] for row in model_rows
                ),
                "peak_vram_gib": max(
                    row["peak_vram_gib"] for row in model_rows
                ),
                "model_load_seconds": model_rows[0]["model_load_seconds"],
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluación emparejada DEV de Qwen3.5-9B base y QLoRA."
    )
    parser.add_argument("--prompt", choices=("zero", "one", "few"), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("La evaluación QLoRA requiere CUDA.")
    adapter = ADAPTER_ROOT / ADAPTER_NAMES[args.prompt]
    if not adapter.exists():
        raise FileNotFoundError(f"No existe el adaptador: {adapter}")

    records, contexts = load_records()
    records = [record for record in records if record["split"] == "dev"]
    prompt_path = OUTPUT_ROOT / args.prompt / "training_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(
            "No existe el prompt exacto guardado por el entrenamiento: "
            f"{prompt_path}"
        )
    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    output_dir = OUTPUT_ROOT / args.prompt / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_name in ("base", "qlora"):
        model = None
        tokenizer = None
        try:
            torch.cuda.reset_peak_memory_stats()
            model, tokenizer, load_seconds = load_quantized_model()
            if model_name == "qlora":
                adapter_started = time.perf_counter()
                model = PeftModel.from_pretrained(model, adapter)
                model.eval()
                load_seconds += time.perf_counter() - adapter_started
            rows.extend(
                evaluate_model(
                    model_name,
                    model,
                    tokenizer,
                    records,
                    contexts,
                    system_prompt,
                    args.prompt,
                    load_seconds,
                )
            )
        finally:
            del tokenizer
            del model
            gc.collect()
            torch.cuda.empty_cache()

    write_rows(output_dir / "predictions.csv", rows)
    summaries = summarize(rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "adapter": str(adapter),
                "prompt_strategy": args.prompt,
                "split": "dev",
                "split_source": str(RANKINGS_PATH),
                "same_frozen_semantic_context": True,
                "quantization": {
                    "load_in_4bit": True,
                    "quant_type": "nf4",
                    "double_quant": True,
                },
                "max_length": MAX_LENGTH,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
                "test_used_for_selection": False,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
