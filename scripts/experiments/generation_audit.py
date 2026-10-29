"""Controlled generation audit using frozen retrieval rankings.

This script never changes production retrieval code, model adapters, or prior
results. It consumes rankings produced by audit_retrieval.py and writes to a
new experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate import bleu_score, exact_match_score, f1_score, rouge_score


SEED = 42
MODEL_CONFIGS = {
    "7B_FT_zero": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter": ROOT / "models" / "prompt_training_controlled" / "qwen_lora_controlled_zero",
    },
    "1.5B_FT_zero": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter": ROOT / "models" / "prompt_training_controlled_qwen15" / "qwen15_lora_controlled_zero",
    },
}
METHODS = ("semantic", "bm25", "hybrid")
PROMPT_PATH = ROOT / "prompts" / "controlled_zero.md"
RANKINGS_PATH = ROOT / "results" / "rag_audit_2026-10-29" / "rankings.jsonl"
INDEX_PATH = ROOT / "models" / "context_embeddings.pt"
MAX_LENGTH = 384
MAX_NEW_TOKENS = 32


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_rankings() -> tuple[list[dict], dict[str, str]]:
    with RANKINGS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file]
    index = torch.load(INDEX_PATH, map_location="cpu", weights_only=True)
    contexts = dict(zip(index["context_ids"], index["contexts"]))
    return records, contexts


def pilot_records(records: list[dict], size: int) -> list[dict]:
    dev = [record for record in records if record["split"] == "dev"]
    selected: list[dict] = []
    for record in dev:
        if record["has_answer"] and len(selected) < max(1, size // 2):
            selected.append(record)
    for record in dev:
        if not record["has_answer"] and len(selected) < size:
            selected.append(record)
    for record in dev:
        if len(selected) >= size:
            break
        if record not in selected:
            selected.append(record)
    return selected[:size]


def load_model(model_id: str, adapter_path: Path, device: torch.device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model


def generate(
    model,
    tokenizer,
    question: str,
    context: str,
    system_prompt: str,
    device: torch.device,
) -> tuple[str, float, int, int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Context:\n{context or 'No context is available.'}\n\nQuestion: {question}",
        },
    ]
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
    input_tokens = int(inputs["input_ids"].shape[1])
    inputs = {key: value.to(device) for key, value in inputs.items()}
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
    generated_ids = outputs[:, input_tokens:]
    answer = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()
    return answer, elapsed, int(generated_ids.shape[1]), input_tokens


def evaluate_record(
    model,
    tokenizer,
    record: dict,
    contexts: dict[str, str],
    method: str,
    device: torch.device,
    system_prompt: str,
) -> dict:
    ranking = record[f"{method}_ids"][:5]
    context_id = ranking[0] if ranking else "none"
    context = contexts.get(context_id, "")
    retrieval_seconds = (
        record["semantic_seconds"]
        if method == "semantic"
        else record["bm25_seconds"]
        if method == "bm25"
        else record["semantic_seconds"] + record["bm25_seconds"]
    )
    started = time.perf_counter()
    prediction, generation_seconds, generated_tokens, input_tokens = generate(
        model,
        tokenizer,
        record["question"],
        context,
        system_prompt,
        device,
    )
    total_seconds = time.perf_counter() - started + retrieval_seconds
    reference = record["answer"] or ""
    return {
        "split": record["split"],
        "index": record["index"],
        "question": record["question"],
        "reference": reference,
        "has_answer": record["has_answer"],
        "model": "",
        "retrieval_method": method,
        "retrieval_top_k": 5,
        "generator_context_count": 1,
        "retrieved_context_id": context_id,
        "retrieved_context": context,
        "retrieval_score": record["semantic_score"] if method == "semantic" else None,
        "prediction": prediction,
        "empty_prediction": not bool(prediction.strip()),
        "error": "",
        "retrieval_seconds": retrieval_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "generated_tokens": generated_tokens,
        "input_tokens": input_tokens,
        "tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
        "em": exact_match_score(prediction, reference),
        "f1": f1_score(prediction, reference),
        "bleu": bleu_score(prediction, reference),
        "rouge": rouge_score(prediction, reference),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    unique = {}
    for row in rows:
        key = (row["split"], row["index"], row["model"], row["retrieval_method"])
        unique[key] = row
    return list(unique.values())


def compatible_pilot_rows(
    source: Path,
    selected: list[dict],
    contexts: dict[str, str],
) -> list[dict]:
    if not source.exists():
        return []
    with source.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    selected_by_index = {str(record["index"]): record for record in selected}
    compatible = []
    seen = set()
    for row in rows:
        key = (row["split"], row["index"], row["model"], row["retrieval_method"])
        record = selected_by_index.get(row["index"])
        if (
            record is None
            or row["split"] != "dev"
            or row["retrieval_method"] not in METHODS
            or row["retrieval_top_k"] != "5"
            or row["retrieved_context_id"] != record[f"{row['retrieval_method']}_ids"][0]
            or row["question"] != record["question"]
            or row["reference"] != (record["answer"] or "")
            or row["model"] not in MODEL_CONFIGS
            or key in seen
        ):
            continue
        row["generator_context_count"] = "1"
        row["retrieved_context"] = contexts[row["retrieved_context_id"]]
        seen.add(key)
        compatible.append(row)
    return compatible


def write_analysis(rows: list[dict], output: Path) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    for column in (
        "f1",
        "total_seconds",
        "retrieval_seconds",
        "generation_seconds",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.to_csv(output / "retrieval_latency.csv", index=False)
    examples = []
    for index, group in frame.groupby("index"):
        if group["f1"].astype(float).nunique() > 1:
            best = group.loc[group["f1"].astype(float).idxmax()]
            worst = group.loc[group["f1"].astype(float).idxmin()]
            examples.append({
                "index": index,
                "question": best["question"],
                "best_model": best["model"],
                "best_method": best["retrieval_method"],
                "best_f1": best["f1"],
                "worst_method": worst["retrieval_method"],
                "worst_f1": worst["f1"],
                "best_context_id": best["retrieved_context_id"],
                "worst_context_id": worst["retrieved_context_id"],
                "best_prediction": best["prediction"],
                "worst_prediction": worst["prediction"],
                "diagnosis": "Ranking methods changed the generated result; inspect context evidence manually.",
            })
    pd.DataFrame(examples).to_csv(output / "error_analysis.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 5))
    for method, group in frame.groupby("retrieval_method"):
        axis.scatter(
            group["total_seconds"].astype(float),
            group["f1"].astype(float) * 100,
            label=method,
            alpha=0.7,
        )
    axis.set(xlabel="Total seconds per question", ylabel="F1 (%)", title="Quality versus total latency")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "quality_vs_latency.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = frame.groupby("retrieval_method")[["retrieval_seconds", "generation_seconds"]].mean()
    grouped.plot.bar(ax=axis)
    axis.set_ylabel("Mean seconds per question")
    axis.set_title("Retrieval versus generation latency")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "retrieval_vs_generation.png", dpi=160)
    plt.close(figure)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    groups = sorted({(row["model"], row["retrieval_method"]) for row in rows})
    for model, method in groups:
        group = [row for row in rows if row["model"] == model and row["retrieval_method"] == method]
        successes = [row for row in group if not row.get("error")]
        answerable = [row for row in successes if row["has_answer"] in (True, "True")]
        unanswerable = [row for row in successes if row["has_answer"] in (False, "False")]
        f1_values = [float(row["f1"]) for row in successes]
        em_values = [float(row["em"]) for row in successes]
        bleu_values = [float(row["bleu"]) for row in successes]
        rouge_values = [float(row["rouge"]) for row in successes]
        generation_values = [float(row["generation_seconds"]) for row in successes]
        retrieval_values = [float(row["retrieval_seconds"]) for row in successes]
        total_values = [float(row["total_seconds"]) for row in successes]
        token_rate_values = [float(row["tokens_per_second"]) for row in successes]
        empty_values = [
            row["empty_prediction"] in (True, "True")
            for row in successes
        ]
        summaries.append({
            "model": model,
            "retrieval_method": method,
            "n_total": len(group),
            "n_success": len(successes),
            "n_errors": len(group) - len(successes),
            "EM_percent": 100 * np.mean(em_values) if successes else 0.0,
            "F1_percent": 100 * np.mean(f1_values) if successes else 0.0,
            "answerable_F1_percent": 100 * np.mean(
                [float(row["f1"]) for row in answerable]
            ) if answerable else 0.0,
            "unanswerable_F1_percent": 100 * np.mean(
                [float(row["f1"]) for row in unanswerable]
            ) if unanswerable else 0.0,
            "BLEU_percent": 100 * np.mean(bleu_values) if successes else 0.0,
            "ROUGE_percent": 100 * np.mean(rouge_values) if successes else 0.0,
            "empty_prediction_percent": 100 * np.mean(empty_values) if successes else 0.0,
            "generation_mean_seconds": np.mean(generation_values) if successes else 0.0,
            "generation_p95_seconds": np.percentile(generation_values, 95) if successes else 0.0,
            "retrieval_mean_seconds": np.mean(retrieval_values) if successes else 0.0,
            "retrieval_p95_seconds": np.percentile(retrieval_values, 95) if successes else 0.0,
            "total_mean_seconds": np.mean(total_values) if successes else 0.0,
            "tokens_per_second": np.mean(token_rate_values) if successes else 0.0,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rag_generation_audit_2026-10-29")
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--split", choices=("pilot", "dev", "test"), default="pilot")
    parser.add_argument("--pilot-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = torch.device(args.device)
    records, contexts = load_rankings()
    if args.split == "pilot":
        selected = pilot_records(records, args.pilot_size)
    else:
        selected = [record for record in records if record["split"] == args.split]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    predictions_path = args.output / "generation_predictions.csv"
    existing = compatible_pilot_rows(args.reuse, selected, contexts) if args.reuse else []
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8", newline="") as file:
            existing.extend(list(csv.DictReader(file)))
    existing = deduplicate_rows(existing)
    done = {
        (row["split"], row["index"], row["model"], row["retrieval_method"])
        for row in existing
    }
    rows = existing
    errors = []
    for model_name, config in MODEL_CONFIGS.items():
        if not config["adapter"].exists():
            raise FileNotFoundError(f"Adapter not found: {config['adapter']}")
        model = None
        tokenizer = None
        pending = [
            record
            for record in selected
            if any(
                (record["split"], str(record["index"]), model_name, method)
                not in done
                for method in METHODS
            )
        ]
        if not pending:
            continue
        try:
            load_started = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = load_model(config["model_id"], config["adapter"], device)
            load_seconds = time.perf_counter() - load_started
            for record in selected:
                for method in METHODS:
                    key = (record["split"], str(record["index"]), model_name, method)
                    if key in done:
                        continue
                    try:
                        row = evaluate_record(
                            model, tokenizer, record, contexts, method, device, system_prompt
                        )
                        row["model"] = model_name
                        row["model_load_seconds"] = load_seconds
                        rows.append(row)
                        rows = deduplicate_rows(rows)
                        write_rows(predictions_path, rows)
                        print(
                            f"{model_name} {method} {record['index']} "
                            f"F1={row['f1']:.3f} time={row['generation_seconds']:.2f}s",
                            flush=True,
                        )
                    except Exception as exc:
                        errors.append({
                            "model": model_name,
                            "retrieval_method": method,
                            "index": record["index"],
                            "error": repr(exc),
                        })
        finally:
            del tokenizer
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_rows(args.output / "generation_metrics.csv", summarize(rows))
    config = {
        "phase": "pilot" if args.split == "pilot" else args.split,
        "seed": SEED,
        "split_source": "results/rag_audit_2026-10-29/rankings.jsonl",
        "selected_records": [record["index"] for record in selected],
        "models": {
            name: {"model_id": value["model_id"], "adapter": str(value["adapter"])}
            for name, value in MODEL_CONFIGS.items()
        },
        "methods": METHODS,
        "retrieval_top_k": 5,
        "generator_context_count": 1,
        "contexts_sent_to_generator": 1,
        "prompt": str(PROMPT_PATH),
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "device": str(device),
        "index_path": str(INDEX_PATH),
        "model_load_excluded_from_per_question_latency": True,
    }
    (args.output / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_analysis(rows, args.output)
    if errors:
        write_rows(args.output / "execution_errors.csv", errors)
    summaries = summarize(rows)
    report = [
        "# RAG generation audit",
        "",
        f"- Phase: `{config['phase']}`.",
        f"- Rows completed: {len(rows)}.",
        f"- Unique DEV questions selected: {len(selected)}.",
        f"- Compatible pilot rows reused: {len(compatible_pilot_rows(args.reuse, selected, contexts)) if args.reuse else 0}.",
        f"- Retrieval candidates: `{config['retrieval_top_k']}`; generator contexts: `{config['generator_context_count']}`.",
        "- Test was not evaluated or used for selection.",
        "- Model loading is excluded from per-question latency.",
        "- Retrieval latency is reused from frozen Phase 1 rankings; the generator receives only ranking position 1.",
        "- Error analysis labels are diagnostic candidates requiring context inspection, not proof of causality.",
        "",
        "## Metrics",
        "",
        "| Model | Method | EM (%) | F1 (%) | Answerable F1 (%) | Unanswerable F1 (%) | Retrieval mean (s) | Generation mean (s) | Total mean (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        report.append(
            f"| {summary['model']} | {summary['retrieval_method']} | "
            f"{summary['EM_percent']:.2f} | {summary['F1_percent']:.2f} | "
            f"{summary['answerable_F1_percent']:.2f} | {summary['unanswerable_F1_percent']:.2f} | "
            f"{summary['retrieval_mean_seconds']:.4f} | {summary['generation_mean_seconds']:.4f} | "
            f"{summary['total_mean_seconds']:.4f} |"
        )
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "errors": errors, "config": config}, indent=2))


if __name__ == "__main__":
    main()
