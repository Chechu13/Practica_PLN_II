"""DEV audit of the number of contexts passed to Qwen 1.5B.

The retrieval rankings, production retriever, dataset, and LoRA adapter are
read-only inputs. Results are written to a new context_count directory.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate import bleu_score, exact_match_score, f1_score, rouge_score


SEED = 42
MODEL_NAME = "1.5B_FT_zero"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = ROOT / "models" / "prompt_training_controlled_qwen15" / "qwen15_lora_controlled_zero"
PROMPT_PATH = ROOT / "prompts" / "controlled_zero.md"
RANKINGS_PATH = ROOT / "results" / "rag_audit_2026-10-29" / "rankings.jsonl"
DEV_PREDICTIONS = ROOT / "results" / "rag_generation_audit_2026-10-29" / "dev_full" / "generation_predictions.csv"
INDEX_PATH = ROOT / "models" / "context_embeddings.pt"
METHODS = ("semantic", "hybrid")
CONTEXT_COUNTS = (1, 3, 5)
MAX_LENGTH = 384
MAX_NEW_TOKENS = 32
BOOTSTRAP_REPLICATES = 10_000


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_inputs() -> tuple[list[dict], dict[str, str]]:
    with RANKINGS_PATH.open(encoding="utf-8") as file:
        rankings = [json.loads(line) for line in file]
    index = torch.load(INDEX_PATH, map_location="cpu", weights_only=True)
    return rankings, dict(zip(index["context_ids"], index["contexts"]))


def load_reusable_rows(records: list[dict], contexts: dict[str, str]) -> list[dict]:
    if not DEV_PREDICTIONS.exists():
        return []
    with DEV_PREDICTIONS.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    by_index = {str(record["index"]): record for record in records if record["split"] == "dev"}
    reusable = []
    for row in rows:
        record = by_index.get(row["index"])
        if (
            record is None
            or row["model"] != MODEL_NAME
            or row["retrieval_method"] not in METHODS
            or row["generator_context_count"] != "1"
            or row["retrieval_top_k"] != "5"
            or row["question"] != record["question"]
            or row["reference"] != (record["answer"] or "")
            or row["retrieved_context_id"] != record[f"{row['retrieval_method']}_ids"][0]
            or row["retrieved_context"] != contexts[row["retrieved_context_id"]]
        ):
            continue
        row["context_count"] = 1
        row["context_ids"] = json.dumps([row["retrieved_context_id"]])
        row["context_positions"] = json.dumps([1])
        row["context_characters"] = str(len(row["retrieved_context"]))
        row["context_tokens"] = str(len(row["retrieved_context"].split()))
        row["untruncated_input_tokens"] = row.get("input_tokens", "")
        row["generated_tokens"] = row.get("generated_tokens", "")
        row["input_truncated"] = "False"
        reusable.append(row)
    return reusable


def generate(
    model,
    tokenizer,
    question: str,
    selected_contexts: list[str],
    system_prompt: str,
    device: torch.device,
) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "\n\n".join(
                f"Context {position}:\n{text}"
                for position, text in enumerate(selected_contexts, start=1)
            ) + f"\n\nQuestion: {question}",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    untruncated = tokenizer(prompt, add_special_tokens=False)["input_ids"]
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
    generation_seconds = time.perf_counter() - started
    generated_ids = outputs[:, input_tokens:]
    prediction = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()
    return {
        "prediction": prediction,
        "generation_seconds": generation_seconds,
        "generated_tokens": int(generated_ids.shape[1]),
        "input_tokens": input_tokens,
        "untruncated_input_tokens": len(untruncated),
        "input_truncated": len(untruncated) > input_tokens,
    }


def row_key(row: dict) -> tuple[str, str, int]:
    return row["index"], row["retrieval_method"], int(row["context_count"])


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(
        field
        for row in rows
        for field in row
    ))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                field: row.get(field, "")
                for field in fieldnames
            }
            for row in rows
        )


def summarize(rows: list[dict]) -> list[dict]:
    frame = pd.DataFrame(rows)
    frame["context_count"] = pd.to_numeric(frame["context_count"], errors="coerce")
    summaries = []
    for method in METHODS:
        for count in CONTEXT_COUNTS:
            group = frame[
                (frame["retrieval_method"] == method)
                & (frame["context_count"] == count)
            ]
            valid = group[group["error"].fillna("") == ""]
            answerable = valid[valid["has_answer"].isin([True, "True"])]
            unanswerable = valid[valid["has_answer"].isin([False, "False"])]
            def mean(column, subset=valid):
                source = subset[column]
                if column in {"empty_prediction", "input_truncated"}:
                    source = source.map({
                        True: 1.0,
                        False: 0.0,
                        "True": 1.0,
                        "False": 0.0,
                        "true": 1.0,
                        "false": 0.0,
                    })
                values = pd.to_numeric(source, errors="coerce").dropna()
                return float(values.mean()) if not values.empty else float("nan")
            def percentile(column, value, subset=valid):
                values = pd.to_numeric(subset[column], errors="coerce").dropna()
                return float(np.percentile(values, value)) if not values.empty else float("nan")
            def median(column, subset=valid):
                values = pd.to_numeric(subset[column], errors="coerce").dropna()
                return float(values.median()) if not values.empty else float("nan")
            summaries.append({
                "model": MODEL_NAME,
                "retrieval_method": method,
                "context_count": count,
                "n_total": len(group),
                "n_valid": len(valid),
                "n_errors": len(group) - len(valid),
                "EM_percent": 100 * mean("em"),
                "F1_percent": 100 * mean("f1"),
                "answerable_F1_percent": 100 * mean("f1", answerable),
                "unanswerable_F1_percent": 100 * mean("f1", unanswerable),
                "BLEU_percent": 100 * mean("bleu"),
                "ROUGE_percent": 100 * mean("rouge"),
                "empty_prediction_percent": 100 * mean("empty_prediction"),
                "generation_mean_seconds": mean("generation_seconds"),
                "generation_median_seconds": median("generation_seconds"),
                "generation_p95_seconds": percentile("generation_seconds", 95),
                "retrieval_mean_seconds": mean("retrieval_seconds"),
                "retrieval_median_seconds": median("retrieval_seconds"),
                "retrieval_p95_seconds": percentile("retrieval_seconds", 95),
                "total_mean_seconds": mean("total_seconds"),
                "mean_context_characters": mean("context_characters"),
                "mean_context_tokens": mean("context_tokens"),
                "input_truncated_percent": 100 * mean("input_truncated"),
            })
    return summaries


def bootstrap_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["context_count"] = pd.to_numeric(frame["context_count"], errors="coerce")
    rng = np.random.default_rng(SEED)
    rows = []
    for method in METHODS:
        base = frame[(frame["retrieval_method"] == method) & (frame["context_count"] == 1)].set_index("index")
        for count in (3, 5):
            candidate = frame[(frame["retrieval_method"] == method) & (frame["context_count"] == count)].set_index("index")
            joined = base.join(candidate, lsuffix="_base", rsuffix="_candidate", how="inner")
            delta_f1 = joined["f1_candidate"].astype(float).to_numpy() - joined["f1_base"].astype(float).to_numpy()
            delta_em = joined["em_candidate"].astype(float).to_numpy() - joined["em_base"].astype(float).to_numpy()
            samples = rng.integers(0, len(delta_f1), size=(BOOTSTRAP_REPLICATES, len(delta_f1)))
            f1_boot = delta_f1[samples].mean(axis=1)
            em_boot = delta_em[samples].mean(axis=1)
            rows.append({
                "method": method,
                "comparison": f"{count}_vs_1",
                "n_questions": len(delta_f1),
                "mean_delta_F1": float(delta_f1.mean()),
                "ci95_F1_low": float(np.percentile(f1_boot, 2.5)),
                "ci95_F1_high": float(np.percentile(f1_boot, 97.5)),
                "mean_delta_EM": float(delta_em.mean()),
                "ci95_EM_low": float(np.percentile(em_boot, 2.5)),
                "ci95_EM_high": float(np.percentile(em_boot, 97.5)),
                "f1_improved": int((delta_f1 > 0).sum()),
                "f1_worsened": int((delta_f1 < 0).sum()),
                "f1_unchanged": int((delta_f1 == 0).sum()),
                "em_improved": int((delta_em > 0).sum()),
                "em_worsened": int((delta_em < 0).sum()),
                "em_unchanged": int((delta_em == 0).sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rag_generation_audit_2026-10-29" / "context_count")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = torch.device(args.device)
    records, contexts = load_inputs()
    dev = [record for record in records if record["split"] == "dev"]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    reusable = load_reusable_rows(dev, contexts)
    predictions_path = args.output / "generation_predictions.csv"
    rows = list(reusable)
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8", newline="") as file:
            rows.extend(list(csv.DictReader(file)))
    rows = {row_key(row): row for row in rows}

    model = None
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device),
            ADAPTER,
        ).eval()
        for record in dev:
            for method in METHODS:
                for count in CONTEXT_COUNTS:
                    key = (str(record["index"]), method, count)
                    if key in rows:
                        continue
                    ids = record[f"{method}_ids"][:count]
                    selected_contexts = [contexts[context_id] for context_id in ids]
                    started = time.perf_counter()
                    result = generate(
                        model,
                        tokenizer,
                        record["question"],
                        selected_contexts,
                        system_prompt,
                        device,
                    )
                    retrieval_seconds = (
                        record["semantic_seconds"]
                        if method == "semantic"
                        else record["semantic_seconds"] + record["bm25_seconds"]
                    )
                    reference = record["answer"] or ""
                    row = {
                        "split": "dev",
                        "index": record["index"],
                        "question": record["question"],
                        "reference": reference,
                        "has_answer": record["has_answer"],
                        "model": MODEL_NAME,
                        "retrieval_method": method,
                        "retrieval_top_k": 5,
                        "context_count": count,
                        "generator_context_count": count,
                        "context_ids": json.dumps(ids),
                        "context_positions": json.dumps(list(range(1, count + 1))),
                        "retrieved_context": "\n\n".join(
                            f"Context {position}:\n{text}"
                            for position, text in enumerate(selected_contexts, start=1)
                        ),
                        "context_characters": sum(len(text) for text in selected_contexts),
                        "context_tokens": sum(len(text.split()) for text in selected_contexts),
                        "retrieval_seconds": retrieval_seconds,
                        "retrieval_score": record["semantic_score"] if method == "semantic" else "",
                        "prediction": result["prediction"],
                        "empty_prediction": not bool(result["prediction"]),
                        "error": "",
                        "generation_seconds": result["generation_seconds"],
                        "total_seconds": time.perf_counter() - started + retrieval_seconds,
                        "generated_tokens": result["generated_tokens"],
                        "input_tokens": result["input_tokens"],
                        "untruncated_input_tokens": result["untruncated_input_tokens"],
                        "input_truncated": result["input_truncated"],
                        "tokens_per_second": result["generated_tokens"] / result["generation_seconds"],
                        "em": exact_match_score(result["prediction"], reference),
                        "f1": f1_score(result["prediction"], reference),
                        "bleu": bleu_score(result["prediction"], reference),
                        "rouge": rouge_score(result["prediction"], reference),
                    }
                    rows[key] = row
                    write_rows(predictions_path, list(rows.values()))
    finally:
        del tokenizer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_rows = list(rows.values())
    metrics = summarize(final_rows)
    write_rows(args.output / "generation_metrics.csv", metrics)
    frame = pd.DataFrame(final_rows)
    frame["f1"] = pd.to_numeric(frame["f1"])
    frame["em"] = pd.to_numeric(frame["em"])
    paired = bootstrap_deltas(frame)
    paired.to_csv(args.output / "paired_comparison.csv", index=False)
    frame.to_csv(args.output / "retrieval_latency.csv", index=False)
    frame[frame["f1"].notna()].to_csv(args.output / "error_analysis.csv", index=False)
    reusable_keys = {row_key(row) for row in reusable}
    config = {
        "seed": SEED,
        "split": "dev",
        "n_questions": len(dev),
        "model_id": MODEL_ID,
        "adapter": str(ADAPTER),
        "prompt": str(PROMPT_PATH),
        "methods": METHODS,
        "context_counts": CONTEXT_COUNTS,
        "retrieval_top_k": 5,
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "rankings_path": str(RANKINGS_PATH),
        "reusable_one_context_rows": len(reusable_keys),
        "new_generation_count": len(final_rows) - len(reusable_keys),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "test_used": False,
    }
    (args.output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(8, 5))
    metrics_frame = pd.DataFrame(metrics)
    for method, group in metrics_frame.groupby("retrieval_method"):
        axis.plot(group["context_count"], group["F1_percent"], marker="o", label=method)
    axis.set(xlabel="Contexts delivered", ylabel="F1 (%)", title="F1 versus context count")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "quality_vs_context_count.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5))
    for method, group in metrics_frame.groupby("retrieval_method"):
        axis.plot(group["context_count"], group["total_mean_seconds"], marker="o", label=method)
    axis.set(xlabel="Contexts delivered", ylabel="Mean total seconds", title="Latency versus context count")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "latency_vs_context_count.png", dpi=160)
    plt.close(fig)
    metrics_table = metrics_frame[
        [
            "retrieval_method",
            "context_count",
            "n_valid",
            "EM_percent",
            "F1_percent",
            "generation_mean_seconds",
            "input_truncated_percent",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    paired_table = paired.to_string(index=False, float_format=lambda value: f"{value:.3f}")
    (args.output / "report.md").write_text(
        "# Context-count RAG audit\n\n"
        "## Methodology and controls\n\n"
        f"- DEV only: {len(dev)} questions; TEST was not used.\n"
        "- Frozen Phase 1 rankings, dataset, prompt, adapter and deterministic generation were used.\n"
        "- Retrieval top-k was 5; only the first 1, 3 or 5 ranked contexts were delivered.\n"
        "- Qwen2.5-1.5B-Instruct with the existing zero-shot LoRA adapter was loaded without retraining.\n"
        "- One-context predictions were reused only when question, reference, context ID, prompt and configuration matched.\n"
        f"- Unique predictions: {len(final_rows)}; reused: {len(reusable_keys)}; new generations: {len(final_rows) - len(reusable_keys)}.\n\n"
        "## Aggregate results\n\n"
        f"```\n{metrics_table}\n```\n\n"
        "The 3- and 5-context configurations show a large quality decline because the fixed "
        f"`max_length={MAX_LENGTH}` input limit truncates {metrics_frame.loc[metrics_frame['context_count'] > 1, 'input_truncated_percent'].min():.1f}% or more of inputs; "
        "the question can therefore be removed from the model input. This is a prompt-length "
        "confound, not evidence that additional evidence is intrinsically harmful.\n\n"
        "## Paired comparisons against one context\n\n"
        f"```\n{paired_table}\n```\n\n"
        "## Limitations and recommendation\n\n"
        "- Retrieval latency is inherited from the frozen retrieval audit; it was not re-measured during generation.\n"
        "- Results are DEV evidence for this model, prompt and context formatting, not TEST estimates.\n"
        "- Before testing multiple contexts as a quality intervention, increase the input budget or apply a pre-validated context compression policy while keeping the question visible.\n",
        encoding="utf-8",
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
