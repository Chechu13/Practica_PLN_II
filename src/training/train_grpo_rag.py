"""GRPO con ContextRetriever y particiones FIT/DEV/TEST congeladas.

Compatible con TRL 1.17.0: limita los prompts antes de GRPOConfig.
Solo FIT se utiliza para entrenar. DEV y TEST quedan reservados.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluate import f1_score
from src.retrieval.context_retriever import ContextRetriever


SEED = 42
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
INDEX_PATH = ROOT / "models" / "context_embeddings.pt"
RANKINGS_PATH = ROOT / "results" / "rag_audit_2026-10-29" / "rankings.jsonl"
PROMPT_PATH = ROOT / "prompts" / "controlled_zero.md"

MAX_PROMPT_LENGTH = 384
MAX_COMPLETION_LENGTH = 64
RETRIEVAL_TOP_K = 1


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def completion_text(completion) -> str:
    """Admite completions en formato string o chat."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        content = completion.get("content", "")
        return content.strip() if isinstance(content, str) else str(content).strip()
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    return str(completion).strip()


def reward_answer_f1(completions, reference, **kwargs):
    """Reward basado en el F1 token-level del evaluador del proyecto."""
    if len(completions) != len(reference):
        raise ValueError(
            f"Número de completions ({len(completions)}) distinto "
            f"del de referencias ({len(reference)})."
        )
    return [
        float(f1_score(completion_text(pred), str(ref or "")))
        for pred, ref in zip(completions, reference)
    ]


def load_frozen_splits():
    if not RANKINGS_PATH.exists():
        raise FileNotFoundError(
            f"No existe {RANKINGS_PATH}. Ejecuta primero "
            "`uv run python scripts/experiments/audit_retrieval.py`."
        )

    records = [
        json.loads(line)
        for line in RANKINGS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = {
        name: [row for row in records if row.get("split") == name]
        for name in ("fit", "dev", "test")
    }
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"La partición {name} está vacía.")
    indices = [row["index"] for row in records]
    if len(indices) != len(set(indices)):
        raise ValueError("Hay índices duplicados en rankings.jsonl.")
    return splits["fit"], splits["dev"], splits["test"]


def retrieve_fit_contexts(fit_records, retriever):
    samples = []
    for i, row in enumerate(fit_records, start=1):
        results = retriever.search(row["question"], top_k=RETRIEVAL_TOP_K)
        best = results[0] if results else {}
        samples.append({
            "index": int(row["index"]),
            "question": row["question"],
            "reference": row.get("answer") or "",
            "context": best.get("context", ""),
            "retrieved_context_id": best.get("context_id"),
            "retrieval_score": best.get("score"),
            "has_answer": bool(row.get("has_answer")),
        })
        if i % 25 == 0 or i == len(fit_records):
            print(f"Contextos recuperados: {i}/{len(fit_records)}", flush=True)
    return samples


def truncate_prompt(prompt, tokenizer):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) > MAX_PROMPT_LENGTH:
        # Conserva la parte final, donde está la pregunta y el marcador de respuesta.
        token_ids = token_ids[-MAX_PROMPT_LENGTH:]
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def make_dataset(samples, tokenizer, system_prompt):
    rows = []
    for sample in samples:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{sample['context'] or 'No context is available.'}"
                    f"\n\nQuestion: {sample['question']}"
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        rows.append({
            "prompt": truncate_prompt(prompt, tokenizer),
            "reference": sample["reference"],
            "question_index": sample["index"],
        })
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description="GRPO con RAG y particiones congeladas FIT/DEV/TEST."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "grpo_rag_qwen15",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"El directorio {args.output} ya contiene archivos. "
            "Usa --overwrite para permitir su reutilización."
        )
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No existe el índice RAG: {INDEX_PATH}")
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"No existe el prompt: {PROMPT_PATH}")

    set_seed()
    device = torch.device(args.device)
    print(f"Dispositivo: {device}")
    print(f"Modelo: {MODEL_ID}")

    
    fit_records, dev_records, test_records = load_frozen_splits()
    print(
        f"Particiones: FIT={len(fit_records)}, DEV={len(dev_records)}, "
        f"TEST={len(test_records)}"
    )
    print("Solo FIT se utilizará para entrenar.")

    # RAG: recuperar el contexto top-1 para cada pregunta de FIT.
    print("Inicializando ContextRetriever...")
    retriever = ContextRetriever(str(INDEX_PATH), device=str(device))
    train_samples = retrieve_fit_contexts(fit_records, retriever)
    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    train_dataset = make_dataset(train_samples, tokenizer, system_prompt)
    if len(train_dataset) < 4:
        raise ValueError("Se necesitan al menos 4 ejemplos para num_generations=4.")

    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "split_info.json", {
        "seed": SEED,
        "split_source": str(RANKINGS_PATH),
        "training_partition": "fit",
        "train_size": len(fit_records),
        "dev_size": len(dev_records),
        "test_size": len(test_records),
        "dev_used_for_training": False,
        "test_used_for_training": False,
        "fit_indices": [int(x["index"]) for x in fit_records],
        "dev_indices": [int(x["index"]) for x in dev_records],
        "test_indices": [int(x["index"]) for x in test_records],
    })
    save_json(args.output / "train_retrieval.json", train_samples)
    (args.output / "training_prompt.txt").write_text(
        system_prompt, encoding="utf-8"
    )

    lengths = [
        len(tokenizer.encode(x, add_special_tokens=False))
        for x in train_dataset["prompt"]
    ]
    print(
        f"Prompts: máximo {max(lengths)} tokens "
        f"(límite {MAX_PROMPT_LENGTH})."
    )

    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    # TRL 1.17.0 no admite max_prompt_length en GRPOConfig.
    training_args = GRPOConfig(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        report_to="none",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        remove_unused_columns=False,
        seed=SEED,
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_answer_f1],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nIniciando entrenamiento GRPO...")
    started = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - started

    adapter_dir = args.output / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    save_json(args.output / "experiment_config.json", {
        "model_id": MODEL_ID,
        "seed": SEED,
        "trl_version": "1.17.0",
        "split_source": str(RANKINGS_PATH),
        "training_partition": "fit",
        "train_size": len(fit_records),
        "dev_size": len(dev_records),
        "test_size": len(test_records),
        "dev_used_for_training": False,
        "test_used_for_training": False,
        "retriever": "ContextRetriever",
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "retrieval_index": str(INDEX_PATH),
        "prompt": str(PROMPT_PATH),
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "reward": "src.evaluate.f1_score",
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_generations": 4,
        "training_seconds": training_seconds,
    })
    print(f"Adaptador guardado en: {adapter_dir}")
    print(f"Tiempo de entrenamiento: {training_seconds:.2f} s")


if __name__ == "__main__":
    main()
