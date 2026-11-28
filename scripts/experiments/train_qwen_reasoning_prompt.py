"""Fine-tune Qwen2.5-7B LoRA adapters with structured reasoning prompts.

This experiment is intentionally isolated from existing controlled adapters.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocess.preprocess import prepare_dataset
from src.retrieval.context_retriever import ContextRetriever

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SEED = 42
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 96
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
PROMPT_FILES = {
    "zero_reasoning": "controlled_zero_reasoning.md",
    "one_reasoning": "controlled_one_reasoning.md",
    "few_reasoning": "controlled_few_reasoning.md",
}
ADAPTER_NAMES = {
    "zero_reasoning": "qwen25_7b_lora_zero_reasoning",
    "one_reasoning": "qwen25_7b_lora_one_reasoning",
    "few_reasoning": "qwen25_7b_lora_few_reasoning",
}
INDEX_PATH = ROOT / "models" / "context_embeddings.pt"
PROMPTS_DIR = ROOT / "prompts"
MODELS_DIR = ROOT / "models" / "prompt_training_reasoning"
RESULTS_DIR = ROOT / "results" / "prompt_training_reasoning"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def retrieve_samples(
    samples: list[dict], retriever: ContextRetriever
) -> list[dict]:
    retrieved = []
    for index, sample in enumerate(samples, start=1):
        result = retriever.retrieve(sample["question"])
        retrieved.append({
            **sample,
            "context": result["context"],
            "context_id": result["context_id"],
            "retrieval_score": float(result["score"]),
        })
        if index % 25 == 0 or index == len(samples):
            print(f"Contextos recuperados: {index}/{len(samples)}", flush=True)
    return retrieved


def select_examples(train_samples: list[dict]) -> list[dict]:
    """Select up to four demonstrations from the training partition only."""
    answerable = [
        item for item in train_samples
        if item.get("has_answer") and item.get("answer") and item.get("context")
    ]
    unanswerable = [
        item for item in train_samples if not item.get("has_answer")
    ]
    if len(answerable) < 3:
        raise ValueError("Se necesitan al menos tres ejemplos respondibles.")
    short = min(answerable, key=lambda item: len(item["answer"].split()))
    medium_candidates = [
        item for item in answerable
        if item is not short and len(item["answer"].split()) <= 5
    ]
    medium = medium_candidates[0] if medium_candidates else next(
        item for item in answerable if item is not short
    )
    remaining = [
        item for item in answerable
        if item is not short and item is not medium
    ]
    long_answer = max(remaining, key=lambda item: len(item["answer"].split()))
    selected = [short, medium, long_answer]
    if unanswerable:
        selected.append(unanswerable[0])
    for item in answerable:
        if len(selected) == 4:
            break
        if item not in selected:
            selected.append(item)
    return [
        {
            "question": item["question"],
            "context": item["context"],
            "answer": item.get("answer") or "",
        }
        for item in selected[:4]
    ]


def build_system_prompt(condition: str, examples: list[dict]) -> str:
    template_path = PROMPTS_DIR / PROMPT_FILES[condition]
    template = template_path.read_text(encoding="utf-8").strip()
    if condition == "zero_reasoning":
        return template
    if condition == "one_reasoning":
        if not examples:
            raise ValueError("One-shot necesita al menos un ejemplo de train.")
        example = examples[0]
        return template.format(
            example_context=example["context"],
            example_question=example["question"],
            example_answer=example["answer"],
        )
    if condition == "few_reasoning":
        if len(examples) < 4:
            raise ValueError("Few-shot necesita cuatro ejemplos de train.")
        values = {}
        for index, example in enumerate(examples[:4], start=1):
            values[f"example_{index}_context"] = example["context"]
            values[f"example_{index}_question"] = example["question"]
            values[f"example_{index}_answer"] = example["answer"]
        return template.format(**values)
    raise ValueError(f"Condición desconocida: {condition}")


class QuestionAnsweringDataset(Dataset):
    """The target is only the concise gold answer, never private analysis."""

    def __init__(
        self, samples: list[dict], tokenizer: Any, system_prompt: str
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self.samples[index]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{sample.get('context') or 'No context is available.'}"
                    f"\n\nQuestion: {sample['question']}"
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH - MAX_TARGET_LENGTH - 1,
        )["input_ids"]
        answer_ids = self.tokenizer(
            sample.get("answer") or "",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_TARGET_LENGTH,
        )["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("El tokenizer no tiene eos_token_id.")
        return {
            "input_ids": prompt_ids + answer_ids + [eos_id],
            "labels": [-100] * len(prompt_ids) + answer_ids + [eos_id],
        }


def collate_batch(
    batch: list[dict], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, masks = [], [], []
    for item in batch:
        padding = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        masks.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def load_model() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    dtype = (
        torch.bfloat16
        if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if DEVICE.type == "cuda" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype
    ).to(DEVICE)
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    return model, tokenizer


def train_model(
    model: Any,
    samples: list[dict],
    tokenizer: Any,
    system_prompt: str,
) -> list[float]:
    dataset = QuestionAnsweringDataset(samples, tokenizer, system_prompt)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(
            batch, tokenizer.pad_token_id
        ),
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    updates = math.ceil(len(loader) / GRADIENT_ACCUMULATION_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * updates
    )
    losses = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(EPOCHS):
        total = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.item())
            if step % 25 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Paso {step}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )
        mean_loss = total / max(1, len(loader))
        losses.append(mean_loss)
        print(f"Epoch {epoch + 1} | Loss media: {mean_loss:.4f}", flush=True)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena Qwen2.5-7B con prompts de razonamiento estructurado."
    )
    parser.add_argument(
        "--prompt",
        choices=tuple(PROMPT_FILES),
        required=True,
        help="Condición de prompting que se va a entrenar.",
    )
    args = parser.parse_args()
    condition = args.prompt

    # Each run replaces only the artifacts belonging to the selected condition.
    # Do this before loading the retriever or model to fail fast and avoid stale outputs.
    experiment_dir = RESULTS_DIR / condition
    adapter_dir = MODELS_DIR / ADAPTER_NAMES[condition]
    for output_dir in (experiment_dir, adapter_dir):
        if output_dir.exists():
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()

    set_seed()

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No existe el índice local: {INDEX_PATH}. "
            "Ejecuta scripts/create_context_embeddings.py."
        )
    dataset = prepare_dataset(
        str(ROOT / "data" / "questions_train.json"),
        str(ROOT / "data" / "contexts.json"),
    )
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=SEED
    )
    retriever = ContextRetriever(str(INDEX_PATH), device=DEVICE)
    train_samples = retrieve_samples(train_data, retriever)
    examples = (
        [] if condition == "zero_reasoning" else select_examples(train_samples)
    )
    system_prompt = build_system_prompt(condition, examples)

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "training_prompt.txt").write_text(
        system_prompt, encoding="utf-8"
    )
    save_json(experiment_dir / "examples.json", examples)
    save_json(
        experiment_dir / "split_info.json",
        {
            "seed": SEED,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "test_used_for_training": False,
            "model_id": MODEL_ID,
            "prompt": condition,
            "dataset_size": len(dataset),
            "epochs": EPOCHS,
            "train_questions": [item["question"] for item in train_data],
            "test_questions": [item["question"] for item in test_data],
        },
    )

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model()
        losses = train_model(model, train_samples, tokenizer, system_prompt)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        save_json(
            adapter_dir / "metadata.json",
            {
                "model_id": MODEL_ID,
                "prompt": condition,
                "prompt_file": PROMPT_FILES[condition],
                "seed": SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "max_length": MAX_LENGTH,
                "max_target_length": MAX_TARGET_LENGTH,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "retriever": "ContextRetriever (local only)",
                "train_size": len(train_data),
                "test_size": len(test_data),
                "test_used_for_training": False,
                "training_losses": losses,
            },
        )
        print(f"Adaptador guardado en: {adapter_dir}")
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
