"""QLoRA training for Qwen3.5-9B with the controlled prompt experiment.

This is intentionally independent from the existing LoRA experiments. It
requires CUDA and bitsandbytes; it never falls back to unquantized LoRA.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.preprocess import prepare_dataset
from src.retrieval.context_retriever import ContextRetriever


MODEL_ID = "Qwen/Qwen3.5-9B"
SEED = 42
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 96
EPOCHS = 1
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0

INDEX_PATH = PROJECT_ROOT / "models" / "context_embeddings.pt"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_DIR = PROJECT_ROOT / "models" / "prompt_training_controlled_qwen35_9b"
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_training_controlled_qwen35_9b"

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

# These are the token-mixer projections present in the Qwen3.5 text model.
# The two gate projections are deliberately excluded: they control decay and
# mixing rather than projecting the token representation itself.
LORA_MODULE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_prompt(prompt_name: str) -> str:
    path = PROMPTS_DIR / PROMPT_FILES[prompt_name]
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra la plantilla del prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(prompt_name: str, examples: list[dict]) -> str:
    template = load_prompt(prompt_name)
    if prompt_name == "zero":
        return template
    if prompt_name == "one":
        if not examples:
            raise ValueError("One-shot necesita al menos un ejemplo.")
        example = examples[0]
        return template.format(
            example_context=example["context"],
            example_question=example["question"],
            example_answer=example["answer"],
        )
    if prompt_name == "few":
        if not examples:
            raise ValueError("Few-shot necesita al menos un ejemplo.")
        values = {}
        for i, example in enumerate(examples[:4], start=1):
            values[f"example_{i}_context"] = example["context"]
            values[f"example_{i}_question"] = example["question"]
            values[f"example_{i}_answer"] = example["answer"]
        try:
            return template.format(**values)
        except KeyError as exc:
            raise ValueError(
                f"Falta el marcador {exc} en {PROMPT_FILES['few']}."
            ) from exc
    raise ValueError(f"Estrategia desconocida: {prompt_name}")


def retrieve_samples(
    samples: list[dict], retriever: ContextRetriever
) -> list[dict]:
    retrieved_samples = []
    for index, sample in enumerate(samples, start=1):
        result = retriever.retrieve(sample["question"])
        retrieved_samples.append(
            {
                **sample,
                "context": result["context"],
                "context_id": result["context_id"],
                "retrieval_score": float(result["score"]),
            }
        )
        if index % 25 == 0 or index == len(samples):
            print(f"Contextos recuperados: {index}/{len(samples)}", flush=True)
    return retrieved_samples


def select_examples(train_samples: list[dict]) -> list[dict]:
    """Select up to four demonstrations from retrieved train samples only."""
    answerable = [
        sample
        for sample in train_samples
        if sample.get("has_answer")
        and sample.get("answer")
        and sample.get("context")
    ]
    unanswerable = [sample for sample in train_samples if not sample.get("has_answer")]
    if len(answerable) < 3:
        raise ValueError("No hay suficientes ejemplos respondibles en train.")

    short = min(answerable, key=lambda item: len(item["answer"].split()))
    medium_candidates = [
        item
        for item in answerable
        if item is not short and len(item["answer"].split()) <= 5
    ]
    medium = (
        medium_candidates[0]
        if medium_candidates
        else next(item for item in answerable if item is not short)
    )
    remaining = [
        item for item in answerable if item is not short and item is not medium
    ]
    long_answer = max(remaining, key=lambda item: len(item["answer"].split()))
    selected = [short, medium, long_answer]
    if unanswerable:
        selected.append(unanswerable[0])

    for item in answerable:
        if len(selected) >= 4:
            break
        if item not in selected:
            selected.append(item)

    return [
        {
            "question": item["question"],
            "context": item["context"],
            "answer": item.get("answer") or "",
            "context_id": item.get("context_id"),
            "retrieval_score": item.get("retrieval_score"),
        }
        for item in selected[:4]
    ]


class QuestionAnsweringDataset(Dataset):
    def __init__(
        self,
        samples: list[dict],
        tokenizer: Any,
        system_prompt: str,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        sample = self.samples[index]
        context = sample.get("context") or "No context is available."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nQuestion: {sample['question']}"
                ),
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt_text,
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


def collate_batch(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_masks = [], [], []
    for item in batch:
        padding = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention_masks.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def load_tokenizer() -> Any:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("El tokenizer no tiene pad_token_id.")
    tokenizer.truncation_side = "left"
    return tokenizer


def check_bitsandbytes() -> str:
    try:
        version = importlib.metadata.version("bitsandbytes")
        import bitsandbytes as bnb
        if not getattr(bnb, "__version__", version):
            raise ImportError("bitsandbytes no expone una versión válida")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError(
            "QLoRA requiere bitsandbytes, pero no está instalado. "
            "Ejecuta `uv sync` tras actualizar pyproject.toml. "
            "No se aplica fallback a LoRA sin cuantización."
        ) from exc
    return version


def choose_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Este experimento QLoRA requiere CUDA; no se ejecuta en CPU."
        )
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def discover_target_modules(model: nn.Module) -> list[str]:
    discovered = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in LORA_MODULE_SUFFIXES
    }
    missing = set(LORA_MODULE_SUFFIXES) - discovered
    if missing:
        raise RuntimeError(
            "La implementación cargada de Qwen3.5 no expone todos los "
            f"módulos esperados. Faltan: {sorted(missing)}. "
            "No se omiten silenciosamente capas de la arquitectura."
        )
    return [name for name in LORA_MODULE_SUFFIXES if name in discovered]


def set_cache_disabled(model: nn.Module) -> None:
    model.config.use_cache = False
    language_model = getattr(model, "language_model", None)
    if language_model is not None and hasattr(language_model, "config"):
        language_model.config.use_cache = False


def load_finetunable_model() -> tuple[nn.Module, dict[str, Any]]:
    bnb_version = check_bitsandbytes()
    compute_dtype = choose_compute_dtype()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map={"": 0},
    )
    target_modules = discover_target_modules(model)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    set_cache_disabled(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return model, {
        "bitsandbytes_version": bnb_version,
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "target_modules": target_modules,
        "trainable_parameters": trainable_parameters,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "use_double_quant": True,
        },
    }


def train_model(
    model: nn.Module,
    samples: list[dict],
    tokenizer: Any,
    system_prompt: str,
) -> list[dict[str, float]]:
    dataset = QuestionAnsweringDataset(samples, tokenizer, system_prompt)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    if not len(loader):
        raise ValueError("El conjunto de entrenamiento está vacío.")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE)
    updates_per_epoch = math.ceil(len(loader) / GRADIENT_ACCUMULATION_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * updates_per_epoch
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch_metrics = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            if step % 25 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | Paso {step}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )
        mean_loss = total_loss / len(loader)
        epoch_metrics.append({"epoch": epoch + 1, "mean_loss": mean_loss})
        print(f"Epoch {epoch + 1} | Loss media: {mean_loss:.4f}", flush=True)
    return epoch_metrics


def dependency_versions() -> dict[str, str]:
    names = ("torch", "transformers", "peft", "accelerate", "bitsandbytes")
    return {
        name: importlib.metadata.version(name)
        for name in names
        if importlib.util.find_spec(name) is not None
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA de Qwen3.5-9B con prompts controlados."
    )
    parser.add_argument(
        "--prompt",
        choices=("zero", "one", "few"),
        required=True,
        help="Estrategia de prompt que se va a entrenar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite sobrescribir los artefactos de esta estrategia.",
    )
    args = parser.parse_args()
    prompt_name = args.prompt
    adapter_dir = MODELS_DIR / ADAPTER_NAMES[prompt_name]
    result_dir = RESULTS_DIR / prompt_name
    if (adapter_dir.exists() or result_dir.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Ya existen artefactos para {prompt_name}. "
            "Usa --overwrite explícitamente para sobrescribirlos."
        )

    set_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requiere una GPU CUDA disponible.")
    print(f"Dispositivo: {torch.cuda.get_device_name(0)}")
    print(f"Estrategia: {prompt_name}")
    print(f"Modelo base: {MODEL_ID}")

    dataset = prepare_dataset(
        str(PROJECT_ROOT / "data" / "questions_train.json"),
        str(PROJECT_ROOT / "data" / "contexts.json"),
    )
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=SEED
    )
    retriever = ContextRetriever(str(INDEX_PATH), device="cuda")
    train_samples = retrieve_samples(train_data, retriever)
    examples = [] if prompt_name == "zero" else select_examples(train_samples)
    system_prompt = build_system_prompt(prompt_name, examples)

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "training_prompt.txt").write_text(
        system_prompt, encoding="utf-8"
    )
    save_json(result_dir / "examples.json", examples)
    save_json(
        result_dir / "split_info.json",
        {
            "seed": SEED,
            "train_size": len(train_data),
            "test_size": len(test_data),
            "test_used_for_training": False,
            "prompt_strategy": prompt_name,
            "model_id": MODEL_ID,
            "train_questions": [item["question"] for item in train_data],
            "test_questions": [item["question"] for item in test_data],
        },
    )
    save_json(result_dir / "train_retrieval.json", train_samples)

    tokenizer = load_tokenizer()
    model = None
    started = time.perf_counter()
    try:
        model, qlora_info = load_finetunable_model()
        losses = train_model(model, train_samples, tokenizer, system_prompt)
        training_seconds = time.perf_counter() - started
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        save_json(
            adapter_dir / "metadata.json",
            {
                "model_id": MODEL_ID,
                "prompt_strategy": prompt_name,
                "prompt_file": PROMPT_FILES[prompt_name],
                "adapter": ADAPTER_NAMES[prompt_name],
                "seed": SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "max_length": MAX_LENGTH,
                "max_target_length": MAX_TARGET_LENGTH,
                "max_grad_norm": MAX_GRAD_NORM,
                "gradient_checkpointing": True,
                "optimizer": "AdamW",
                "scheduler": "CosineAnnealingLR",
                "train_size": len(train_data),
                "test_size": len(test_data),
                "test_used_for_training": False,
                "training_losses": losses,
                "training_seconds": training_seconds,
                "peak_vram_gib": peak_vram,
                "dependency_versions": dependency_versions(),
                "cuda": {
                    "torch_cuda_version": torch.version.cuda,
                    "device_name": torch.cuda.get_device_name(0),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                },
                **qlora_info,
            },
        )
        print(f"Adaptador guardado en: {adapter_dir}")
        print(f"Pico de VRAM: {peak_vram:.2f} GiB")
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "CUDA OOM durante QLoRA. No se aplicó ningún fallback. "
            "Modifica explícitamente MAX_LENGTH, "
            "GRADIENT_ACCUMULATION_STEPS o el conjunto de módulos LoRA "
            "antes de repetir el experimento."
        ) from exc
    finally:
        if model is not None:
            del model
        del tokenizer
        del retriever
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
