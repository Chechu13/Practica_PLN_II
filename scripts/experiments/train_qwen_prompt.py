
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

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

INDEX_PATH = PROJECT_ROOT / "models" / "context_embeddings.pt"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_DIR = PROJECT_ROOT / "models" / "prompt_training"
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_training"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PROMPTS
# ============================================================

PROMPT_FILES = {
    "zero": "primer.md",
    "one": "one_shot.md",
    "few": "few_shot.md",
}

ADAPTER_NAMES = {
    "zero": "qwen_lora_zero",
    "one": "qwen_lora_one",
    "few": "qwen_lora_few",
}


def load_prompt(prompt_name):
    path = PROMPTS_DIR / PROMPT_FILES[prompt_name]

    if not path.exists():
        raise FileNotFoundError(
            f"No se encuentra el prompt: {path}"
        )

    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def build_system_prompt(prompt_name, examples):
    template = load_prompt(prompt_name)

    if prompt_name == "zero":
        return template

    if prompt_name == "one":
        example = examples[0]

        return template.format(
            example_context=example["context"],
            example_question=example["question"],
            example_answer=example["answer"],
        )

    if prompt_name == "few":
        selected = examples[:4]
        values = {}

        for i, example in enumerate(selected, start=1):
            values[f"example_{i}_context"] = example["context"]
            values[f"example_{i}_question"] = example["question"]
            values[f"example_{i}_answer"] = example["answer"]

        return template.format(**values)

    raise ValueError(f"Prompt desconocido: {prompt_name}")


# ============================================================
# LOCAL RETRIEVAL
# ============================================================

def retrieve_samples(samples, retriever):
    """
    Recupera un contexto del índice local para cada pregunta.
    No utiliza FallbackRetriever ni búsqueda web.
    """

    retrieved_samples = []

    for i, sample in enumerate(samples, start=1):
        result = retriever.retrieve(sample["question"])

        retrieved_samples.append({
            **sample,
            "context": result["context"],
            "context_id": result["context_id"],
            "retrieval_score": float(result["score"]),
        })

        if i % 25 == 0 or i == len(samples):
            print(
                f"Contextos locales: {i}/{len(samples)}",
                flush=True,
            )

    return retrieved_samples


# ============================================================
# EXAMPLE SELECTION
# ============================================================

def select_examples(train_data):
    """
    Selecciona ejemplos únicamente del train split.
    Se reutilizan los mismos ejemplos durante todo el
    entrenamiento de cada estrategia.
    """

    answerable = [
        x for x in train_data
        if x.get("has_answer")
        and x.get("answer")
        and x.get("context")
    ]

    unanswerable = [
        x for x in train_data
        if not x.get("has_answer")
    ]

    if len(answerable) < 3:
        raise ValueError(
            "No hay suficientes ejemplos respondibles "
            "para construir los prompts."
        )

    short = min(
        answerable,
        key=lambda x: len(x["answer"].split()),
    )

    medium_candidates = [
        x for x in answerable
        if x is not short
        and len(x["answer"].split()) <= 5
    ]

    medium = (
        medium_candidates[0]
        if medium_candidates
        else next(x for x in answerable if x is not short)
    )

    remaining = [
        x for x in answerable
        if x is not short and x is not medium
    ]

    long_answer = max(
        remaining,
        key=lambda x: len(x["answer"].split()),
    )

    examples = [short, medium, long_answer]

    if unanswerable:
        examples.append(unanswerable[0])

    if len(examples) < 4:
        for item in answerable:
            if item not in examples:
                examples.append(item)
            if len(examples) == 4:
                break

    return [
        {
            "question": x["question"],
            "context": x["context"],
            "answer": x.get("answer") or "",
        }
        for x in examples
    ]


# ============================================================
# DATASET
# ============================================================

class QuestionAnsweringDataset(Dataset):

    def __init__(self, samples, tokenizer, system_prompt):
        self.samples = samples
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        context = sample["context"] or "No context is available."

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {sample['question']}"
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Reserve tokens for the target answer.
        # Left truncation preserves the current question
        # at the end of the prompt if it is too long.
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH - MAX_TARGET_LENGTH - 1,
        )["input_ids"]

        answer = sample.get("answer") or ""

        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_TARGET_LENGTH,
        )["input_ids"]

        input_ids = (
            prompt_ids
            + answer_ids
            + [self.tokenizer.eos_token_id]
        )

        labels = (
            [-100] * len(prompt_ids)
            + answer_ids
            + [self.tokenizer.eos_token_id]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


def collate_batch(batch, pad_token_id):
    max_length = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        padding = max_length - len(item["input_ids"])

        input_ids.append(
            item["input_ids"] + [pad_token_id] * padding
        )

        labels.append(
            item["labels"] + [-100] * padding
        )

        attention_masks.append(
            [1] * len(item["input_ids"]) + [0] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_masks),
    }


# ============================================================
# MODEL AND LORA
# ============================================================

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep the end of long prompts, where the current
    # question and its context appear.
    tokenizer.truncation_side = "left"

    return tokenizer


def load_finetunable_model():
    dtype = (
        torch.bfloat16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(DEVICE)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(model, samples, tokenizer, system_prompt):
    dataset = QuestionAnsweringDataset(
        samples,
        tokenizer,
        system_prompt,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(
            batch,
            tokenizer.pad_token_id,
        ),
    )

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    updates_per_epoch = math.ceil(
        len(loader) / GRADIENT_ACCUMULATION_STEPS
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * updates_per_epoch,
    )

    model.train()
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, batch in enumerate(loader, start=1):
            batch = {
                key: value.to(DEVICE)
                for key, value in batch.items()
            }

            loss = model(**batch).loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            if (
                step % GRADIENT_ACCUMULATION_STEPS == 0
                or step == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=1.0,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            if step % 25 == 0 or step == len(loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Paso {step}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True,
                )

        print(
            f"Epoch {epoch + 1} | "
            f"Loss media: {total_loss / len(loader):.4f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        choices=["zero", "one", "few"],
        required=True,
        help="Estrategia de prompting que se va a entrenar.",
    )

    args = parser.parse_args()
    prompt_name = args.prompt

    set_seed()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {DEVICE}")
    print(f"Estrategia: {prompt_name}")
    print(f"Modelo base: {MODEL_ID}")

    # --------------------------------------------------------
    # DATA SPLIT
    # --------------------------------------------------------

    dataset = prepare_dataset(
        str(PROJECT_ROOT / "data" / "questions_train.json"),
        str(PROJECT_ROOT / "data" / "contexts.json"),
    )

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=SEED,
    )

    print(
        f"Train: {len(train_data)} | "
        f"Test reservado: {len(test_data)}"
    )

    # --------------------------------------------------------
    # LOCAL RETRIEVAL ONLY
    # --------------------------------------------------------

    retriever = ContextRetriever(
        str(INDEX_PATH),
        device=DEVICE,
    )

    print("\nRecuperando contextos locales del train...")

    train_samples = retrieve_samples(
        train_data,
        retriever,
    )

    # --------------------------------------------------------
    # EXAMPLES FROM TRAIN ONLY
    # --------------------------------------------------------

    if prompt_name == "zero":
        examples = []
        print("\nZero-shot: no se utilizan ejemplos en el prompt.")

    else:
        examples = select_examples(train_samples)

        print("\nEjemplos utilizados en el prompt:")

        for i, example in enumerate(examples, start=1):
            print(f"\nEjemplo {i}")
            print(f"Pregunta: {example['question']}")
            print(f"Respuesta: {example['answer']}")

    system_prompt = build_system_prompt(
        prompt_name,
        examples,
    )

    # Save the exact prompt and examples used.
    experiment_dir = RESULTS_DIR / prompt_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(
        experiment_dir / "training_prompt.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(system_prompt)

    with open(
        experiment_dir / "examples.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            examples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # LOAD AND TRAIN
    # --------------------------------------------------------

    tokenizer = load_tokenizer()
    model = load_finetunable_model()

    print("\nIniciando fine-tuning...")

    train_model(
        model,
        train_samples,
        tokenizer,
        system_prompt,
    )

    # --------------------------------------------------------
    # SAVE ADAPTER
    # --------------------------------------------------------

    adapter_name = ADAPTER_NAMES[prompt_name]
    output_dir = MODELS_DIR / adapter_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "model_id": MODEL_ID,
        "prompt_strategy": prompt_name,
        "prompt_file": PROMPT_FILES[prompt_name],
        "adapter": adapter_name,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "retriever": "ContextRetriever (local only)",
        "train_size": len(train_data),
        "test_size": len(test_data),
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nAdapter guardado en: {output_dir}")
    print(f"Metadatos guardados en: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
