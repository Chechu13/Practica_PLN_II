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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.preprocess import prepare_dataset
from src.retrieval.context_retriever import ContextRetriever
from src.retrieval.fallback_retriever import FallbackRetriever


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 384
EPOCHS = 1
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4
BATCH_SIZE = 1
SEED = 42

INDEX_PATH = PROJECT_ROOT / "models" / "context_embeddings.pt"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparison_2x2"
MODELS_DIR = PROJECT_ROOT / "models" / "comparison_2x2"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt():
    prompt_path = PROMPTS_DIR / "primer.md"
    with open(prompt_path, encoding="utf-8") as f:
        return f.read().strip()


SYSTEM_PROMPT = load_prompt()


def make_messages(item):
    context = item["context"] or "No context is available."

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {item['question']}"
            ),
        },
    ]


class QuestionAnsweringDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def _encode(self, sample):
        prompt = self.tokenizer.apply_chat_template(
            make_messages(sample),
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        answer = sample["answer"] or ""
        answer_ids = self.tokenizer(
            answer,
            add_special_tokens=False,
        )["input_ids"]

        available = MAX_LENGTH - len(prompt_ids) - 1
        answer_ids = answer_ids[:max(0, available)]

        input_ids = prompt_ids + answer_ids + [
            self.tokenizer.eos_token_id
        ]
        labels = [-100] * len(prompt_ids) + answer_ids + [
            self.tokenizer.eos_token_id
        ]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self._encode(self.samples[index])


def collate_batch(batch, pad_token_id):
    max_length = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for item in batch:
        padding = max_length - len(item["input_ids"])

        input_ids.append(
            item["input_ids"] + [pad_token_id] * padding
        )
        labels.append(
            item["labels"] + [-100] * padding
        )
        attention_mask.append(
            [1] * len(item["input_ids"]) + [0] * padding
        )

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def retrieve_samples(samples, retriever):
    retrieved_samples = []

    for i, sample in enumerate(samples, start=1):
        retrieved = retriever.retrieve(sample["question"])

        retrieved_samples.append({
            "question": sample["question"],
            "answer": sample["answer"],
            "has_answer": sample["has_answer"],
            "context": retrieved.get("context", ""),
            "context_id": retrieved.get("context_id", "none"),
            "score": float(retrieved.get("score", 0.0)),
            "source": retrieved.get("source", "none"),
            "retrieval": retrieved.get("retrieval", "none"),
        })

        if i % 25 == 0 or i == len(samples):
            print(f"Retrieval: {i}/{len(samples)}", flush=True)

    return retrieved_samples


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_finetunable_model():
    dtype = (
        torch.bfloat16
        if DEVICE.type == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
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


def train_model(model, train_samples, tokenizer):
    dataset = QuestionAnsweringDataset(
        train_samples,
        tokenizer,
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

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
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

    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()

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
                    model.parameters(),
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


def main():
    set_seed()

    print(f"Dispositivo: {DEVICE}")

    dataset = prepare_dataset(
        str(PROJECT_ROOT / "data" / "questions_train.json"),
        str(PROJECT_ROOT / "data" / "contexts.json"),
    )

    train_data, test_data = train_test_split(
        dataset,
        test_size=0.2,
        random_state=SEED,
    )

    def build_retriever():
        return FallbackRetriever(
        str(INDEX_PATH),
        device=DEVICE,
        threshold=0.60,
        margin=0.03,
    )


    CACHE_NAME = "fallback_retrieval_cache.json"
    ADAPTER_NAME = "qwen_lora_fallback"


    RETRIEVER = build_retriever()
    train_retrieved = retrieve_samples(train_data, RETRIEVER)
    test_retrieved = retrieve_samples(test_data, RETRIEVER)

    cache_name = CACHE_NAME
    cache_path = RESULTS_DIR / cache_name

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_retrieved,
                "test": test_retrieved,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    tokenizer = load_tokenizer()
    model = load_finetunable_model()

    print("\nIniciando fine-tuning...")
    train_model(
        model,
        train_retrieved,
        tokenizer,
    )

    output_dir = MODELS_DIR / ADAPTER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nAdapter guardado en: {output_dir}")
    print(f"Retrieval cache guardada en: {cache_path}")


if __name__ == "__main__":
    main()
