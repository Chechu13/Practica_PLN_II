import json
import math
import os
import random
import sys
from functools import lru_cache

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.evaluate import evaluate_predictions
from src.preprocess.preprocess import prepare_dataset
from src.retrieval.fallback_retriever import FallbackRetriever


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_LENGTH = 384
EPOCHS = 1
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4
CONTEXT_INDEX_PATH = os.path.join(PROJECT_ROOT, "models", "context_embeddings.pt")
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
PROMPT_NAME = "primer.md"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=None)
def load_prompt(prompt_name):
    if not prompt_name.endswith(".md"):
        prompt_name = f"{prompt_name}.md"
    prompt_path = os.path.join(PROMPTS_DIR, prompt_name)
    if not os.path.isfile(prompt_path):
        available_prompts = sorted(
            name for name in os.listdir(PROMPTS_DIR) if name.endswith(".md")
        )
        raise FileNotFoundError(
            f"No se encontro el prompt '{prompt_name}' en {PROMPTS_DIR}. "
            f"Disponibles: {', '.join(available_prompts) or 'ninguno'}"
        )
    with open(prompt_path, encoding="utf-8") as prompt_file:
        return prompt_file.read().strip()


def make_prompt(item, prompt_name=PROMPT_NAME):
    context = item["context"] or "No context is available."
    messages = [
        {
            "role": "system",
            "content": load_prompt(prompt_name),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {item['question']}",
        },
    ]
    return messages


class QuestionAnsweringDataset(Dataset):
    def __init__(self, samples, tokenizer, retriever=None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.retrieved_samples = {}

    def _resolve_context(self, index):
        sample = self.samples[index]
        if sample["context"] or self.retriever is None:
            return sample
        if index not in self.retrieved_samples:
            retrieved = self.retriever.retrieve(sample["question"])
            self.retrieved_samples[index] = {
                **sample,
                "context": retrieved["context"],
            }
        return self.retrieved_samples[index]

    def _encode(self, sample):
        prompt = self.tokenizer.apply_chat_template(
            make_prompt(sample), tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH
        )["input_ids"]
        answer = sample["answer"] or ""
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        available = MAX_LENGTH - len(prompt_ids) - 1
        answer_ids = answer_ids[: max(0, available)]
        input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + answer_ids + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self._encode(self._resolve_context(index))


def collate_batch(batch, pad_token_id):
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        padding = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        labels.append(item["labels"] + [-100] * padding)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * padding)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def train_model(model, train_data, tokenizer, device, retriever=None):
    dataset = QuestionAnsweringDataset(train_data, tokenizer, retriever=retriever)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
    )
    updates_per_epoch = math.ceil(len(loader) / GRADIENT_ACCUMULATION_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * updates_per_epoch
    )
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            if step % 10 == 0 or step == len(loader):
                print(
                    f"  Paso {step}/{len(loader)} | "
                    f"Perdida: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                    flush=True,
                )
        print(f"Epoca {epoch + 1}/{EPOCHS} | Perdida media: {total_loss / len(loader):.4f}")


def generate_answer(model, tokenizer, item, device):
    prompt = tokenizer.apply_chat_template(
        make_prompt(item), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def metrics_by_answer_status(results):
    answerable = [item for item in results if item["has_answer"]]
    unanswerable = [item for item in results if not item["has_answer"]]
    return {
        "overall": evaluate_predictions(results),
        "answerable": evaluate_predictions(answerable),
        "unanswerable": evaluate_predictions(unanswerable),
    }


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    dataset = prepare_dataset(
        os.path.join(PROJECT_ROOT, "data", "questions_train.json"),
        os.path.join(PROJECT_ROOT, "data", "contexts.json"),
    )
    retriever = FallbackRetriever(CONTEXT_INDEX_PATH, device=device, threshold=0.60)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(
        f"Datos de entrenamiento: {len(train_data)} | Datos de test: {len(test_data)} | "
        f"Contextos recuperados: {len(retriever.context_ids)}"
    )

    print(f"Cargando {MODEL_ID} y preparando LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    print("\nIniciando fine-tuning LoRA...")
    train_model(model, train_data, tokenizer, device, retriever=retriever)

    print("\nEvaluando en el conjunto de TEST...")
    model.eval()
    results = []
    for item in test_data:
        retrieved = retriever.retrieve(item["question"])
        inference_item = {
            **item,
            "context": retrieved["context"],
        }
        results.append(
            {
                "question": item["question"],
                "prediction": generate_answer(
                    model, tokenizer, inference_item, device
                ),
                "reference": item["answer"],
                "has_answer": item["has_answer"],
                "retrieved_context_id": retrieved["context_id"],
                "retrieval_score": retrieved["score"],
            }
        )

    results_dir = os.path.join(PROJECT_ROOT, "results")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    adapter_dir = os.path.join(models_dir, "qwen_lora_adapter")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    predictions_path = os.path.join(results_dir, "huggingface_predictions.json")
    metrics_path = os.path.join(results_dir, "huggingface_metrics.json")
    metrics = metrics_by_answer_status(results)
    with open(predictions_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)

    print(f"Adaptador guardado en: {adapter_dir}")
    print(f"Predicciones guardadas en: {predictions_path}")
    print(f"Metricas guardadas en: {metrics_path}")
    for name, values in metrics.items():
        print(
            f"{name}: EM={values['exact_match']}% | "
            f"F1={values['f1_score']}% | n={values['total_evaluated']}"
        )

if __name__ == "__main__":
    main()