import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocess.preprocess import prepare_dataset
from src.retrieval.context_retriever import ContextRetriever

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
MAX_LENGTH = 512
MAX_TARGET_LENGTH = 96
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

INDEX_PATH = PROJECT_ROOT / "models" / "context_embeddings.pt"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_DIR = PROJECT_ROOT / "models" / "prompt_training_controlled_qwen15"
RESULTS_DIR = PROJECT_ROOT / "results" / "prompt_training_controlled_qwen15"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_FILES = {
    "zero": "controlled_zero.md",
    "one": "controlled_one.md",
    "few": "controlled_few.md",
}
ADAPTER_NAMES = {
    "zero": "qwen15_lora_controlled_zero",
    "one": "qwen15_lora_controlled_one",
    "few": "qwen15_lora_controlled_few",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt(prompt_name):
    path = PROMPTS_DIR / PROMPT_FILES[prompt_name]
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra la plantilla del prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(prompt_name, examples):
    """Construye los prompts controlados usados en el experimento de 7B."""
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
                f"Falta el marcador {exc} en {PROMPT_FILES['few']}. "
                "Comprueba los marcadores de la plantilla."
            ) from exc
    raise ValueError(f"Estrategia desconocida: {prompt_name}")


def retrieve_samples(samples, retriever):
    """Recupera contextos del índice local, sin búsqueda web ni fallback."""
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
            print(f"Contextos recuperados: {i}/{len(samples)}", flush=True)
    return retrieved_samples


def select_examples(train_samples):
    """Selecciona hasta cuatro demostraciones exclusivamente de train."""
    answerable = [
        x for x in train_samples
        if x.get("has_answer") and x.get("answer") and x.get("context")
    ]
    unanswerable = [x for x in train_samples if not x.get("has_answer")]
    if len(answerable) < 3:
        raise ValueError("No hay suficientes ejemplos respondibles en train.")

    short = min(answerable, key=lambda x: len(x["answer"].split()))
    medium_candidates = [
        x for x in answerable
        if x is not short and len(x["answer"].split()) <= 5
    ]
    medium = (
        medium_candidates[0]
        if medium_candidates
        else next(x for x in answerable if x is not short)
    )
    remaining = [x for x in answerable if x is not short and x is not medium]
    long_answer = max(remaining, key=lambda x: len(x["answer"].split()))
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
            "question": x["question"],
            "context": x["context"],
            "answer": x.get("answer") or "",
        }
        for x in selected[:4]
    ]


class QuestionAnsweringDataset(Dataset):
    def __init__(self, samples, tokenizer, system_prompt):
        self.samples = samples
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        context = sample.get("context") or "No context is available."
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {sample['question']}",
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
        return {
            "input_ids": prompt_ids + answer_ids + [eos_id],
            "labels": [-100] * len(prompt_ids) + answer_ids + [eos_id],
        }


def collate_batch(batch, pad_token_id):
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


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    return tokenizer


def load_finetunable_model():
    if DEVICE.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model = model.to(DEVICE)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model


def train_model(model, samples, tokenizer, system_prompt):
    dataset = QuestionAnsweringDataset(samples, tokenizer, system_prompt)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=LEARNING_RATE)
    updates_per_epoch = math.ceil(len(loader) / GRADIENT_ACCUMULATION_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * updates_per_epoch
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(DEVICE) for key, value in batch.items()}
            loss = model(**batch).loss
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()
            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
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
        print(
            f"Epoch {epoch + 1} | Loss media: {total_loss / len(loader):.4f}",
            flush=True,
        )


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tuning controlado de Qwen2.5-1.5B-Instruct "
            "con prompts zero-shot, one-shot o few-shot."
        )
    )
    parser.add_argument(
        "--prompt", choices=["zero", "one", "few"], required=True,
        help="Estrategia de prompt que se va a entrenar.",
    )
    args = parser.parse_args()
    prompt_name = args.prompt
    set_seed()

    print(f"Dispositivo: {DEVICE}")
    print(f"Estrategia: {prompt_name}")
    print(f"Modelo base: {MODEL_ID}")
    print(f"Épocas: {EPOCHS}")

    dataset = prepare_dataset(
        str(PROJECT_ROOT / "data" / "questions_train.json"),
        str(PROJECT_ROOT / "data" / "contexts.json"),
    )
    train_data, test_data = train_test_split(
        dataset, test_size=0.2, random_state=SEED
    )
    print(f"Train: {len(train_data)} | Test reservado: {len(test_data)}")

    retriever = ContextRetriever(str(INDEX_PATH), device=DEVICE)
    print("\nRecuperando contextos locales de train...")
    train_samples = retrieve_samples(train_data, retriever)

    if prompt_name == "zero":
        examples = []
        print("\nZero-shot: no se incluyen ejemplos.")
    else:
        examples = select_examples(train_samples)
        print("\nEjemplos seleccionados:")
        for i, example in enumerate(examples, start=1):
            print(f"\nEjemplo {i}")
            print(f"Pregunta: {example['question']}")
            print(f"Respuesta: {example['answer']}")

    system_prompt = build_system_prompt(prompt_name, examples)
    experiment_dir = RESULTS_DIR / prompt_name
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
            "prompt_strategy": prompt_name,
            "model_id": MODEL_ID,
            "epochs": EPOCHS,
        },
    )

    tokenizer = load_tokenizer()
    model = None
    try:
        model = load_finetunable_model()
        print("\nIniciando fine-tuning...")
        train_model(model, train_samples, tokenizer, system_prompt)

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
        }
        save_json(output_dir / "metadata.json", metadata)
        print(f"\nAdaptador guardado en: {output_dir}")
        print(f"Metadatos guardados en: {output_dir / 'metadata.json'}")
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
