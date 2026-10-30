"""Script de entrenamiento con GRPO (Group Relative Policy Optimization).

Sigue el enfoque reproducible del proyecto, cargando el modelo base,
el adaptador o configuración de prompts, y optimizando mediante recompensas.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocess.preprocess import prepare_dataset

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # O tu modelo base/adaptador
MAX_PROMPT_LENGTH = 384
MAX_COMPLETION_LENGTH = 64


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# FUNCIONES DE RECOMPENSA (Reward Functions)
# ============================================================
# GRPO evalúa múltiples generaciones (completions) por prompt.
# Aquí defines qué hace que una respuesta sea mejor que otra.

def reward_exact_match(completions, references, **kwargs):
    """Recompensa basada en coincidencia exacta con la referencia."""
    rewards = []
    for completion, reference in zip(completions, references):
        score = 1.0 if completion.strip() == reference.strip() else 0.0
        rewards.append(score)
    return rewards


def reward_format_check(completions, **kwargs):
    """Recompensa si la respuesta sigue un formato estructurado (ej. no está vacía)."""
    rewards = []
    for completion in completions:
        score = 0.5 if len(completion.strip()) > 0 else 0.0
        rewards.append(score)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento GRPO para Qwen")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "grpo_adapter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed()
    device = torch.device(args.device)
    print(f"🚀 Dispositivo de entrenamiento: {device}")

    # 1. Cargar datos usando tu pipeline existente
    dataset_raw = prepare_dataset(
        ROOT / "data" / "questions_train.json",
        ROOT / "data" / "contexts.json",
    )
    
    # Convertir a formato compatible para GRPO (requiere 'prompt' y 'reference')
    formatted_data = []
    for item in dataset_raw[:100]:  # Subconjunto de prueba para el script
        context = item.get("context") or "No context is available."
        prompt = f"Context:\n{context}\n\nQuestion: {item['question']}\nAnswer:"
        formatted_data.append({
            "prompt": prompt,
            "reference": item["answer"] or ""
        })

    train_dataset = Dataset.from_list(formatted_data)

    # 2. Tokenizador y Modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )

    # 3. Configuración de GRPO (GRPOConfig)
    training_args = GRPOConfig(
        output_dir=str(args.output),
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,          # Número de respuestas generadas por prompt para calcular ventaja
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        logging_steps=1,
        save_steps=10,
        report_to="none",           # Cambiar a "wandb" si usas seguimiento
        bf16=True if device.type == "cuda" else False,
    )

    # Configuración opcional de LoRA para eficiencia (como usas en tus otros entrenamientos)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # 4. Inicializar el GRPOTrainer de TRL
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_exact_match, reward_format_check],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    print("\n🏁 Iniciando entrenamiento GRPO...")
    trainer.train()

    # 5. Guardar artefactos
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"✅ Modelo entrenado con GRPO guardado en: {args.output}")


if __name__ == "__main__":
    main()