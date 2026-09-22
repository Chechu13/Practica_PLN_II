import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from models.encoder_decoder import EncoderDecoderTransformer
from src.evaluate import evaluate_predictions
from src.preprocess.preprocess import CustomTokenizer, prepare_dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_source(item):
    return f"{item['context']} pregunta: {item['question']}"


def train_model(model, train_data, tokenizer, device, epochs=30):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    eos_id = tokenizer.word2idx["<EOS>"]

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for item in train_data:
            source_ids = tokenizer.encode(make_source(item))
            target_ids = tokenizer.encode(item["answer"])
            source = torch.tensor([source_ids], device=device)
            target = torch.tensor([target_ids], device=device)
            target_input = torch.cat(
                [torch.tensor([[eos_id]], device=device), target[:, :-1]], dim=1
            )

            optimizer.zero_grad()
            predictions = model(source, target_input)
            loss = criterion(
                predictions.reshape(-1, tokenizer.vocab_size), target.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            mean_loss = total_loss / len(train_data)
            print(f"Epoca {epoch + 1}/{epochs} | Perdida media: {mean_loss:.4f}")


def generate_answer(model, item, tokenizer, device, max_tokens=32):
    model.eval()
    source = torch.tensor([tokenizer.encode(make_source(item))], device=device)
    eos_id = tokenizer.word2idx["<EOS>"]
    generated = [eos_id]

    with torch.no_grad():
        for _ in range(max_tokens):
            target_input = torch.tensor([generated], device=device)
            predictions = model(source, target_input)
            next_token = predictions[0, -1].argmax().item()
            generated.append(next_token)
            if next_token == eos_id:
                break

    return tokenizer.decode(generated[1:])


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

    dataset = prepare_dataset("data/questions_train.json", "data/contexts.json")
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Datos de entrenamiento: {len(train_data)} | Datos de test: {len(test_data)}")

    tokenizer = CustomTokenizer()
    train_texts = [make_source(item) for item in train_data]
    train_texts += [item["answer"] for item in train_data]
    tokenizer.build_vocab(train_texts)
    print(f"Tamano del vocabulario: {tokenizer.vocab_size} tokens")

    model = EncoderDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_src_len=512,
        max_tgt_len=64,
    ).to(device)
    print("\nIniciando entrenamiento del encoder-decoder (30 epocas)...")
    train_model(model, train_data, tokenizer, device)

    print("\nEvaluando en el conjunto de TEST...")
    results = []
    for item in test_data:
        results.append(
            {
                "question": item["question"],
                "prediction": generate_answer(model, item, tokenizer, device),
                "reference": item["answer"],
                "has_answer": item["has_answer"],
            }
        )

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    predictions_path = os.path.join(
        results_dir, "encoder_decoder_predictions.json"
    )
    metrics_path = os.path.join(results_dir, "encoder_decoder_metrics.json")
    metrics = metrics_by_answer_status(results)

    with open(predictions_path, "w", encoding="utf-8") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)

    print(f"Predicciones guardadas en: {predictions_path}")
    print(f"Metricas guardadas en: {metrics_path}")
    print("\nResultados del encoder-decoder:")
    for name, values in metrics.items():
        print(
            f"{name}: EM={values['exact_match']}% | "
            f"F1={values['f1_score']}% | "
            f"n={values['total_evaluated']}"
        )


if __name__ == "__main__":
    main()
