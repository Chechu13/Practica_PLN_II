import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Configurar rutas para encontrar los módulos
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.preprocess.preprocess import prepare_dataset, CustomTokenizer
from src.evaluate import evaluate_predictions
from models.simple_decoder import SimpleDecoder

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")

    print("Cargando y preprocesando datos...")
    dataset = prepare_dataset('data/questions_train.json', 'data/contexts.json')
    
    # Dividir en Train (80%) y Test (20%) de verdad
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Datos de entrenamiento: {len(train_data)} | Datos de test: {len(test_data)}")

    # Construir vocabulario únicamente con el texto de entrenamiento
    train_texts = [f"pregunta: {item['question']} respuesta: {item['answer']}" for item in train_data]
    
    print("Construyendo vocabulario...")
    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(train_texts)
    print(f"Tamaño del vocabulario: {tokenizer.vocab_size} tokens")

    # Instanciar modelo
    model = SimpleDecoder(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("\nIniciando entrenamiento real (30 épocas)...")
    epochs = 30
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for text in train_texts:
            tokens = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
            if tokens.size(1) < 2:
                continue
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            optimizer.zero_grad()
            predictions = model(inputs)
            
            loss = criterion(predictions.view(-1, tokenizer.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Época {epoch+1}/{epochs} | Pérdida media: {total_loss/len(train_texts):.4f}")

    print("\nEvaluando en el conjunto de TEST (preguntas nunca vistas)...")
    model.eval()
    evaluation_results = []
    
    with torch.no_grad():
        for item in test_data:
            q_text = item['question']
            ref_answer = item['answer']
            
            prompt = f"pregunta: {q_text} respuesta:"
            input_ids = torch.tensor(tokenizer.encode(prompt)[:-1]).unsqueeze(0).to(device)
            
            # Generar tokens uno a uno (inferencia)
            for _ in range(5): # Límite de 5 tokens generados
                preds = model(input_ids)
                next_token = torch.argmax(preds[0, -1, :]).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
                if next_token == 2: # <EOS>
                    break
            
            # Decodificar y aislar la parte de la respuesta generada
            full_generated = tokenizer.decode(input_ids[0].cpu().tolist())
            
            # Intentar extraer lo que hay después de 'respuesta:'
            if "respuesta:" in full_generated:
                pred_answer = full_generated.split("respuesta:")[-1].strip()
            else:
                pred_answer = full_generated
                
            evaluation_results.append({
                "question": q_text,
                "prediction": pred_answer,
                "reference": ref_answer,
                "has_answer": item["has_answer"]
            })

    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "custom_decoder_predictions.json")
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(evaluation_results, results_file, ensure_ascii=False, indent=2)
    print(f"Resultados guardados en: {results_path}")

    # Calcular métricas usando el evaluador genérico
    metrics = evaluate_predictions(evaluation_results)
    
    print("\n==============================")
    print("📊 RESULTADOS DE LA EVALUACIÓN:")
    print(f"Exact Match (EM): {metrics['exact_match']}%")
    print(f"F1-Score: {metrics['f1_score']}%")
    print(f"Total evaluados: {metrics['total_evaluated']}")
    print("==============================")

if __name__ == "__main__":
    main()