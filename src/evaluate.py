import re
from collections import Counter

def normalize_answer(s):
    """
    Normaliza el texto para una comparación justa:
    - Pasa a minúsculas
    - Elimina artículos y puntuación básica
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(['.', ',', '!', '?', ':', ';', "'", '"']))
    return ' '.join(s.split())

def exact_match_score(prediction, reference):
    """Calcula si la predicción coincide exactamente con la referencia."""
    return int(normalize_answer(prediction) == normalize_answer(reference))

def f1_score(prediction, reference):
    """Calcula el F1-score token a token (muy útil para respuestas cortas)."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return int(pred_tokens == ref_tokens)
        
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_predictions(results):
    """
    Evalúa una lista de diccionarios con 'prediction' y 'reference'.
    Devuelve las métricas medias y globales.
    """
    total = len(results)
    if total == 0:
        return {
            "exact_match": 0.0,
            "f1_score": 0.0,
            "total_evaluated": 0
        }

    em_total = 0
    f1_total = 0.0

    for item in results:
        pred = item.get('prediction', '')
        ref = item.get('reference', '')
        
        em_total += exact_match_score(pred, ref)
        f1_total += f1_score(pred, ref)

    metrics = {
        "exact_match": round(em_total / total * 100, 2), # En porcentaje
        "f1_score": round(f1_total / total * 100, 2),    # En porcentaje
        "total_evaluated": total
    }
    return metrics