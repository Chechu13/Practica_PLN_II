import json
import re

def load_json(filepath):
    """Carga un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def clean_text(text):
    """
    Limpieza básica de texto útil para cualquier modelo.
    - Elimina espacios múltiples.
    - Elimina saltos de línea innecesarios.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text) # Reemplaza múltiples espacios/saltos por un solo espacio
    return text.strip()

def prepare_dataset(questions_path, contexts_path=None):
    """
    Carga y limpia las preguntas. Si se proporciona el path de contextos,
    los cruza usando el 'context_id'.
    """
    questions = load_json(questions_path)
    
    contexts_dict = {}
    if contexts_path:
        contexts = load_json(contexts_path)
        # Crear un diccionario para búsqueda rápida: {id: texto}
        contexts_dict = {item['id']: clean_text(item['text']) for item in contexts}
        
    dataset = []
    for q_item in questions:
        q_text = clean_text(q_item.get('question', ''))
        a_text = clean_text(q_item.get('answer', ''))
        c_id = q_item.get('context', None)
        c_text = contexts_dict.get(c_id, "") if c_id else ""
        
        dataset.append({
            "question": q_text,
            "answer": a_text,
            "context": c_text,
            "has_answer": bool(a_text),
            "has_context": bool(c_text)
        })
        
    return dataset

class CustomTokenizer:
    """
    Tokenizador mejorado a nivel de palabra que maneja minúsculas,
    puntuación básica y tokens especiales.
    """
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
        self.vocab_size = 3

    def _tokenize_string(self, text):
        # Convertir a minúsculas y separar puntuación básica
        text = text.lower()
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        return text.split()

    def build_vocab(self, texts):
        words = []
        for text in texts:
            words.extend(self._tokenize_string(text))
        
        for word in set(words):
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        tokens = self._tokenize_string(text)
        return [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in tokens] + [self.word2idx["<EOS>"]]

    def decode(self, indices):
        words = [self.idx2word.get(i, "<UNK>") for i in indices if i not in [0, 2]]
        # Unir y arreglar espacios antes de la puntuación
        text = " ".join(words)
        text = re.sub(r'\s+([.,!?()])', r'\1', text)
        return text