import json
import os
import re

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer


class ContextRetriever:
    """Busca el contexto más cercano a una pregunta por similitud coseno."""

    def __init__(self, index_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        index = torch.load(index_path, map_location="cpu", weights_only=True)
        self.context_ids = index["context_ids"]
        self.contexts = index["contexts"]
        self.embeddings = functional.normalize(index["embeddings"].float(), dim=1)
        self.model_name = index["model_name"]
        self.term_document_frequency = {}
        for context in self.contexts:
            for term in set(re.findall(r"[a-zA-ZÀ-ÿ]{4,}", context.lower())):
                self.term_document_frequency[term] = self.term_document_frequency.get(term, 0) + 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    @classmethod
    def build_index(
        cls,
        contexts_path,
        index_path,
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        device=None,
        batch_size=32,
        max_length=512,
    ):
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with open(contexts_path, encoding="utf-8") as contexts_file:
            raw_contexts = json.load(contexts_file)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        texts = [item["text"] for item in raw_contexts]
        embeddings = []
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                embeddings.append(cls._encode_batch(model, batch).cpu())

        index = {
            "model_name": model_name,
            "context_ids": [item["id"] for item in raw_contexts],
            "contexts": texts,
            "embeddings": torch.cat(embeddings),
        }
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        torch.save(index, index_path)
        return index

    @staticmethod
    def _encode_batch(model, batch):
        outputs = model(**batch)
        token_embeddings = outputs.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return functional.normalize(pooled, dim=1)

    def search(self, question, top_k=1):
        encoded = self.tokenizer(
            [question], padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            question_embedding = self._encode_batch(self.model, encoded).cpu()
        semantic_scores = torch.matmul(question_embedding, self.embeddings.T)[0]
        query_terms = {
            term
            for term in re.findall(r"[a-zA-ZÀ-ÿ]{4,}", question.lower())
            if self.term_document_frequency.get(term, 0) <= 5
        }
        lexical_bonus = torch.tensor(
            [
                0.5
                if query_terms.intersection(
                    set(re.findall(r"[a-zA-ZÀ-ÿ]{4,}", context.lower()))
                )
                else 0.0
                for context in self.contexts
            ],
            dtype=semantic_scores.dtype,
        )
        scores = semantic_scores + lexical_bonus
        values, indices = torch.topk(scores, k=min(top_k, len(self.context_ids)))
        return [
            {
                "context_id": self.context_ids[index],
                "context": self.contexts[index],
                "score": float(value),
                "semantic_score": float(semantic_scores[index]),
            }
            for value, index in zip(values, indices)
        ]

    def retrieve(self, question):
        return self.search(question, top_k=1)[0]