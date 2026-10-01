import re

import torch
import torch.nn.functional as functional
from ddgs import DDGS
from transformers import AutoModel, AutoTokenizer


class WebRetriever:
    """
    Recupera información desde Internet y realiza un segundo
    ranking de los snippets utilizando similitud semántica.

    La búsqueda web NO se considera automáticamente fiable.
    """

    MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

    # Umbral mínimo para aceptar un resultado web.
    MIN_WEB_SCORE = 0.45

    # Número de resultados que pedimos al buscador.
    SEARCH_RESULTS = 5

    def __init__(self, device=None):
        self.device = torch.device(
            device or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)

        self.model.eval()

    @staticmethod
    def _encode_batch(model, batch):
        outputs = model(**batch)

        token_embeddings = outputs.last_hidden_state

        mask = (
            batch["attention_mask"]
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        pooled = (
            token_embeddings * mask
        ).sum(dim=1) / mask.sum(dim=1).clamp(
            min=1e-9
        )

        return functional.normalize(
            pooled,
            dim=1
        )

    def _embed_texts(self, texts):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():
            embeddings = self._encode_batch(
                self.model,
                encoded
            )

        return embeddings

    @staticmethod
    def _clean_snippet(text):
        if not text:
            return ""

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    @staticmethod
    def _has_useful_content(text):
        """
        Evita snippets demasiado cortos o prácticamente vacíos.
        """

        if not text:
            return False

        words = re.findall(
            r"\b\w+\b",
            text
        )

        return len(words) >= 8

    def _rank_results(self, question, results):
        """
        Ordena los resultados web por similitud semántica
        con la pregunta.
        """

        valid_results = []

        for result in results:
            body = self._clean_snippet(
                result.get("body", "")
            )

            if not self._has_useful_content(body):
                continue

            valid_results.append(
                {
                    "title": result.get(
                        "title",
                        ""
                    ),
                    "body": body,
                    "href": result.get(
                        "href",
                        ""
                    ),
                }
            )

        if not valid_results:
            return []

        texts = [
            result["body"]
            for result in valid_results
        ]

        question_embedding = self._embed_texts(
            [question]
        )

        snippet_embeddings = self._embed_texts(
            texts
        )

        scores = torch.matmul(
            question_embedding,
            snippet_embeddings.T
        )[0]

        ranked = []

        for result, score in zip(
            valid_results,
            scores
        ):
            ranked.append(
                {
                    **result,
                    "score": float(score)
                }
            )

        ranked.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked

    def retrieve(self, question):
        """
        Busca en Internet y devuelve solamente un contexto
        si encuentra un resultado suficientemente relevante.
        """

        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        question,
                        max_results=self.SEARCH_RESULTS
                    )
                )

            if not results:
                return self._empty_result()

            ranked = self._rank_results(
                question,
                results
            )

            if not ranked:
                return self._empty_result()

            best = ranked[0]

            print(
                f"🌐 Mejor resultado web: "
                f"{best['score']:.3f} | "
                f"{best['title']}"
            )

            if best["score"] < self.MIN_WEB_SCORE:
                print(
                    f"⚠️ Resultado web descartado: "
                    f"score {best['score']:.3f} "
                    f"< {self.MIN_WEB_SCORE}"
                )

                return self._empty_result()
            selected = ranked[:2]

            combined_context = "\n\n".join(
                [
                    (
                        f"Source: {item['title']}\n"
                        f"{item['body']}"
                    )
                    for item in selected
                ]
            )

            return {
                "context": combined_context,
                "context_id": best["href"],
                "score": best["score"],
                "source": "web",
                "sources": [
                    item["href"]
                    for item in selected
                ],
            }

        except Exception as e:
            print(
                f"❌ Error en la búsqueda web "
                f"de '{question}': {e}"
            )

            return self._empty_result()

    @staticmethod
    def _empty_result():
        return {
            "context": "",
            "context_id": "none",
            "score": 0.0,
            "source": "none",
            "sources": [],
        }