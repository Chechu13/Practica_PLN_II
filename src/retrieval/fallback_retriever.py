import re

from src.retrieval.context_retriever import ContextRetriever
from src.retrieval.web_retriever import WebRetriever


class FallbackRetriever:
    """
    Sistema híbrido de recuperación:

        1. Intenta encontrar contexto local.
        2. Comprueba que el resultado local sea suficientemente fiable.
        3. Si no lo es, utiliza Internet.
        4. Si Internet tampoco es suficientemente relevante,
           devuelve contexto vacío.

    El objetivo es evitar introducir información de baja calidad
    en el LLM.
    """

    # Score mínimo para aceptar contexto local.
    LOCAL_THRESHOLD = 0.60

    # Diferencia mínima entre el primer y segundo resultado.
    # Si los dos resultados son prácticamente igual de buenos,
    # significa que el retrieval no tiene un ganador claro.
    LOCAL_MARGIN = 0.03

    # Score mínimo de la búsqueda web.
    WEB_THRESHOLD = 0.45

    def __init__(
        self,
        index_path,
        device,
        threshold=None,
        margin=None,
    ):
        self.local_retriever = ContextRetriever(
            index_path,
            device=device
        )

        self.web_retriever = WebRetriever(
            device=device
        )

        self.threshold = (
            threshold
            if threshold is not None
            else self.LOCAL_THRESHOLD
        )

        self.margin = (
            margin
            if margin is not None
            else self.LOCAL_MARGIN
        )

    @staticmethod
    def _looks_like_valid_question(question):
        """
        Filtro básico para evitar búsquedas web absurdas.

        No pretende ser un clasificador perfecto.
        Simplemente evita enviar preguntas claramente
        conversacionales, incompletas o dependientes del usuario.
        """

        if not question:
            return False

        question = question.strip()

        if len(question) < 4:
            return False

        # Preguntas que no contienen información suficiente.
        invalid_exact = {
            "...",
            "???",
            "??",
            "?",
            "42.",
            "hello.",
            "hello",
            "good morning.",
            "good morning",
            "no.",
            "no",
            "fine by me.",
            "fine by me",
            "actually, never mind.",
            "actually, never mind",
            "undo that.",
            "undo that",
            "fix it.",
            "fix it",
            "continue from there.",
            "continue from there",
            "repeat what you said at the beginning.",
            "repeat what you said at the beginning",
        }

        if question.lower() in invalid_exact:
            return False

        # Preguntas claramente dependientes de información privada
        # o del estado actual del usuario.
        personal_patterns = [
            r"\bmy\b.*\bright now\b",
            r"\bmy\b.*\bbehind me\b",
            r"\bmy\b.*\bwearing\b",
            r"\bwhat is my name\b",
            r"\bwhen is my birthday\b",
            r"\bwhat('?s| is) in my\b",
            r"\bwhich of my\b",
            r"\bwhat did you say\b",
            r"\bwhat you said\b",
            r"\bis that the one\b",
            r"\bthe one you meant\b",
            r"\bwho told them\b",
            r"\band the other\b",
            r"\bhow much did she pay\b",
        ]

        for pattern in personal_patterns:
            if re.search(
                pattern,
                question.lower()
            ):
                return False

        # Frases que son órdenes/conversación y no preguntas de conocimiento.
        conversational_patterns = [
            r"^(hello|hi|hey)\b",
            r"^(good morning|good evening)\b",
            r"^(no|yes)\.?$",
            r"^(fix|undo|continue|repeat)\b",
            r"^(actually,? never mind)\b",
        ]

        for pattern in conversational_patterns:
            if re.search(
                pattern,
                question.lower()
            ):
                return False

        return True

    def _retrieve_local(self, question):
        """
        Recupera los 3 candidatos locales y calcula si el primero
        tiene una ventaja suficientemente clara.
        """

        candidates = self.local_retriever.search(
            question,
            top_k=3
        )

        if not candidates:
            return None, False

        best = candidates[0]

        if len(candidates) >= 2:
            second = candidates[1]

            margin = (
                best["score"]
                - second["score"]
            )
        else:
            margin = float("inf")

        best["margin"] = margin

        # Condición de aceptación local.
        score_ok = (
            best["score"]
            >= self.threshold
        )

        margin_ok = (
            margin >= self.margin
        )

        accepted = (
            score_ok
            and margin_ok
        )

        print(
            f"📚 Local | "
            f"score={best['score']:.3f} | "
            f"margin={margin:.3f} | "
            f"accepted={accepted}"
        )

        return best, accepted

    def retrieve(self, question):
        """
        Recuperación híbrida local + web.
        """

        # ---------------------------------------------------------
        # 0. Validar pregunta
        # ---------------------------------------------------------

        if not self._looks_like_valid_question(
            question
        ):
            print(
                f"🚫 Pregunta no apta para retrieval: "
                f"{question}"
            )

            return {
                "context": "",
                "context_id": "none",
                "score": 0.0,
                "source": "none",
                "retrieval": "rejected_question",
            }

        # ---------------------------------------------------------
        # 1. Buscar en corpus local
        # ---------------------------------------------------------

        local_result, local_accepted = (
            self._retrieve_local(question)
        )

        if local_accepted:
            return {
                **local_result,
                "source": "local",
                "retrieval": "local",
            }

        # ---------------------------------------------------------
        # 2. Local insuficiente → Internet
        # ---------------------------------------------------------

        if local_result is not None:
            print(
                f"🌐 Confianza local insuficiente "
                f"({local_result['score']:.3f}). "
                f"Buscando en web..."
            )
        else:
            print(
                "🌐 Sin resultado local. "
                "Buscando en web..."
            )

        web_result = (
            self.web_retriever.retrieve(
                question
            )
        )

        # ---------------------------------------------------------
        # 3. Comprobar resultado web
        # ---------------------------------------------------------

        if (
            web_result["context"]
            and web_result["score"]
            >= self.WEB_THRESHOLD
        ):
            print(
                f"✅ Usando contexto web "
                f"(score={web_result['score']:.3f})"
            )

            return {
                **web_result,
                "source": "web",
                "retrieval": "web",
            }

        # ---------------------------------------------------------
        # 4. Web tampoco es suficientemente fiable
        # ---------------------------------------------------------

        print(
            "⚠️ Ni el contexto local ni la web "
            "han producido un contexto suficientemente fiable."
        )

        return {
            "context": "",
            "context_id": "none",
            "score": 0.0,
            "source": "none",
            "retrieval": "no_reliable_context",
        }

    @property
    def context_ids(self):
        return self.local_retriever.context_ids