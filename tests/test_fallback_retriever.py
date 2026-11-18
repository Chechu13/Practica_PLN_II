from unittest.mock import Mock

import pytest

from src.retrieval.fallback_retriever import FallbackRetriever


@pytest.mark.parametrize("question", [
    "", None, "   ", "...", "???", "no", "Hello", "Good morning", "Fix it",
    "What is my name?", "What is in my bag?", "Which of my files is newest?",
    "What did you say before?", "Is that the one?", "How much did she pay?",
])
def test_rejects_invalid_or_conversational_questions(question):
    assert not FallbackRetriever._looks_like_valid_question(question)


@pytest.mark.parametrize("question", [
    "Who discovered penicillin?", "What is the capital of France?",
    "When was the first iPhone released?", "Explain photosynthesis in plants.",
])
def test_accepts_knowledge_questions(question):
    assert FallbackRetriever._looks_like_valid_question(question)


def make_retriever(local_result=None, accepted=False, web_result=None):
    instance = FallbackRetriever.__new__(FallbackRetriever)
    instance.threshold = 0.60
    instance.margin = 0.03
    instance.local_retriever = Mock()
    instance.web_retriever = Mock()
    instance._retrieve_local = Mock(return_value=(local_result, accepted))
    instance.web_retriever.retrieve.return_value = web_result or {
        "context": "", "context_id": "none", "score": 0.0, "source": "none", "sources": [],
    }
    instance.local_retriever.context_ids = ["c1"]
    return instance


def test_returns_local_when_accepted():
    retriever = make_retriever({"context": "Local evidence", "context_id": "c1", "score": 0.9}, True)
    result = retriever.retrieve("Who discovered penicillin?")
    assert result["context"] == "Local evidence"
    assert result["source"] == result["retrieval"] == "local"
    retriever.web_retriever.retrieve.assert_not_called()


def test_uses_web_when_local_is_not_accepted():
    web = {"context": "Web evidence", "context_id": "url", "score": 0.8, "source": "web", "sources": ["url"]}
    retriever = make_retriever({"context": "weak", "score": 0.2}, False, web)
    result = retriever.retrieve("Who discovered penicillin?")
    assert result["source"] == result["retrieval"] == "web"
    assert result["context"] == "Web evidence"


@pytest.mark.parametrize("score", [0.0, 0.4499])
def test_rejects_web_below_threshold(score):
    retriever = make_retriever(None, False, {"context": "Weak evidence", "score": score, "context_id": "url"})
    assert retriever.retrieve("Who discovered penicillin?") == {
        "context": "", "context_id": "none", "score": 0.0,
        "source": "none", "retrieval": "no_reliable_context",
    }


def test_accepts_web_at_exact_threshold():
    retriever = make_retriever(None, False, {"context": "Accepted", "score": 0.45, "context_id": "url"})
    assert retriever.retrieve("Who discovered penicillin?")["source"] == "web"


def test_rejects_empty_web_context_even_with_high_score():
    retriever = make_retriever(None, False, {"context": "", "score": 0.99, "context_id": "url"})
    assert retriever.retrieve("Who discovered penicillin?")["source"] == "none"


def test_invalid_question_skips_retrievers():
    retriever = make_retriever()
    assert retriever.retrieve("hello")["retrieval"] == "rejected_question"
    retriever._retrieve_local.assert_not_called()
    retriever.web_retriever.retrieve.assert_not_called()


def test_context_ids_property():
    assert make_retriever().context_ids == ["c1"]


@pytest.mark.parametrize(("scores", "threshold", "margin", "expected"), [
    ([0.8, 0.7], 0.6, 0.03, True),
    ([0.59, 0.2], 0.6, 0.03, False),
    ([0.8, 0.79], 0.6, 0.03, False),
    ([0.6], 0.6, 0.03, True),
])
def test_local_acceptance_policy(scores, threshold, margin, expected):
    instance = FallbackRetriever.__new__(FallbackRetriever)
    instance.threshold, instance.margin = threshold, margin
    instance.local_retriever = Mock()
    instance.local_retriever.search.return_value = [{"score": score} for score in scores]
    _, accepted = instance._retrieve_local("question")
    assert accepted is expected


def test_local_retriever_empty_candidates():
    instance = FallbackRetriever.__new__(FallbackRetriever)
    instance.threshold, instance.margin = 0.6, 0.03
    instance.local_retriever = Mock()
    instance.local_retriever.search.return_value = []
    assert instance._retrieve_local("question") == (None, False)
