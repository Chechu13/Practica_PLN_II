from unittest.mock import Mock, patch

import pytest
import torch

from src.retrieval.web_retriever import WebRetriever


def test_clean_snippet_collapses_whitespace():
    assert WebRetriever._clean_snippet("  A\n useful\t snippet ") == "A useful snippet"


@pytest.mark.parametrize("text", ["", None, "one two three", " short text "])
def test_short_snippets_are_not_useful(text):
    assert not WebRetriever._has_useful_content(text)


def test_eight_words_are_useful():
    assert WebRetriever._has_useful_content("one two three four five six seven eight")


def test_empty_result_shape():
    assert WebRetriever._empty_result() == {
        "context": "", "context_id": "none", "score": 0.0, "source": "none", "sources": [],
    }


def test_rank_results_filters_and_sorts():
    retriever = WebRetriever.__new__(WebRetriever)
    retriever._embed_texts = Mock(side_effect=[
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.5, 0.5], [1.0, 0.0]]),
    ])
    ranked = retriever._rank_results("query", [
        {"title": "short", "body": "few words", "href": "bad"},
        {"title": "first", "body": "one two three four five six seven eight", "href": "first"},
        {"title": "best", "body": "another useful result with enough words to pass filter", "href": "best"},
    ])
    assert [item["href"] for item in ranked] == ["best", "first"]
    assert ranked[0]["score"] == pytest.approx(1.0)


def test_rank_results_empty_does_not_embed():
    retriever = WebRetriever.__new__(WebRetriever)
    retriever._embed_texts = Mock()
    assert retriever._rank_results("query", [{"body": "too short"}]) == []
    retriever._embed_texts.assert_not_called()


def test_retrieve_empty_search():
    retriever = WebRetriever.__new__(WebRetriever)
    with patch("src.retrieval.web_retriever.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.return_value = []
        assert retriever.retrieve("question") == WebRetriever._empty_result()


def test_retrieve_search_error_is_handled():
    retriever = WebRetriever.__new__(WebRetriever)
    with patch("src.retrieval.web_retriever.DDGS", side_effect=RuntimeError("offline")):
        assert retriever.retrieve("question") == WebRetriever._empty_result()


def test_retrieve_discards_low_score():
    retriever = WebRetriever.__new__(WebRetriever)
    retriever._rank_results = Mock(return_value=[
        {"title": "Low", "body": "snippet body", "href": "url", "score": 0.2}
    ])
    with patch("src.retrieval.web_retriever.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.return_value = [{"body": "valid snippet"}]
        assert retriever.retrieve("question") == WebRetriever._empty_result()


def test_retrieve_combines_two_best_results():
    retriever = WebRetriever.__new__(WebRetriever)
    retriever._rank_results = Mock(return_value=[
        {"title": "Best", "body": "best body", "href": "best-url", "score": 0.9},
        {"title": "Second", "body": "second body", "href": "second-url", "score": 0.8},
        {"title": "Third", "body": "third body", "href": "third-url", "score": 0.7},
    ])
    with patch("src.retrieval.web_retriever.DDGS") as ddgs:
        ddgs.return_value.__enter__.return_value.text.return_value = [{"body": "long enough snippet"}]
        result = retriever.retrieve("question")
    assert result["context_id"] == "best-url"
    assert result["sources"] == ["best-url", "second-url"]
    assert "Source: Best" in result["context"]
    assert "Source: Second" in result["context"]
    assert "Third" not in result["context"]
