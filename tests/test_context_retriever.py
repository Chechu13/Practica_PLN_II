from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as functional

from src.retrieval.context_retriever import ContextRetriever


def make_retriever():
    retriever = ContextRetriever.__new__(ContextRetriever)
    retriever.device = torch.device("cpu")
    retriever.context_ids = ["a", "b", "c"]
    retriever.contexts = [
        "The capital of France is Paris.",
        "Berlin is the capital of Germany.",
        "Paris has museums and historic buildings.",
    ]
    retriever.embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    retriever.model_name = "dummy"
    retriever.term_document_frequency = {"capital": 2, "france": 1, "paris": 2}
    retriever.tokenizer = Mock(return_value={
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    })
    retriever.model = Mock()
    return retriever


def test_encode_batch_masked_mean_pooling():
    model = Mock()
    model.return_value.last_hidden_state = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [100.0, 100.0]]])
    batch = {"input_ids": torch.tensor([[1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}
    result = ContextRetriever._encode_batch(model, batch)
    expected = functional.normalize(torch.tensor([[0.5, 0.5]]), dim=1)
    assert torch.allclose(result, expected)


def test_encode_batch_all_padding_is_finite():
    model = Mock()
    model.return_value.last_hidden_state = torch.zeros((1, 2, 3))
    batch = {"input_ids": torch.tensor([[0, 0]]), "attention_mask": torch.tensor([[0, 0]])}
    assert torch.isfinite(ContextRetriever._encode_batch(model, batch)).all()


def test_search_combines_semantic_and_lexical_scores():
    retriever = make_retriever()
    retriever._encode_batch = Mock(return_value=torch.tensor([[1.0, 0.0]]))
    results = retriever.search("France capital", top_k=2)
    assert len(results) == 2
    assert results[0]["context_id"] == "a"
    assert results[0]["score"] == pytest.approx(1.15)
    assert results[0]["semantic_score"] == pytest.approx(1.0)
    assert results[0]["lexical_bonus"] == pytest.approx(0.15)


@pytest.mark.parametrize("top_k", [1, 2, 3, 20])
def test_search_respects_top_k(top_k):
    retriever = make_retriever()
    retriever._encode_batch = Mock(return_value=torch.tensor([[1.0, 0.0]]))
    assert len(retriever.search("question", top_k=top_k)) == min(top_k, 3)


def test_retrieve_returns_best_result():
    retriever = make_retriever()
    retriever._encode_batch = Mock(return_value=torch.tensor([[1.0, 0.0]]))
    assert retriever.retrieve("France capital")["context_id"] == "a"


def test_search_empty_index_returns_empty_list():
    retriever = make_retriever()
    retriever.context_ids, retriever.contexts = [], []
    retriever.embeddings = torch.empty((0, 2))
    retriever._encode_batch = Mock(return_value=torch.tensor([[1.0, 0.0]]))
    assert retriever.search("question") == []


def test_build_index_batches_and_saves_tiny_corpus(temp_json, tmp_path):
    path = temp_json("contexts.json", [
        {"id": "one", "text": "First context"},
        {"id": "two", "text": "Second context"},
    ])
    output = tmp_path / "nested" / "index.pt"
    tokenizer = Mock(side_effect=lambda texts, **kwargs: {
        "input_ids": torch.tensor([[1, 2]] * len(texts)),
        "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
    })
    model = Mock()
    model.to.return_value = model

    with patch("src.retrieval.context_retriever.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
        "src.retrieval.context_retriever.AutoModel.from_pretrained", return_value=model
    ), patch.object(ContextRetriever, "_encode_batch", return_value=torch.tensor([[1.0, 0.0, 0.0]])):
        index = ContextRetriever.build_index(path, output, model_name="dummy", device="cpu", batch_size=1)

    assert index["context_ids"] == ["one", "two"]
    assert index["contexts"] == ["First context", "Second context"]
    assert output.exists()
    assert torch.load(output, map_location="cpu", weights_only=True)["model_name"] == "dummy"
