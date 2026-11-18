import json

import pytest

from src.preprocess.preprocess import CustomTokenizer, clean_text, load_json, prepare_dataset


@pytest.mark.parametrize(("value", "expected"), [
    ("  Hello   world  ", "Hello world"),
    ("a\nb\t c", "a b c"),
    ("", ""),
    (None, ""),
    (123, ""),
    (["text"], ""),
])
def test_clean_text(value, expected):
    assert clean_text(value) == expected


def test_load_json_utf8(temp_json):
    assert load_json(temp_json("data.json", [{"name": "árbol"}])) == [{"name": "árbol"}]


def test_load_json_invalid_json(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_json(path)


def test_prepare_dataset_without_contexts(temp_json):
    questions = temp_json("questions.json", [
        {"question": "  Who   ?", "answer": "  Alice  ", "context": "c1"},
        {"question": "Why?", "answer": "", "context": None},
        {},
    ])
    assert prepare_dataset(questions) == [
        {"question": "Who ?", "answer": "Alice", "context": "", "has_answer": True, "has_context": False},
        {"question": "Why?", "answer": "", "context": "", "has_answer": False, "has_context": False},
        {"question": "", "answer": "", "context": "", "has_answer": False, "has_context": False},
    ]


def test_prepare_dataset_joins_contexts_by_id(temp_json):
    questions = temp_json("questions.json", [
        {"question": " Who? ", "answer": "A", "context": "known"},
        {"question": "Where?", "answer": "B", "context": "missing"},
        {"question": "No id", "answer": "C"},
    ])
    contexts = temp_json("contexts.json", [
        {"id": "known", "text": " A   useful\ncontext. "},
        {"id": "unused", "text": "Unused."},
    ])
    dataset = prepare_dataset(questions, contexts)
    assert dataset[0]["context"] == "A useful context."
    assert dataset[0]["has_context"] is True
    assert dataset[1]["context"] == ""
    assert dataset[2]["context"] == ""


def test_prepare_dataset_empty_files(temp_json):
    assert prepare_dataset(temp_json("q.json", []), temp_json("c.json", [])) == []


def test_tokenizer_special_tokens_are_stable():
    tokenizer = CustomTokenizer()
    assert tokenizer.word2idx == {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
    assert tokenizer.idx2word == {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
    assert tokenizer.vocab_size == 3


def test_tokenizer_build_encode_decode():
    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(["Hello, world!", "HELLO again"])
    encoded = tokenizer.encode("Hello, unknown!")
    assert encoded[-1] == tokenizer.word2idx["<EOS>"]
    assert tokenizer.word2idx["<UNK>"] in encoded
    assert len(encoded) == 4
    assert tokenizer.vocab_size == len(tokenizer.word2idx)
    assert tokenizer.decode(tokenizer.encode("HELLO, WORLD!")) == "hello, world!"


def test_tokenizer_decode_ignores_padding_and_eos():
    tokenizer = CustomTokenizer()
    tokenizer.build_vocab(["one two"])
    assert tokenizer.decode([0, tokenizer.word2idx["one"], tokenizer.word2idx["two"], 2, 0]) == "one two"


def test_tokenizer_decode_unknown_index():
    assert CustomTokenizer().decode([999]) == "<UNK>"


def test_tokenizer_empty_string():
    tokenizer = CustomTokenizer()
    assert tokenizer.encode("") == [2]
    assert tokenizer.decode([2]) == ""
