"""Unit tests for the reasoning-prompt experiment (no model/GPU required)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "train_qwen_reasoning_prompt.py"
SPEC = importlib.util.spec_from_file_location("train_qwen_reasoning_prompt", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class DummyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "encoded prompt"

    def __call__(self, text, **kwargs):
        if text == "encoded prompt":
            return {"input_ids": [1, 2, 3]}
        return {"input_ids": [4, 5]}


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("zero_reasoning", "internally"),
        ("one_reasoning", "Example:"),
        ("few_reasoning", "Example 1:"),
    ],
)
def test_build_system_prompt(condition, expected, monkeypatch):
    monkeypatch.setattr(
        TRAIN,
        "PROMPTS_DIR",
        ROOT / "prompts",
    )
    examples = [
        {"context": f"context {i}", "question": f"question {i}", "answer": f"answer {i}"}
        for i in range(1, 5)
    ]
    prompt = TRAIN.build_system_prompt(condition, examples)
    assert expected in prompt
    assert "Do not reveal your analysis" in prompt
    assert "shortest complete answer" in prompt


def test_reasoning_dataset_keeps_final_answer_as_only_target():
    tokenizer = DummyTokenizer()
    dataset = TRAIN.QuestionAnsweringDataset(
        [{
            "context": "Evidence about Paris.",
            "question": "What is the capital?",
            "answer": "Paris",
        }],
        tokenizer,
        "Reason internally; return only the answer.",
    )
    item = dataset[0]
    assert item["input_ids"] == [1, 2, 3, 4, 5, 99]
    assert item["labels"] == [-100, -100, -100, 4, 5, 99]
    assert "Context:" in tokenizer.messages[1]["content"]
    assert "Question: What is the capital?" in tokenizer.messages[1]["content"]


def test_reasoning_dataset_rejects_missing_eos(monkeypatch):
    class MissingEosTokenizer(DummyTokenizer):
        eos_token_id = None

    dataset = TRAIN.QuestionAnsweringDataset(
        [{"context": "ctx", "question": "q", "answer": "a"}],
        MissingEosTokenizer(),
        "prompt",
    )
    with pytest.raises(ValueError, match="eos_token_id"):
        dataset[0]


def test_few_shot_requires_four_demonstrations():
    with pytest.raises(ValueError, match="cuatro ejemplos"):
        TRAIN.build_system_prompt(
            "few_reasoning",
            [{"context": "c", "question": "q", "answer": "a"}],
        )
