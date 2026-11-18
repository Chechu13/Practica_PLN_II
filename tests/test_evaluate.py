import math

import pytest

from src.evaluate import normalize_answer, exact_match_score, f1_score, bleu_score, rouge_score, evaluate_predictions


@pytest.mark.parametrize(("value", "expected"), [
    ("The quick, brown fox!", "quick brown fox"),
    ("  A   cat.  ", "cat"),
    ("AN apple", "apple"),
    ("It's John's book.", "its johns book"),
    ("", ""),
    (None, ""),
    (42, ""),
])
def test_normalize_answer(value, expected):
    assert normalize_answer(value) == expected


@pytest.mark.parametrize(("pred", "ref", "expected"), [
    ("The cat!", "cat", 1), ("cat", "dog", 0), ("", "", 1), ("cat", "", 0), (None, "", 1),
])
def test_exact_match(pred, ref, expected):
    assert exact_match_score(pred, ref) == expected


@pytest.mark.parametrize(("pred", "ref", "expected"), [
    ("red blue", "red blue", 1.0),
    ("red blue", "red green", 0.5),
    ("red", "red blue", 2 / 3),
    ("red blue", "green yellow", 0.0),
    ("", "", 1),
    ("red", "", 0),
])
def test_f1(pred, ref, expected):
    assert f1_score(pred, ref) == pytest.approx(expected)


def test_f1_duplicate_tokens():
    assert f1_score("cat cat dog", "cat dog dog") == pytest.approx(2 / 3)


@pytest.mark.parametrize(("pred", "ref", "expected"), [
    ("a b", "a b", 1.0),
    ("a b", "a c", 0.5),
    ("a", "a b", math.exp(-1)),
    ("a b", "c d", 0.0),
    ("", "", 1.0),
])
def test_bleu_1(pred, ref, expected):
    assert bleu_score(pred, ref) == pytest.approx(expected)


@pytest.mark.parametrize(("pred", "ref", "expected"), [
    ("a b", "a b", 1.0),
    ("a b", "a c", 0.5),
    ("a b c", "a b", 1.0),
    ("a", "a b", 0.5),
    ("a b", "c d", 0.0),
    ("", "", 1.0),
])
def test_rouge_1(pred, ref, expected):
    assert rouge_score(pred, ref) == pytest.approx(expected)


def test_evaluate_empty():
    assert evaluate_predictions([]) == {"exact_match": 0.0, "f1_score": 0.0, "total_evaluated": 0}


def test_evaluate_percentages():
    metrics = evaluate_predictions([
        {"prediction": "The cat", "reference": "cat"},
        {"prediction": "blue", "reference": "red"},
    ])
    assert metrics == {
        "exact_match": 50.0, "f1_score": 50.0, "bleu_score": 50.0,
        "rouge_score": 50.0, "total_evaluated": 2,
    }


def test_evaluate_missing_fields():
    metrics = evaluate_predictions([{}, {"prediction": "x"}])
    assert metrics["total_evaluated"] == 2
    assert metrics["exact_match"] == 50.0
    assert metrics["f1_score"] == 50.0


def test_evaluate_does_not_mutate_input():
    rows = [{"prediction": "a", "reference": "a"}]
    before = [r.copy() for r in rows]
    evaluate_predictions(rows)
    assert rows == before
