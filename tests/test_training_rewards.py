import random
from unittest.mock import patch

import numpy as np

from src.training.train_grpo import reward_exact_match, reward_format_check, set_seed


def test_reward_exact_match_trims_whitespace():
    assert reward_exact_match([" answer ", "wrong"], ["answer", "answer"]) == [1.0, 0.0]


def test_reward_exact_match_empty_strings():
    assert reward_exact_match(["", " "], ["", ""]) == [1.0, 1.0]


def test_reward_exact_match_empty_batch():
    assert reward_exact_match([], []) == []


def test_reward_format_check():
    assert reward_format_check([" answer ", "", "  "]) == [0.5, 0.0, 0.0]


def test_reward_functions_accept_extra_kwargs():
    assert reward_exact_match(["x"], ["x"], prompts=["p"]) == [1.0]
    assert reward_format_check(["x"], prompts=["p"]) == [0.5]


def test_set_seed_reproducible_python_numpy():
    set_seed(17)
    first = random.random(), np.random.random()
    set_seed(17)
    assert first == (random.random(), np.random.random())


def test_set_seed_cuda_when_available():
    with patch("src.training.train_grpo.torch.cuda.is_available", return_value=True), patch(
        "src.training.train_grpo.torch.cuda.manual_seed_all"
    ) as mocked_seed:
        set_seed(123)
        mocked_seed.assert_called_once_with(123)
