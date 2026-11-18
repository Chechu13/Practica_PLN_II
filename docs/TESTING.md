# Testing

The project uses **pytest**. The unit suite is designed to run on CPU without
downloading Transformer checkpoints, making live web requests, or training
models. Model loading, search-engine access and filesystem operations are
mocked where appropriate; small tensor-level tests exercise the PyTorch models.

## Run locally

```bash
uv sync --extra test
uv run pytest
```

Run a specific module or subset:

```bash
uv run pytest tests/test_preprocess.py
uv run pytest tests/test_fallback_retriever.py -k threshold
```

Optional coverage report:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Scope

The suite covers preprocessing and tokenization, QA metrics, semantic retrieval
ranking and lexical scoring, hybrid local/web fallback decisions, web snippet
filtering/ranking/error handling, Transformer tensor shapes and positional
boundaries, and pure GRPO reward/seed helpers.

Training loops, pretrained checkpoint loading, real search-engine calls, and
full-corpus embedding generation are integration or hardware-dependent tasks.
They are intentionally excluded from the default unit test run.
