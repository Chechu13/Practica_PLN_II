# Testing and code quality

The project uses **pytest**. The unit suite is designed to run on CPU without
downloading Transformer checkpoints, making live web requests, or training
models. Model loading, search-engine access and filesystem operations are
mocked where appropriate; small tensor-level tests exercise the PyTorch models.

## Install development dependencies

```bash
uv sync --extra test --extra quality
```

## Run tests

```bash
uv run pytest
```

Run a specific module or subset:

```bash
uv run pytest tests/test_preprocess.py
uv run pytest tests/test_fallback_retriever.py -k threshold
```

Coverage report:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Static analysis

The static-analysis tools are optional development dependencies. Run them from
the repository root:

```bash
uv run flake8 src tests
uv run bandit -c pyproject.toml -r src
uv run vulture src --min-confidence 80
uv run mypy src
```

The configuration is intentionally conservative at first. Review each finding
in context rather than suppressing warnings globally: some dynamic patterns
used by PyTorch and Transformers can be difficult for static analyzers to
infer. Add narrow, documented exclusions only when a finding is confirmed to
be a false positive.

## Scope

The suite covers preprocessing and tokenization, QA metrics, semantic retrieval
ranking and lexical scoring, hybrid local/web fallback decisions, web snippet
filtering/ranking/error handling, Transformer tensor shapes and positional
boundaries, and pure GRPO reward/seed helpers.

Training loops, pretrained checkpoint loading, real search-engine calls, and
full-corpus embedding generation are integration or hardware-dependent tasks.
They are intentionally excluded from the default unit test run.

## Continuous integration

GitHub Actions runs the CPU unit-test suite on pushes and pull requests to
`master` and `main`. The static-analysis commands above can also be run locally
before opening a pull request.
