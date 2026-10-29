"""Reproducible, retrieval-only audit for the local RAG corpus.

The script does not modify the production retriever, models, adapters, or
previous results. It writes all artifacts below a dedicated output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocess.preprocess import prepare_dataset
from src.retrieval.context_retriever import ContextRetriever


SEED = 42
TEST_SIZE = 0.2
MAX_K = 10
SEMANTIC_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
MARGINS = (0.0, 0.03, 0.06, 0.1)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_bm25(contexts: list[str]) -> tuple[list[Counter[str]], dict[str, int], float]:
    documents = [Counter(tokens(text)) for text in contexts]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())
    average_length = float(np.mean([sum(document.values()) for document in documents]))
    return documents, dict(document_frequency), average_length


def bm25_scores(
    query: str,
    documents: list[Counter[str]],
    document_frequency: dict[str, int],
    average_length: float,
    total_documents: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> np.ndarray:
    query_terms = Counter(tokens(query))
    scores = np.zeros(total_documents, dtype=np.float64)
    for term, query_frequency in query_terms.items():
        df = document_frequency.get(term, 0)
        if not df:
            continue
        idf = math.log(1.0 + (total_documents - df + 0.5) / (df + 0.5))
        term_scores = np.zeros(total_documents, dtype=np.float64)
        for index, document in enumerate(documents):
            term_frequency = document.get(term, 0)
            if not term_frequency:
                continue
            length = sum(document.values())
            denominator = term_frequency + k1 * (1.0 - b + b * length / average_length)
            term_scores[index] = idf * term_frequency * (k1 + 1.0) / denominator
        scores += term_scores * (1.0 + 0.05 * min(query_frequency, 3))
    return scores


def rank_indices(scores: np.ndarray, limit: int = MAX_K) -> list[int]:
    limit = min(limit, len(scores))
    return np.argsort(-scores, kind="stable")[:limit].tolist()


def rrf_rank(semantic: list[int], lexical: list[int], limit: int = MAX_K, k: int = 60) -> list[int]:
    combined: dict[int, float] = {}
    for rank, index in enumerate(semantic, start=1):
        combined[index] = combined.get(index, 0.0) + 1.0 / (k + rank)
    for rank, index in enumerate(lexical, start=1):
        combined[index] = combined.get(index, 0.0) + 1.0 / (k + rank)
    return [
        index
        for index, _ in sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def ranking_metrics(records: list[dict], method: str, k: int) -> dict:
    labeled = [record for record in records if record["gold_context_id"]]
    hits = [
        record["gold_context_id"] in record[f"{method}_ids"][:k]
        for record in labeled
    ]
    reciprocal_ranks = []
    for record in labeled:
        try:
            rank = record[f"{method}_ids"].index(record["gold_context_id"]) + 1
        except ValueError:
            rank = None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return {
        "method": method,
        "k": k,
        "n_labeled": len(labeled),
        "hit_rate": float(np.mean(hits)) if hits else 0.0,
        "recall_at_k": float(np.mean(hits)) if hits else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def abstention_metrics(records: list[dict], threshold: float, margin: float) -> dict:
    decisions = []
    for record in records:
        accepted = (
            record["semantic_score"] >= threshold
            and record["semantic_margin"] >= margin
        )
        decisions.append((record["has_answer"], accepted))
    answerable = [accepted for has_answer, accepted in decisions if has_answer]
    unanswerable = [accepted for has_answer, accepted in decisions if not has_answer]
    return {
        "threshold": threshold,
        "margin": margin,
        "coverage": float(np.mean([accepted for _, accepted in decisions])),
        "answerable_accept_rate": float(np.mean(answerable)) if answerable else 0.0,
        "unanswerable_abstention_rate": (
            float(np.mean([not accepted for accepted in unanswerable]))
            if unanswerable
            else 0.0
        ),
    }


def choose_abstention_config(dev_records: list[dict]) -> dict:
    candidates = [
        abstention_metrics(dev_records, threshold, margin)
        for threshold in SEMANTIC_THRESHOLDS
        for margin in MARGINS
    ]
    return max(
        candidates,
        key=lambda item: (
            item["answerable_accept_rate"] + item["unanswerable_abstention_rate"],
            -abs(item["coverage"] - 0.7),
        ),
    )


def write_plot(metrics: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for method, group in metrics.groupby("method"):
        axis.plot(group["k"], group["recall_at_k"] * 100, marker="o", label=method)
    axis.set(
        title="Retrieval recall by method",
        xlabel="top-k",
        ylabel="Recall@k (%)",
        xticks=sorted(metrics["k"].unique()),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "rag_audit_2026-10-29",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    dataset = prepare_dataset(
        ROOT / "data" / "questions_train.json",
        ROOT / "data" / "contexts.json",
    )
    raw_questions = json.loads(
        (ROOT / "data" / "questions_train.json").read_text(encoding="utf-8")
    )
    train, test = train_test_split(
        dataset,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=[item["has_answer"] for item in dataset],
    )
    fit, dev = train_test_split(
        train,
        test_size=0.2,
        random_state=SEED,
        stratify=[item["has_answer"] for item in train],
    )

    retriever = ContextRetriever(
        ROOT / "models" / "context_embeddings.pt",
        device=args.device,
    )
    documents, document_frequency, average_length = load_bm25(retriever.contexts)
    context_to_index = {context_id: index for index, context_id in enumerate(retriever.context_ids)}
    records = []

    for index, item in enumerate(dataset):
        semantic_started = time.perf_counter()
        semantic = retriever.search(item["question"], top_k=MAX_K)
        semantic_seconds = time.perf_counter() - semantic_started
        semantic_ids = [result["context_id"] for result in semantic]
        semantic_scores = [result["score"] for result in semantic]
        semantic_margin = (
            semantic_scores[0] - semantic_scores[1]
            if len(semantic_scores) > 1
            else float("inf")
        )

        bm25_started = time.perf_counter()
        lexical_scores = bm25_scores(
            item["question"],
            documents,
            document_frequency,
            average_length,
            len(documents),
        )
        lexical_ids = [
            retriever.context_ids[context_index]
            for context_index in rank_indices(lexical_scores)
        ]
        bm25_seconds = time.perf_counter() - bm25_started
        hybrid_ids = [
            retriever.context_ids[context_index]
            for context_index in rrf_rank(
                [context_to_index[context_id] for context_id in semantic_ids],
                rank_indices(lexical_scores),
            )
        ]
        records.append(
            {
                "index": index,
                "question": item["question"],
                "answer": item["answer"],
                "has_answer": item["has_answer"],
                "gold_context_id": raw_questions[index].get("context"),
                "semantic_ids": semantic_ids,
                "bm25_ids": lexical_ids,
                "hybrid_ids": hybrid_ids,
                "semantic_score": semantic_scores[0] if semantic_scores else 0.0,
                "semantic_margin": semantic_margin,
                "semantic_seconds": semantic_seconds,
                "bm25_seconds": bm25_seconds,
            }
        )

    split_by_question = {
        item["question"]: split
        for split, items in (("fit", fit), ("dev", dev), ("test", test))
        for item in items
    }
    for record in records:
        record["split"] = split_by_question[record["question"]]

    dev_records = [record for record in records if record["split"] == "dev"]
    chosen = choose_abstention_config(dev_records)
    metrics = pd.DataFrame(
        ranking_metrics(records, method, k)
        for method in ("semantic", "bm25", "hybrid")
        for k in (1, 3, 5, 10)
    )
    metrics.to_csv(args.output / "retrieval_metrics.csv", index=False)
    write_plot(metrics, args.output / "recall_at_k.png")

    threshold_rows = []
    for split in ("fit", "dev", "test"):
        split_records = [record for record in records if record["split"] == split]
        for threshold in SEMANTIC_THRESHOLDS:
            for margin in MARGINS:
                row = abstention_metrics(split_records, threshold, margin)
                row["split"] = split
                threshold_rows.append(row)
    pd.DataFrame(threshold_rows).to_csv(args.output / "abstention_sweep.csv", index=False)

    with (args.output / "rankings.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "seed": SEED,
        "dataset_size": len(dataset),
        "fit_size": len(fit),
        "dev_size": len(dev),
        "test_size": len(test),
        "labeled_context_questions": sum(bool(record["gold_context_id"]) for record in records),
        "unlabeled_questions": sum(not bool(record["gold_context_id"]) for record in records),
        "chosen_abstention_config_from_dev": chosen,
        "elapsed_seconds": time.perf_counter() - started,
        "device": args.device,
        "index_path": str(ROOT / "models" / "context_embeddings.pt"),
        "methods": ["semantic", "bm25", "hybrid_rrf"],
        "note": "Test metrics are descriptive only; threshold selection used dev exclusively.",
    }
    (args.output / "run_config.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "# RAG retrieval audit",
        "",
        "This report was generated by `scripts/experiments/audit_retrieval.py`.",
        "It does not evaluate Qwen generation and does not modify existing results.",
        "",
        f"- Dataset: {len(dataset)} questions; fit/dev/test = {len(fit)}/{len(dev)}/{len(test)}.",
        f"- Questions with annotated local context: {summary['labeled_context_questions']}.",
        f"- Questions without annotated local context: {summary['unlabeled_questions']}.",
        f"- Dev-selected abstention configuration: `{json.dumps(chosen)}`.",
        "",
        "Retrieval metrics are computed only on questions with an annotated context id.",
        "Abstention metrics use the dataset's answer-present flag and are reported separately.",
    ]
    (args.output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
