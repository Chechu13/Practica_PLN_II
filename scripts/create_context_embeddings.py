import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.retrieval.context_retriever import ContextRetriever


def main():
    parser = argparse.ArgumentParser(description="Genera embeddings para todos los contextos.")
    parser.add_argument(
        "--contexts",
        default=os.path.join(PROJECT_ROOT, "data", "contexts.json"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "models", "context_embeddings.pt"),
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        help="Modelo de Transformers usado para generar los embeddings.",
    )
    args = parser.parse_args()
    index = ContextRetriever.build_index(args.contexts, args.output, model_name=args.model)
    print(f"Embeddings guardados en: {args.output}")
    print(f"Contextos indexados: {len(index['context_ids'])}")
    print(f"Dimensiones: {index['embeddings'].shape[1]}")


if __name__ == "__main__":
    main()