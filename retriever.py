from pathlib import Path
from typing import Any
import json
import logging

import faiss
import numpy as np

from vector_store import INDEX_PATH, METADATA_PATH, get_embedding_model

logger = logging.getLogger(__name__)


def load_vector_store() -> tuple[faiss.Index | None, list[dict[str, Any]]]:
    """Load the FAISS index and metadata store from disk."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, []

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def search_documents(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Execute cosine similarity search over the FAISS vector index."""
    index, metadata = load_vector_store()
    if index is None or not metadata:
        logger.warning(
            "Vector store index or metadata not found. Return empty results."
        )
        return []

    model = get_embedding_model()
    query_vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or idx >= len(metadata):
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_query = "What is the policy?"
    matches = search_documents(test_query, top_k=2)
    print(f"Query: {test_query}")
    for i, match in enumerate(matches, 1):
        print(
            f"[{i}] Score: {match['score']:.4f} | Source: {match['source']} (Page: {match['page']})"
        )
        print(f"Content snippet: {match['text'][:150]}...\n")