from pathlib import Path
from typing import Any
import json
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from document_ingestion import create_chunks, load_all_documents, load_document

logger = logging.getLogger(__name__)

VECTORSTORE_DIR = Path("data/vectorstore")
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def save_metadata(metadata: list[dict[str, Any]]) -> None:
    """Save chunk metadata array to JSON storage."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata() -> list[dict[str, Any]]:
    """Load existing chunk metadata from JSON storage."""
    if not METADATA_PATH.exists():
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def add_document_to_vector_store(file_path: Path) -> dict[str, int]:
    """Process a single document file, compute embeddings, and update the FAISS index."""
    logger.info(f"Indexing document: {file_path.name}")
    documents = load_document(file_path)
    if not documents:
        raise ValueError(
            f"Unable to extract text from file '{file_path.name}'."
        )

    chunks = create_chunks(documents)
    if not chunks:
        raise ValueError(
            f"No text chunks could be generated from '{file_path.name}'."
        )

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
    else:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))

    metadata = load_metadata()
    metadata.extend(chunks)
    save_metadata(metadata)

    logger.info(
        f"Indexed {len(chunks)} chunks from {file_path.name}. Total FAISS vectors: {index.ntotal}"
    )
    return {"chunks_added": len(chunks), "total_vectors": index.ntotal}


def build_vector_store_from_scratch() -> dict[str, int]:
    """Build a fresh FAISS index and metadata store from all files in data/documents."""
    documents = load_all_documents()
    chunks = create_chunks(documents)
    if not chunks:
        logger.warning("No documents available for indexing.")
        return {"chunks_added": 0, "total_vectors": 0}

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    save_metadata(chunks)

    logger.info(
        f"Built index with {len(documents)} documents, {len(chunks)} chunks, and {index.ntotal} vectors."
    )
    return {"chunks_added": len(chunks), "total_vectors": index.ntotal}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = build_vector_store_from_scratch()
    print(f"Vector store initial build complete: {res}")