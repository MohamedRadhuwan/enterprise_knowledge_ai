from typing import Any
import logging
import os

import ollama

from retriever import search_documents

logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

_ollama_client: ollama.Client | None = None


def get_ollama_client() -> ollama.Client:
    """Initialize and retrieve the shared Ollama API client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=OLLAMA_HOST)
    return _ollama_client


def build_context_string(results: list[dict[str, Any]]) -> str:
    """Format retrieved document chunks into a structured context payload."""
    context_blocks = []
    for idx, res in enumerate(results, start=1):
        source = res.get("source", "Unknown Document")
        page = res.get("page")
        page_info = f"Page {page}" if page is not None else "Page N/A"
        text = res.get("text", "").strip()

        context_blocks.append(
            f"[Source #{idx}] ({source}, {page_info})\n{text}"
        )
    return "\n\n".join(context_blocks)


def generate_answer(question: str, top_k: int = 3) -> dict[str, Any]:
    """Execute RAG pipeline: retrieve context chunks and synthesize a grounded LLM answer."""
    results = search_documents(question, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant information was found in the indexed enterprise knowledge base.",
            "sources": [],
        }

    context = build_context_string(results)
    system_prompt = (
        "You are Enterprise Knowledge AI, an intelligent corporate assistant.\n"
        "Your task is to answer the user's question accurately using ONLY the provided CONTEXT.\n"
        "Strict Guidelines:\n"
        "1. Do NOT assume, speculate, or incorporate outside knowledge.\n"
        "2. If the context does not contain enough information to answer, state clearly that the document does not contain the answer.\n"
        "3. Maintain a professional, concise, and direct tone.\n"
        "4. Reference source file names when appropriate."
    )

    user_prompt = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}"

    try:
        client = get_ollama_client()
        response = client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer_text = response["message"]["content"]
    except Exception as exc:
        logger.error(f"LLM synthesis failed: {exc}", exc_info=True)
        raise RuntimeError(f"LLM communication error: {exc}") from exc

    sources = [
        {
            "source": r["source"],
            "page": r["page"],
            "score": r["score"],
            "snippet": r["text"][:200],
        }
        for r in results
    ]

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"RAG Pipeline test running on Ollama model: {LLM_MODEL} @ {OLLAMA_HOST}")
    sample_q = "What is the leave policy limit?"
    res = generate_answer(sample_q)
    print("\n--- Answer ---")
    print(res["answer"])
    print("\n--- Sources ---")
    for s in res["sources"]:
        print(f"File: {s['source']} | Page: {s['page']} | Score: {s['score']:.4f}")