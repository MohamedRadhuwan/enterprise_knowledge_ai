from pathlib import Path
from typing import Any
import logging

from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")


def load_txt(file_path: Path) -> list[dict[str, Any]]:
    """Extract raw text from a UTF-8 encoded text file."""
    text = file_path.read_text(encoding="utf-8")
    return [{"text": text, "source": file_path.name, "page": 1}]


def load_pdf(file_path: Path) -> list[dict[str, Any]]:
    """Extract page-level text from a PDF document."""
    reader = PdfReader(file_path)
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            documents.append(
                {"text": text, "source": file_path.name, "page": page_number}
            )
    return documents


def load_docx(file_path: Path) -> list[dict[str, Any]]:
    """Extract body text from a Microsoft Word (.docx) document."""
    doc = Document(file_path)
    text = "\n".join(
        paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()
    )
    return [{"text": text, "source": file_path.name, "page": None}]


def load_document(file_path: Path) -> list[dict[str, Any]]:
    """Route document parsing based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        logger.warning(f"Unsupported file format skipped: {file_path.name}")
        return []


def load_all_documents(
    directory: Path = DOCUMENTS_DIR,
) -> list[dict[str, Any]]:
    """Load all supported documents from the target directory."""
    if not directory.exists():
        return []
    documents = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            documents.extend(load_document(file_path))
    return documents


def create_chunks(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    """Split documents into overlapping text chunks for vector embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["text"])
        for chunk_text in split_texts:
            chunks.append(
                {
                    "text": chunk_text,
                    "source": doc["source"],
                    "page": doc["page"],
                }
            )
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = load_all_documents()
    print(f"Loaded {len(docs)} document pages/sections.")
    chunked = create_chunks(docs)
    print(f"Created {len(chunked)} chunks.")