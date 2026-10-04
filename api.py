from pathlib import Path
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_pipeline import generate_answer
from vector_store import add_document_to_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("data/documents")
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}

app = FastAPI(
    title="Enterprise Knowledge AI API",
    description="High-performance, local RAG document intelligence microservice",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Target query for RAG search.")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context passages to retrieve.")


class SourceItem(BaseModel):
    source: str
    page: int | None = None
    score: float
    snippet: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int
    total_vectors: int


@app.get("/", summary="Root Health Endpoint")
def root():
    return {
        "status": "online",
        "service": "Enterprise Knowledge AI API",
        "version": "1.0.0",
    }


@app.get("/health", summary="Health Check")
def health_check():
    return {"status": "healthy"}


@app.post("/ask", response_model=AnswerResponse, summary="Query Knowledge Base")
def ask_question(request: QuestionRequest):
    logger.info(f"Processing ask request: '{request.question}'")
    query_text = request.question.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = generate_answer(query_text, top_k=request.top_k)
        return {
            "question": query_text,
            "answer": result["answer"],
            "sources": result["sources"],
        }
    except Exception as exc:
        logger.error(f"Error answering question: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating answer: {exc}",
        )


@app.post("/upload", response_model=UploadResponse, summary="Upload & Index Document")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Upload rejected for unsupported extension: {ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DOCUMENTS_DIR / filename

    if file_path.exists():
        logger.warning(f"Upload rejected for existing file: {filename}")
        raise HTTPException(
            status_code=409,
            detail=f"A document named '{filename}' already exists.",
        )

    try:
        content = await file.read()
        file_path.write_bytes(content)
        result = add_document_to_vector_store(file_path)
    except Exception as exc:
        logger.error(f"Failed to process uploaded file '{filename}': {exc}", exc_info=True)
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {exc}",
        )

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": filename,
        "chunks_added": result["chunks_added"],
        "total_vectors": result["total_vectors"],
    }