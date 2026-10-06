"""FastAPI service exposing ingestion, search and agentic Q&A."""
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import DocumentAgent
from .config import get_settings
from .embeddings import build_embedder
from .retrieval import STRATEGIES, HybridRetriever
from .vectorstore import build_store

ALLOWED_SUFFIXES = {".md", ".txt", ".pdf"}
STATIC = Path(__file__).parent / "static"
state: dict = {}


def build_app_state(settings=None) -> dict:
    settings = settings or get_settings()
    retriever = HybridRetriever(
        build_store(settings.vector_store, settings.chroma_dir),
        build_embedder(settings.embedder, settings.embedding_model),
        settings.chunk_words,
        settings.chunk_overlap,
    )
    if settings.load_sample_docs and not retriever.sources():
        for p in sorted(Path(settings.sample_docs_dir).glob("*")):
            if p.suffix.lower() in ALLOWED_SUFFIXES:
                retriever.add_file(p)
    agent = DocumentAgent(retriever, settings.openai_api_key, settings.openai_model)
    return {"retriever": retriever, "agent": agent}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.update(build_app_state())
    yield
    state.clear()


app = FastAPI(title="Agentic RAG Document Q&A", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    source: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    strategy: str = "hybrid"
    k: int = Field(default=5, ge=1, le=20)
    source: str | None = None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "documents": len(state["retriever"].sources()),
            "llm": state["agent"].client is not None}


@app.get("/documents")
def list_documents():
    return [{"source": s, "chunks": n} for s, n in sorted(state["retriever"].sources().items())]


@app.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Upload a {', '.join(sorted(ALLOWED_SUFFIXES))} file.")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        n = state["retriever"].add_file(dest)
    if n == 0:
        raise HTTPException(422, "No text could be extracted from that file.")
    return {"source": name, "chunks": n}


@app.delete("/documents/{source}")
def delete_document(source: str):
    if source not in state["retriever"].sources():
        raise HTTPException(404, f"No document named {source}.")
    state["retriever"].delete_document(source)
    return {"deleted": source}


@app.post("/search")
def search(req: SearchRequest):
    if req.strategy not in STRATEGIES:
        raise HTTPException(400, f"strategy must be one of {STRATEGIES}")
    hits = state["retriever"].search(req.query, req.k, req.strategy, req.source)
    return [h.to_dict() for h in hits]


@app.post("/ask")
def ask(req: AskRequest):
    result = state["agent"].ask(req.question, req.source)
    return result.__dict__
