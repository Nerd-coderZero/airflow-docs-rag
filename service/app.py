"""
FastAPI service over the evaluated RAG pipeline.

Retrieval and generation use the same code and settings as the evaluation
runs: the prompt, num_ctx, temperature, seed and token cap are imported
from scripts/src/rag_answer.py rather than restated, and the index is the
one written by scripts/src/build_index.py, which checks it reproduces the
dev-run retrieval.

Endpoints:
    GET  /health    index loaded, Ollama reachable, model pulled
    POST /retrieve  top-5 excerpts for a question, no model call
    POST /ask       grounded answer, sources, model and timing

Configuration (environment variables):
    RAG_MODEL       default qwen2.5:7b-instruct-q4_K_M
    RAG_OLLAMA_URL  default http://localhost:11434
    RAG_INDEX_DIR   default <project>/index
    RAG_CORPUS_DIR  default <project>/corpus

Run from the project root:
    uvicorn service.app:app --host 127.0.0.1 --port 8000
"""

import hashlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "src"))

import rag_answer as frozen
from ollama_native import DEFAULT_HOST, chat, model_metadata
from retrieval import EMBEDDING_MODEL

DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"
CHUNKING_STRATEGY = "fixed"
TOP_K = 5
MAX_QUESTION_CHARS = 1000
REQUEST_TIMEOUT_SECONDS = frozen.REQUEST_TIMEOUT_SECONDS


class IndexRejected(RuntimeError):
    pass


class OllamaUnavailable(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Retriever:
    def __init__(self, index_dir: Path, corpus_dir: Path, encoder=None):
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise IndexRejected(f"{manifest_path} not found. Run scripts/src/build_index.py first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.json"

        checks = [
            (self.manifest.get("embedding_model") == EMBEDDING_MODEL,
             f"embedding model {self.manifest.get('embedding_model')!r} != {EMBEDDING_MODEL!r}"),
            (self.manifest.get("chunking_strategy") == CHUNKING_STRATEGY,
             f"chunking strategy {self.manifest.get('chunking_strategy')!r} != {CHUNKING_STRATEGY!r}"),
            (self.manifest.get("corpus_manifest_sha256") == sha256_file(corpus_dir / "manifest.json"),
             "index was built from a different corpus manifest"),
            (self.manifest.get("faiss_index_sha256") == sha256_file(index_path),
             "faiss.index does not match its manifest hash"),
            (self.manifest.get("chunks_sha256") == sha256_file(chunks_path),
             "chunks.json does not match its manifest hash"),
        ]
        failed = [msg for ok, msg in checks if not ok]
        if failed:
            raise IndexRejected("Index rejected: " + "; ".join(failed) + ". Rebuild with build_index.py.")

        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not (self.index.ntotal == len(self.chunks) == self.manifest.get("n_chunks")):
            raise IndexRejected(f"Index rejected: {self.index.ntotal} vectors, {len(self.chunks)} chunks, "
                              f"manifest says {self.manifest.get('n_chunks')}.")

        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.encoder = encoder

    def search(self, question: str, k: int = TOP_K) -> list[dict]:
        q_emb = np.asarray(self.encoder.encode([question], normalize_embeddings=True), dtype="float32")
        scores, indices = self.index.search(q_emb, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            c = self.chunks[int(idx)]
            results.append({
                "rank": rank,
                "score": round(float(score), 6),
                "chunk_id": c["chunk_id"],
                "source_path": c["source_path"],
                "text": c["text"],
            })
        return results


class OllamaGenerator:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def options(self) -> dict:
        return {"temperature": frozen.TEMPERATURE, "seed": frozen.SEED,
                "num_predict": frozen.MAX_TOKENS, "num_ctx": frozen.NUM_CTX}

    def messages(self, question: str, retrieved: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": frozen.SYSTEM_PROMPT},
            {"role": "user", "content": frozen.build_user_message(question, retrieved)},
        ]

    def metadata(self) -> dict:
        try:
            return model_metadata(self.host, self.model)
        except SystemExit as exc:
            raise OllamaUnavailable(str(exc)) from None
        except OSError as exc:
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from None

    def generate(self, question: str, retrieved: list[dict]) -> dict:
        try:
            return chat(self.host, self.model, self.messages(question, retrieved),
                        self.options(), REQUEST_TIMEOUT_SECONDS)
        except OSError as exc:
            raise OllamaUnavailable(f"Ollama request failed: {exc}") from None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class AskRequest(QuestionRequest):
    include_excerpts: bool = False


def _clean_question(raw: str) -> str:
    question = raw.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question is empty")
    return question


def create_app(retriever: Retriever | None = None, generator: OllamaGenerator | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.retriever = retriever or Retriever(
            Path(os.environ.get("RAG_INDEX_DIR", PROJECT_ROOT / "index")),
            Path(os.environ.get("RAG_CORPUS_DIR", PROJECT_ROOT / "corpus")),
        )
        app.state.generator = generator or OllamaGenerator(
            os.environ.get("RAG_OLLAMA_URL", DEFAULT_HOST),
            os.environ.get("RAG_MODEL", DEFAULT_MODEL),
        )
        yield

    app = FastAPI(title="Airflow docs RAG", lifespan=lifespan)

    @app.get("/health")
    def health():
        r: Retriever = app.state.retriever
        g: OllamaGenerator = app.state.generator
        index_info = {
            "n_chunks": r.manifest["n_chunks"],
            "embedding_model": r.manifest["embedding_model"],
            "chunking_strategy": r.manifest["chunking_strategy"],
            "built_at": r.manifest.get("built_at"),
        }
        try:
            meta = g.metadata()
            ollama_info = {"reachable": True, "model": meta["model"], "model_digest": meta["model_digest"],
                           "ollama_version": meta["ollama_version"]}
            status_code, status = 200, "ok"
        except OllamaUnavailable as exc:
            ollama_info = {"reachable": False, "model": g.model, "error": str(exc)}
            status_code, status = 503, "degraded"
        return JSONResponse(status_code=status_code,
                            content={"status": status, "index": index_info, "ollama": ollama_info})

    @app.post("/retrieve")
    def retrieve(req: QuestionRequest):
        question = _clean_question(req.question)
        t0 = time.perf_counter()
        results = app.state.retriever.search(question, TOP_K)
        return {"question": question, "k": TOP_K, "excerpts": results,
                "retrieval_seconds": round(time.perf_counter() - t0, 3)}

    @app.post("/ask")
    def ask(req: AskRequest):
        question = _clean_question(req.question)
        g: OllamaGenerator = app.state.generator
        t0 = time.perf_counter()
        retrieved = app.state.retriever.search(question, TOP_K)
        retrieval_seconds = time.perf_counter() - t0
        try:
            meta = g.metadata()
            out = g.generate(question, retrieved)
        except OllamaUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))

        prompt_tokens = out.get("prompt_eval_count")
        if prompt_tokens is not None and prompt_tokens + frozen.MAX_TOKENS > frozen.NUM_CTX:
            raise HTTPException(status_code=500, detail=(
                f"Prompt of {prompt_tokens} tokens leaves less than {frozen.MAX_TOKENS} tokens of room "
                f"inside num_ctx={frozen.NUM_CTX}; the answer may be based on a cut prompt."))

        sources = []
        for r in retrieved:
            s = {k: r[k] for k in ("rank", "source_path", "score", "chunk_id")}
            if req.include_excerpts:
                s["text"] = r["text"]
            sources.append(s)

        return {
            "question": question,
            "answer": out.get("content"),
            "done_reason": out.get("done_reason"),
            "truncated": out.get("done_reason") == "length",
            "sources": sources,
            "model": {"name": meta["model"], "digest": meta["model_digest"],
                      "ollama_version": meta["ollama_version"]},
            "settings": {"k": TOP_K, "num_ctx": frozen.NUM_CTX, "temperature": frozen.TEMPERATURE,
                         "seed": frozen.SEED, "max_tokens": frozen.MAX_TOKENS},
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": out.get("eval_count")},
            "timing": {"retrieval_seconds": round(retrieval_seconds, 3),
                       "prompt_eval_seconds": out.get("prompt_eval_seconds"),
                       "eval_seconds": out.get("eval_seconds"),
                       "total_seconds": round(time.perf_counter() - t0, 3)},
        }

    return app


app = create_app()
