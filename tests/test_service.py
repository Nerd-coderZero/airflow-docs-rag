import ast
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient

import rag_answer as frozen
from service import app as service

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIM = 16


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(DIM).astype("float32")
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


class FakeGenerator:
    def __init__(self, prompt_tokens=1500, done_reason="stop", reachable=True):
        self.model = "fake:1"
        self.prompt_tokens = prompt_tokens
        self.done_reason = done_reason
        self.reachable = reachable
        self.calls = []

    def metadata(self):
        if not self.reachable:
            raise service.OllamaUnavailable("Ollama is not reachable")
        return {"model": self.model, "model_digest": "abc", "ollama_version": "0.34.4"}

    def generate(self, question, retrieved):
        if not self.reachable:
            raise service.OllamaUnavailable("Ollama is not reachable")
        self.calls.append((question, retrieved))
        return {"content": "grounded answer", "done_reason": self.done_reason,
                "prompt_eval_count": self.prompt_tokens, "eval_count": 12,
                "prompt_eval_seconds": 1.0, "eval_seconds": 2.0}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fake_index(root: Path, n_chunks: int = 12) -> tuple[Path, Path, list[dict]]:
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(json.dumps([{"source_path": "a.rst", "dest_path": "a.rst"}]))
    chunks = [{"chunk_id": f"c{i}", "source_path": f"docs/file{i % 4}.rst", "text": f"chunk text number {i}"}
              for i in range(n_chunks)]
    enc = FakeEncoder()
    vectors = enc.encode([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    idx_dir = root / "index"
    idx_dir.mkdir()
    faiss.write_index(index, str(idx_dir / "faiss.index"))
    (idx_dir / "chunks.json").write_text(json.dumps(chunks))
    manifest = {
        "embedding_model": service.EMBEDDING_MODEL,
        "chunking_strategy": "fixed",
        "n_chunks": n_chunks,
        "dimension": DIM,
        "corpus_manifest_sha256": sha(corpus / "manifest.json"),
        "faiss_index_sha256": sha(idx_dir / "faiss.index"),
        "chunks_sha256": sha(idx_dir / "chunks.json"),
        "built_at": "2026-10-01T00:00:00+0530",
    }
    (idx_dir / "manifest.json").write_text(json.dumps(manifest))
    return idx_dir, corpus, chunks


@pytest.fixture
def index_files(tmp_path):
    return build_fake_index(tmp_path)


@pytest.fixture
def retriever(index_files):
    idx_dir, corpus, _ = index_files
    return service.Retriever(idx_dir, corpus, encoder=FakeEncoder())


def client_for(retriever, generator):
    return TestClient(service.create_app(retriever=retriever, generator=generator))


def test_search_ranks_exact_text_first(retriever):
    results = retriever.search("chunk text number 7")
    assert [r["rank"] for r in results] == [1, 2, 3, 4, 5]
    assert results[0]["chunk_id"] == "c7"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert set(results[0]) == {"rank", "score", "chunk_id", "source_path", "text"}


@pytest.mark.parametrize("field,value,message", [
    ("embedding_model", "other-model", "embedding model"),
    ("chunking_strategy", "structure", "chunking strategy"),
    ("corpus_manifest_sha256", "0" * 64, "different corpus manifest"),
    ("faiss_index_sha256", "0" * 64, "faiss.index does not match"),
])
def test_index_rejected_on_manifest_mismatch(index_files, field, value, message):
    idx_dir, corpus, _ = index_files
    manifest = json.loads((idx_dir / "manifest.json").read_text())
    manifest[field] = value
    (idx_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(service.IndexRejected, match=message):
        service.Retriever(idx_dir, corpus, encoder=FakeEncoder())


def test_index_rejected_on_count_mismatch(index_files):
    idx_dir, corpus, _ = index_files
    manifest = json.loads((idx_dir / "manifest.json").read_text())
    manifest["n_chunks"] = 999
    (idx_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(service.IndexRejected, match="manifest says 999"):
        service.Retriever(idx_dir, corpus, encoder=FakeEncoder())


def test_index_missing(tmp_path):
    with pytest.raises(service.IndexRejected, match="build_index.py"):
        service.Retriever(tmp_path, tmp_path, encoder=FakeEncoder())


def test_health_ok(retriever):
    with client_for(retriever, FakeGenerator()) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index"]["n_chunks"] == 12
    assert body["ollama"]["reachable"] is True


def test_health_degraded_when_ollama_down(retriever):
    with client_for(retriever, FakeGenerator(reachable=False)) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert r.json()["ollama"]["reachable"] is False


def test_retrieve_returns_five_excerpts_with_text(retriever):
    gen = FakeGenerator()
    with client_for(retriever, gen) as c:
        r = c.post("/retrieve", json={"question": "chunk text number 3"})
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 5 and len(body["excerpts"]) == 5
    assert body["excerpts"][0]["chunk_id"] == "c3"
    assert all("text" in e for e in body["excerpts"])
    assert gen.calls == []


def test_ask_returns_answer_sources_without_text_by_default(retriever):
    gen = FakeGenerator()
    with client_for(retriever, gen) as c:
        r = c.post("/ask", json={"question": "  chunk text number 5  "})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "chunk text number 5"
    assert body["answer"] == "grounded answer"
    assert body["truncated"] is False
    assert len(body["sources"]) == 5 and body["sources"][0]["chunk_id"] == "c5"
    assert all("text" not in s for s in body["sources"])
    assert body["settings"] == {"k": 5, "num_ctx": frozen.NUM_CTX, "temperature": frozen.TEMPERATURE,
                                "seed": frozen.SEED, "max_tokens": frozen.MAX_TOKENS}
    assert body["usage"]["prompt_tokens"] == 1500
    question, retrieved = gen.calls[0]
    assert question == "chunk text number 5" and len(retrieved) == 5


def test_ask_include_excerpts(retriever):
    with client_for(retriever, FakeGenerator()) as c:
        r = c.post("/ask", json={"question": "chunk text number 5", "include_excerpts": True})
    assert all("text" in s for s in r.json()["sources"])


def test_ask_flags_truncated_answer(retriever):
    with client_for(retriever, FakeGenerator(done_reason="length")) as c:
        r = c.post("/ask", json={"question": "q"})
    assert r.status_code == 200 and r.json()["truncated"] is True


def test_ask_rejects_context_overflow(retriever):
    tokens = frozen.NUM_CTX - frozen.MAX_TOKENS + 1
    with client_for(retriever, FakeGenerator(prompt_tokens=tokens)) as c:
        r = c.post("/ask", json={"question": "q"})
    assert r.status_code == 500
    assert "num_ctx" in r.json()["detail"]


def test_ask_accepts_prompt_at_room_limit(retriever):
    tokens = frozen.NUM_CTX - frozen.MAX_TOKENS
    with client_for(retriever, FakeGenerator(prompt_tokens=tokens)) as c:
        r = c.post("/ask", json={"question": "q"})
    assert r.status_code == 200


def test_ask_503_when_ollama_down(retriever):
    with client_for(retriever, FakeGenerator(reachable=False)) as c:
        r = c.post("/ask", json={"question": "q"})
    assert r.status_code == 503


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {}])
def test_invalid_questions_rejected(retriever, payload):
    with client_for(retriever, FakeGenerator()) as c:
        assert c.post("/ask", json=payload).status_code == 422
        assert c.post("/retrieve", json=payload).status_code == 422


def test_generator_sends_evaluated_prompt_and_options(monkeypatch):
    captured = {}

    def fake_chat(host, model, messages, options, timeout):
        captured.update(host=host, model=model, messages=messages, options=options)
        return {"content": "x", "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 1}

    monkeypatch.setattr(service, "chat", fake_chat)
    retrieved = [{"rank": i, "source_path": f"s{i}.rst", "text": f"t{i}"} for i in range(1, 6)]
    g = service.OllamaGenerator("http://h:1", "m:1")
    g.generate("What is X?", retrieved)
    assert captured["options"] == {"temperature": 0, "seed": 42, "num_predict": 768, "num_ctx": 8192}
    assert captured["messages"][0] == {"role": "system", "content": frozen.SYSTEM_PROMPT}
    assert captured["messages"][0]["content"].startswith("You are answering questions about Apache Airflow.\n")
    assert captured["messages"][1]["content"] == frozen.build_user_message("What is X?", retrieved)


def test_generator_maps_network_error_to_unavailable(monkeypatch):
    def failing_chat(*args, **kwargs):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(service, "chat", failing_chat)
    with pytest.raises(service.OllamaUnavailable):
        service.OllamaGenerator("http://h:1", "m:1").generate("q", [])


def test_service_k_matches_frozen_retrieval_script():
    tree = ast.parse((PROJECT_ROOT / "scripts" / "src" / "retrieve.py").read_text(encoding="utf-8"))
    values = {t.id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
              for t in n.targets if isinstance(t, ast.Name) and isinstance(n.value, ast.Constant)}
    assert values["K"] == service.TOP_K
    assert values["CHUNKING_STRATEGY"] == service.CHUNKING_STRATEGY
