"""
Retrieval comparison: dense (FAISS) vs BM25 vs hybrid, over both chunking
strategies. Scored against the eval set's dev split using hit@k and MRR,
where "hit" means the retrieved chunk's source_path matches (one of) the
question's expected_source path(s).

Embedding model: BAAI/bge-small-en-v1.5, chosen for being small enough to
run on CPU in this environment while being a competitive open-weight
retrieval embedding model. Run entirely offline after the model download
(no per-query API cost, no network dependency for reproducing results
later), which was the reason an API embedding model was rejected during
the original scoping conversation.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunking import Chunk, build_chunk_set

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_questions(path: Path, split: str = None) -> list[dict]:
    with open(path) as f:
        qs = json.load(f)
    if split:
        qs = [q for q in qs if q["split"] == split]
    return qs


def _expected_sources(q: dict) -> set[str]:
    src = q["expected_source"]
    if src == "none":
        return set()
    return set(src) if isinstance(src, list) else {src}


def build_dense_index(chunks: list[Chunk], model: SentenceTransformer) -> tuple[faiss.Index, np.ndarray]:
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def dense_search(query: str, model: SentenceTransformer, index: faiss.Index, chunks: list[Chunk], k: int) -> list[int]:
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    _, indices = index.search(q_emb, k)
    return [int(i) for i in indices[0] if i != -1]


def bm25_search(query: str, bm25: BM25Okapi, k: int) -> list[int]:
    scores = bm25.get_scores(query.lower().split())
    ranked = np.argsort(scores)[::-1][:k]
    return [int(i) for i in ranked]


def hybrid_search(query: str, model: SentenceTransformer, dense_index: faiss.Index,
                   bm25: BM25Okapi, chunks: list[Chunk], k: int, dense_weight: float = 0.5) -> list[int]:
    """
    Reciprocal rank fusion of dense and BM25 rankings. RRF is used instead
    of a raw score blend because dense (cosine similarity, bounded 0-1) and
    BM25 (unbounded, corpus-frequency dependent) scores are not on
    comparable scales; RRF sidesteps that by ranking on position, not score
    magnitude.
    """
    pool_k = min(len(chunks), max(k * 5, 50))
    dense_ranked = dense_search(query, model, dense_index, chunks, pool_k)
    bm25_ranked = bm25_search(query, bm25, pool_k)

    rrf_scores = {}
    C = 60
    for rank, idx in enumerate(dense_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + dense_weight / (C + rank + 1)
    for rank, idx in enumerate(bm25_ranked):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - dense_weight) / (C + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return [idx for idx, _ in ranked[:k]]


def evaluate_retrieval(questions: list[dict], chunks: list[Chunk], retrieve_fn, k: int = 5) -> dict:
    """
    retrieve_fn(query, k) -> list of chunk indices, ranked best-first.
    Adversarial questions (expected_source == "none") are excluded from
    hit@k/MRR -- there is no source to hit. They are handled separately
    at the answer-generation stage, not the retrieval-scoring stage.
    """
    hits_at_k = []
    reciprocal_ranks = []
    per_question = []

    for q in questions:
        expected = _expected_sources(q)
        if not expected:
            continue

        retrieved_indices = retrieve_fn(q["question"], k)
        retrieved_sources = [chunks[i].source_path for i in retrieved_indices]

        hit = any(src in expected for src in retrieved_sources)
        hits_at_k.append(hit)

        rr = 0.0
        for rank, src in enumerate(retrieved_sources, start=1):
            if src in expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        per_question.append({
            "id": q["id"],
            "hit": hit,
            "reciprocal_rank": rr,
            "expected": sorted(expected),
            "retrieved_top3": retrieved_sources[:3],
        })

    n = len(hits_at_k)
    return {
        "n_questions": n,
        "hit_at_k": sum(hits_at_k) / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "per_question": per_question,
    }
