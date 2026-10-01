"""
Retrieval step for the RAG runs, using the frozen configuration from
docs/DESIGN.md: fixed-window chunking (200 words, 40-word overlap), dense
retrieval with BAAI/bge-small-en-v1.5 over a FAISS inner-product index,
top-5 chunks per question.

Retrieval runs once and its output is saved, so every generation model is
given exactly the same excerpts for the same question. hit@5 and MRR are
recomputed here as a consistency check against the frozen comparison
(fixed-dense: hit@5 0.917, MRR 0.722) for the dev split; a mismatch
exits non-zero. The test split has no earlier number to check against;
its hit@5/MRR from this script are the test-set retrieval result.

Usage (from scripts/src):
    python retrieve.py ../../corpus ../../eval/questions.json --split dev
    python retrieve.py ../../corpus ../../eval/questions.json --split test
"""

import argparse

import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from chunking import build_chunk_set
from retrieval import EMBEDDING_MODEL, _expected_sources, build_dense_index, load_questions

RESULTS_DIR = Path("../../results")
MLFLOW_TRACKING_URI = "sqlite:///" + str(Path("../../mlflow.db").resolve())

CHUNKING_STRATEGY = "fixed"
K = 5
FROZEN_FIXED_DENSE = (0.917, 0.722)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen dense retrieval for one eval split.")
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("../../corpus"))
    parser.add_argument("questions_path", type=Path, nargs="?", default=Path("../../eval/questions.json"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    out_path = RESULTS_DIR / f"rag_{args.split}_retrieval.json"

    questions = load_questions(args.questions_path, split=args.split)
    print(f"Loaded {len(questions)} {args.split} questions")

    chunks = build_chunk_set(args.corpus, CHUNKING_STRATEGY)
    avg_words = sum(c.word_count for c in chunks) / len(chunks)
    print(f"{len(chunks)} chunks ({CHUNKING_STRATEGY}), avg {avg_words:.1f} words")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding chunks and building FAISS index ...")
    t0 = time.time()
    index, _ = build_dense_index(chunks, model)
    embed_seconds = time.time() - t0
    print(f"  done in {embed_seconds:.1f}s")

    records = []
    hits, rrs = [], []
    for q in questions:
        q_emb = np.asarray(model.encode([q["question"]], normalize_embeddings=True), dtype="float32")
        scores, indices = index.search(q_emb, K)
        retrieved = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            c = chunks[int(idx)]
            retrieved.append({
                "rank": rank,
                "score": round(float(score), 6),
                "chunk_id": c.chunk_id,
                "source_path": c.source_path,
                "word_count": c.word_count,
                "text": c.text,
            })

        expected = _expected_sources(q)
        hit, rr = None, None
        if expected:
            sources = [r["source_path"] for r in retrieved]
            hit = any(s in expected for s in sources)
            rr = next((1.0 / r["rank"] for r in retrieved if r["source_path"] in expected), 0.0)
            hits.append(hit)
            rrs.append(rr)

        records.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "expected_source": q["expected_source"],
            "expected_answer": q["expected_answer"],
            "hit_at_5": hit,
            "reciprocal_rank": rr,
            "retrieved": retrieved,
        })
        print(f"  {q['id']}: hit={hit} top1={retrieved[0]['source_path'] if retrieved else None}")

    hit_at_k = sum(hits) / len(hits)
    mrr = sum(rrs) / len(rrs)
    out = {
        "run": {
            "split": args.split,
            "chunking_strategy": CHUNKING_STRATEGY,
            "retrieval_method": "dense",
            "embedding_model": EMBEDDING_MODEL,
            "k": K,
            "n_chunks": len(chunks),
            "avg_chunk_words": round(avg_words, 1),
            "n_scored": len(hits),
            "hit_at_5": round(hit_at_k, 3),
            "mrr": round(mrr, 3),
            "embed_and_index_seconds": round(embed_seconds, 1),
        },
        "questions": records,
    }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    reference = "  (frozen dev comparison: 0.917 / 0.722)" if args.split == "dev" else ""
    print(f"\nhit@{K}={hit_at_k:.3f}  MRR={mrr:.3f}{reference}")
    print(f"Saved retrieved excerpts to {out_path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project-5-rag-retrieval")
    with mlflow.start_run(run_name=f"rag-{args.split}-retrieval"):
        mlflow.log_params({
            "split": args.split,
            "chunking_strategy": CHUNKING_STRATEGY,
            "retrieval_method": "dense",
            "embedding_model": EMBEDDING_MODEL,
            "k": K,
            "n_chunks": len(chunks),
        })
        mlflow.log_metric("hit_at_k", hit_at_k)
        mlflow.log_metric("mrr", mrr)
        mlflow.log_metric("embed_and_index_seconds", embed_seconds)
        mlflow.log_artifact(str(out_path))

    if args.split == "dev" and (round(hit_at_k, 3), round(mrr, 3)) != FROZEN_FIXED_DENSE:
        print(f"MISMATCH with frozen fixed-dense result {FROZEN_FIXED_DENSE}. Do not start the RAG run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
