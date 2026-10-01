"""
Builds the search index the FastAPI service loads, using the frozen
retrieval configuration: fixed-window chunks (200 words, 40-word overlap),
BAAI/bge-small-en-v1.5 embeddings, FAISS inner-product index over
normalized vectors.

Writes to <project>/index/:
    faiss.index    the vector index
    chunks.json    chunk_id, source_path and text, in index order
    manifest.json  settings, counts and SHA-256 hashes the service checks
                   at startup, including the hash of corpus/manifest.json

After building, the dev questions are searched again and the top-5
chunk_ids are compared, in order, with results/rag_dev_retrieval.json (the
excerpts the RAG dev run used). Any difference exits non-zero, so the
service is known to retrieve exactly what was evaluated.

Usage (from scripts/src):
    python build_index.py ../../corpus
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from chunking import build_chunk_set
from retrieval import EMBEDDING_MODEL, build_dense_index

CHUNKING_STRATEGY = "fixed"
WINDOW_WORDS = 200
OVERLAP_WORDS = 40
K = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_against(reference_path: Path, model: SentenceTransformer, index: faiss.Index,
                   chunk_ids: list[str]) -> int:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    mismatches = 0
    max_score_diff = 0.0
    for q in reference["questions"]:
        q_emb = np.asarray(model.encode([q["question"]], normalize_embeddings=True), dtype="float32")
        scores, indices = index.search(q_emb, K)
        got = [chunk_ids[int(i)] for i in indices[0] if i != -1]
        expected = [r["chunk_id"] for r in q["retrieved"]]
        for r, s in zip(q["retrieved"], scores[0]):
            max_score_diff = max(max_score_diff, abs(r["score"] - float(s)))
        if got != expected:
            mismatches += 1
            print(f"  MISMATCH {q['id']}: expected {expected}, got {got}")
    print(f"Verified {len(reference['questions'])} questions against {reference_path}: "
          f"{mismatches} mismatch(es), max score difference {max_score_diff:.2e}")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the service's FAISS index from the pinned corpus.")
    parser.add_argument("corpus", type=Path, nargs="?", default=Path("../../corpus"))
    parser.add_argument("--out", type=Path, default=Path("../../index"))
    parser.add_argument("--verify-against", type=Path, default=Path("../../results/rag_dev_retrieval.json"))
    args = parser.parse_args()

    chunks = build_chunk_set(args.corpus, CHUNKING_STRATEGY)
    print(f"{len(chunks)} chunks ({CHUNKING_STRATEGY}, {WINDOW_WORDS} words, {OVERLAP_WORDS} overlap)")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding chunks and building FAISS index ...", flush=True)
    t0 = time.time()
    index, embeddings = build_dense_index(chunks, model)
    print(f"  done in {time.time() - t0:.1f}s")

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "faiss.index"
    chunks_path = args.out / "chunks.json"
    faiss.write_index(index, str(index_path))
    chunks_path.write_text(json.dumps(
        [{"chunk_id": c.chunk_id, "source_path": c.source_path, "text": c.text} for c in chunks],
        ensure_ascii=False), encoding="utf-8")

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "chunking_strategy": CHUNKING_STRATEGY,
        "window_words": WINDOW_WORDS,
        "overlap_words": OVERLAP_WORDS,
        "n_chunks": len(chunks),
        "dimension": int(embeddings.shape[1]),
        "corpus_manifest_sha256": sha256_file(args.corpus / "manifest.json"),
        "faiss_index_sha256": sha256_file(index_path),
        "chunks_sha256": sha256_file(chunks_path),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {index_path}, {chunks_path}, {args.out / 'manifest.json'}")

    if args.verify_against.exists():
        if verify_against(args.verify_against, model, index, [c.chunk_id for c in chunks]):
            print("Index does not reproduce the evaluated retrieval. Do not use it for the service.")
            sys.exit(1)
        print("Index reproduces the evaluated retrieval.")
    else:
        print(f"{args.verify_against} not found; retrieval reproduction was NOT verified.")


if __name__ == "__main__":
    main()
