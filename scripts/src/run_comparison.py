"""
Runs the full chunking x retrieval-method comparison against the dev set
(15 questions) and logs every combination to MLflow as a separate run.

Combinations: {structure, fixed} chunking x {dense, bm25, hybrid} retrieval
= 6 runs. Each run logs hit@5 and MRR, plus chunk-count and avg-chunk-size
as parameters so a later reader can see what produced each number without
re-running anything.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import mlflow
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from chunking import build_chunk_set
from retrieval import (
    EMBEDDING_MODEL, build_bm25_index, build_dense_index, bm25_search,
    dense_search, evaluate_retrieval, hybrid_search, load_questions,
)

CORPUS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../corpus")
QUESTIONS_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../eval/questions.json")
MLFLOW_TRACKING_URI = "sqlite:///" + str(Path("../../mlflow.db").resolve())
K = 5


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project-5-rag-retrieval")

    dev_questions = load_questions(QUESTIONS_PATH, split="dev")
    print(f"Loaded {len(dev_questions)} dev questions "
          f"({sum(1 for q in dev_questions if q['expected_source'] != 'none')} with a real expected_source)")

    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  loaded in {time.time() - t0:.1f}s")

    results = {}

    for strategy in ("structure", "fixed"):
        print(f"\n=== chunking strategy: {strategy} ===")
        chunks = build_chunk_set(CORPUS_ROOT, strategy)
        sizes = [c.word_count for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        print(f"  {len(chunks)} chunks, avg {avg_size:.1f} words")

        print("  embedding chunks and building FAISS index ...")
        t0 = time.time()
        dense_index, _ = build_dense_index(chunks, model)
        embed_time = time.time() - t0
        print(f"    done in {embed_time:.1f}s")

        print("  building BM25 index ...")
        t0 = time.time()
        bm25 = build_bm25_index(chunks)
        bm25_time = time.time() - t0
        print(f"    done in {bm25_time:.1f}s")

        methods = {
            "dense": lambda q, k: dense_search(q, model, dense_index, chunks, k),
            "bm25": lambda q, k: bm25_search(q, bm25, k),
            "hybrid": lambda q, k: hybrid_search(q, model, dense_index, bm25, chunks, k),
        }

        for method_name, retrieve_fn in methods.items():
            run_name = f"{strategy}-{method_name}"
            print(f"  evaluating {run_name} ...")

            eval_result = evaluate_retrieval(dev_questions, chunks, retrieve_fn, k=K)

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("chunking_strategy", strategy)
                mlflow.log_param("retrieval_method", method_name)
                mlflow.log_param("embedding_model", EMBEDDING_MODEL)
                mlflow.log_param("k", K)
                mlflow.log_param("split", "dev")
                mlflow.log_param("n_chunks", len(chunks))
                mlflow.log_param("avg_chunk_words", round(avg_size, 1))
                mlflow.log_metric("hit_at_k", eval_result["hit_at_k"])
                mlflow.log_metric("mrr", eval_result["mrr"])
                mlflow.log_metric("n_questions_scored", eval_result["n_questions"])
                if method_name == "dense":
                    mlflow.log_metric("embed_and_index_seconds", embed_time)
                if method_name == "bm25":
                    mlflow.log_metric("bm25_index_seconds", bm25_time)

                per_q_path = Path(tempfile.gettempdir()) / f"{run_name}_per_question.json"
                per_q_path.write_text(json.dumps(eval_result["per_question"], indent=2))
                mlflow.log_artifact(str(per_q_path))

            print(f"    hit@{K}={eval_result['hit_at_k']:.3f}  MRR={eval_result['mrr']:.3f}")
            results[run_name] = {
                "chunking_strategy": strategy,
                "retrieval_method": method_name,
                "n_chunks": len(chunks),
                "avg_chunk_words": round(avg_size, 1),
                "hit_at_k": eval_result["hit_at_k"],
                "mrr": eval_result["mrr"],
                "n_questions_scored": eval_result["n_questions"],
            }

    out_path = Path("../../results/dev_comparison.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")

    print("\n=== summary ===")
    print(f"{'run':<20} {'chunks':>7} {'avg_w':>7} {'hit@'+str(K):>7} {'mrr':>7}")
    for name, r in results.items():
        print(f"{name:<20} {r['n_chunks']:>7} {r['avg_chunk_words']:>7} "
              f"{r['hit_at_k']:>7.3f} {r['mrr']:>7.3f}")


if __name__ == "__main__":
    main()
