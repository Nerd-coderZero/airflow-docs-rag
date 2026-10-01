"""
Retrieval-augmented run for one split (--split, default dev): answers each
question from the excerpts saved by retrieve.py, so every model sees
identical context.

Prompt: the system message starts with the same framing line as the framed
closed-book run, followed by a grounding rule (answer only from the
excerpts, say so when they do not cover the question). The user message
lists the 5 excerpts, each labelled with its corpus path, then the
question. No citation is requested.

Generation goes through Ollama's native /api/chat with num_ctx set
explicitly; temperature 0, seed 42 and the 768-token cap match the
closed-book runs. A prompt whose evaluated length leaves less than
max_tokens of room inside num_ctx is flagged, since the server would
otherwise drop part of it silently.

Usage (from scripts/src, after retrieve.py and context_check.py):
    python rag_answer.py --model qwen2.5:7b-instruct-q4_K_M --split dev
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlflow

from ollama_native import DEFAULT_HOST, chat, model_metadata, post_json

RESULTS_DIR = Path("../../results")
MLFLOW_TRACKING_URI = "sqlite:///" + str(Path("../../mlflow.db").resolve())

DOMAIN_FRAMING = "You are answering questions about Apache Airflow."
GROUNDING_RULE = (
    "Answer using only the documentation excerpts provided with the question. "
    "If the excerpts do not contain the answer, say that the provided documentation "
    "does not cover it instead of answering from other knowledge."
)
SYSTEM_PROMPT = f"{DOMAIN_FRAMING}\n{GROUNDING_RULE}"

NUM_CTX = 8192
MAX_TOKENS = 768
TEMPERATURE = 0
SEED = 42
REQUEST_TIMEOUT_SECONDS = 1800
WARMUP_TIMEOUT_SECONDS = 600


def build_user_message(question: str, retrieved: list[dict]) -> str:
    parts = ["Documentation excerpts:"]
    for r in retrieved:
        parts.append(f"[{r['rank']}] Source: {r['source_path']}\n{r['text']}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG dev run against a local Ollama model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--retrieval", type=Path, default=None)
    args = parser.parse_args()
    if args.retrieval is None:
        args.retrieval = RESULTS_DIR / f"rag_{args.split}_retrieval.json"

    if not args.retrieval.exists():
        raise SystemExit(f"{args.retrieval} not found. Run retrieve.py --split {args.split} first.")
    retrieval = json.loads(args.retrieval.read_text(encoding="utf-8"))
    if retrieval["run"]["split"] != args.split:
        raise SystemExit(f"{args.retrieval} holds the {retrieval['run']['split']} split, not {args.split}.")
    questions = retrieval["questions"]

    meta = model_metadata(args.host, args.model)
    print(f"Ollama {meta['ollama_version']}, model {meta['model']} digest {meta['model_digest']}")
    print(f"{len(questions)} {args.split} questions, k={retrieval['run']['k']}, num_ctx={NUM_CTX}")

    print("Loading model into memory ...")
    t0 = time.perf_counter()
    post_json(args.host, "/api/generate",
              {"model": args.model, "keep_alive": "60m", "options": {"num_ctx": NUM_CTX}},
              WARMUP_TIMEOUT_SECONDS)
    load_seconds = time.perf_counter() - t0
    print(f"  loaded in {load_seconds:.1f}s")

    options = {"temperature": TEMPERATURE, "seed": SEED, "num_predict": MAX_TOKENS, "num_ctx": NUM_CTX}
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_start = time.perf_counter()
    answers = []
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:80]}", flush=True)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(q["question"], q["retrieved"])},
        ]
        error = None
        try:
            r = chat(args.host, args.model, messages, options, REQUEST_TIMEOUT_SECONDS)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}")
            r = {"content": None, "done_reason": "error", "prompt_eval_count": None, "eval_count": None,
                 "prompt_eval_seconds": None, "eval_seconds": None, "load_seconds": None,
                 "latency_seconds": None}
        context_room_ok = None
        if r["prompt_eval_count"] is not None:
            context_room_ok = r["prompt_eval_count"] + MAX_TOKENS <= NUM_CTX
            print(f"  prompt {r['prompt_eval_count']} tok in {r['prompt_eval_seconds']}s, "
                  f"answer {r['eval_count']} tok in {r['eval_seconds']}s, done_reason={r['done_reason']}")
            if not context_room_ok:
                print(f"  WARNING: prompt + max_tokens exceeds num_ctx={NUM_CTX}")
            if r["done_reason"] == "length":
                print(f"  WARNING: answer truncated at max_tokens={MAX_TOKENS}")

        answers.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "expected_source": q["expected_source"],
            "expected_answer": q["expected_answer"],
            "retrieval_hit_at_5": q["hit_at_5"],
            "retrieved_sources": [x["source_path"] for x in q["retrieved"]],
            "model_answer": r["content"],
            "done_reason": r["done_reason"],
            "prompt_tokens": r["prompt_eval_count"],
            "completion_tokens": r["eval_count"],
            "prompt_eval_seconds": r["prompt_eval_seconds"],
            "eval_seconds": r["eval_seconds"],
            "latency_seconds": r["latency_seconds"],
            "context_room_ok": context_room_ok,
            "error": error,
        })
    total_seconds = time.perf_counter() - run_start

    n_errors = sum(1 for a in answers if a["model_answer"] is None)
    n_truncated = sum(1 for a in answers if a["done_reason"] == "length")
    n_context_overflow = sum(1 for a in answers if a["context_room_ok"] is False)
    max_prompt = max((a["prompt_tokens"] or 0) for a in answers)

    out = {
        "run": {
            **meta,
            "split": args.split,
            "condition": "rag",
            "system_prompt": SYSTEM_PROMPT,
            "retrieval_file": str(args.retrieval),
            "retrieval_config": retrieval["run"],
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "model_load_seconds": round(load_seconds, 1),
            "total_generation_seconds": round(total_seconds, 1),
            "max_prompt_tokens": max_prompt,
            "n_errors": n_errors,
            "n_truncated": n_truncated,
            "n_context_overflow": n_context_overflow,
            "started_at": started_at,
        },
        "answers": answers,
    }
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    path = RESULTS_DIR / f"rag_{args.split}_{safe}.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(answers)} answers to {path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project-5-rag-retrieval")
    with mlflow.start_run(run_name=f"rag-{args.split}-{args.model}"):
        mlflow.log_params({
            "model": args.model,
            "model_digest": meta["model_digest"],
            "ollama_version": meta["ollama_version"],
            "backend": "ollama-native",
            "split": args.split,
            "condition": "rag",
            "system_prompt": SYSTEM_PROMPT,
            "chunking_strategy": retrieval["run"]["chunking_strategy"],
            "retrieval_method": retrieval["run"]["retrieval_method"],
            "embedding_model": retrieval["run"]["embedding_model"],
            "k": retrieval["run"]["k"],
            "num_ctx": NUM_CTX,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
        })
        mlflow.log_metric("n_errors", n_errors)
        mlflow.log_metric("n_truncated", n_truncated)
        mlflow.log_metric("n_context_overflow", n_context_overflow)
        mlflow.log_metric("max_prompt_tokens", max_prompt)
        mlflow.log_metric("total_generation_seconds", total_seconds)
        mlflow.log_artifact(str(path))

    print(f"{n_errors} error(s), {n_truncated} truncated, {n_context_overflow} context overflow(s), "
          f"max prompt {max_prompt} tokens, {total_seconds / 60:.1f} min generation time.")
    if n_errors or n_context_overflow:
        sys.exit(1)


if __name__ == "__main__":
    main()
