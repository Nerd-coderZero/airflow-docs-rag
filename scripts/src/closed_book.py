"""
Closed-book baseline: asks the questions of one split (--split, default dev)
to a locally served Ollama
model with no retrieval and no corpus context -- only the question, as a
user would type it with no RAG pipeline behind it.

Airflow's documentation is public and very likely in the model's
pretraining data. This run measures what the model already knows before
retrieval is credited with anything.

Answers are saved alongside each question's expected_answer to
results/closed_book_<split>_<model>.json for manual grading. No automated
string matching is used to score correctness.

The earlier hosted-API version (google/gemma-4-31b-it on NVIDIA's API
catalog) is kept unchanged in closed_book_nvidia.py.

With --framed, a single system message (DOMAIN_FRAMING) states the
domain; everything else is identical. Output files and MLflow runs carry a
"_framed" suffix so the two conditions never overwrite each other.

Usage (from scripts/src, Ollama running locally):
    python closed_book.py --model qwen2.5:7b-instruct-q4_K_M
    python closed_book.py --model qwen2.5:7b-instruct-q4_K_M --framed
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import mlflow
from openai import OpenAI

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_QUESTIONS_PATH = Path("../../eval/questions.json")
RESULTS_DIR = Path("../../results")
MLFLOW_TRACKING_URI = "sqlite:///" + str(Path("../../mlflow.db").resolve())

MAX_TOKENS = 768
TEMPERATURE = 0
SEED = 42
REQUEST_TIMEOUT_SECONDS = 900
MAX_RETRIES = 0
WARMUP_TIMEOUT_SECONDS = 600
KEEP_ALIVE = "60m"
DOMAIN_FRAMING = "You are answering questions about Apache Airflow."


def ollama_get(host: str, path: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(host + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_post(host: str, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def model_metadata(host: str, model: str) -> dict:
    try:
        version = ollama_get(host, "/api/version")["version"]
    except OSError as exc:
        raise SystemExit(
            f"Ollama is not reachable at {host} ({exc}). Start the Ollama app "
            f"and confirm with: Invoke-RestMethod {host}/api/version"
        )
    tags = ollama_get(host, "/api/tags").get("models", [])
    match = next((m for m in tags if m.get("name") == model or m.get("model") == model), None)
    if match is None:
        available = ", ".join(sorted(m.get("name", "") for m in tags)) or "none"
        raise SystemExit(f"Model {model} is not pulled. Run: ollama pull {model}  (pulled: {available})")
    details = match.get("details", {})
    return {
        "ollama_version": version,
        "model": model,
        "model_digest": match.get("digest"),
        "model_size_bytes": match.get("size"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "family": details.get("family"),
    }


def load_split_questions(path: Path, split: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    return [q for q in questions if q.get("split") == split]


def warm_up(host: str, model: str) -> float:
    t0 = time.perf_counter()
    ollama_post(host, "/api/generate", {"model": model, "keep_alive": KEEP_ALIVE}, WARMUP_TIMEOUT_SECONDS)
    return time.perf_counter() - t0


def build_messages(question: str, system_prompt: str | None) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": question})
    return messages


def ask_closed_book(client: OpenAI, model: str, question: str, system_prompt: str | None) -> dict:
    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=build_messages(question, system_prompt),
        temperature=TEMPERATURE,
        seed=SEED,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    elapsed = time.perf_counter() - t0
    choice = completion.choices[0]
    usage = completion.usage
    return {
        "answer": (choice.message.content or "").strip(),
        "finish_reason": choice.finish_reason,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "latency_seconds": round(elapsed, 2),
    }


def output_path(model: str, framed: bool, split: str) -> Path:
    suffix = "_framed" if framed else ""
    return RESULTS_DIR / f"closed_book_{split}_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-book baseline against a local Ollama model.")
    parser.add_argument("--model", required=True, help="Exact Ollama tag, e.g. qwen2.5:7b-instruct-q4_K_M")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--framed", action="store_true", help="Prepend the DOMAIN_FRAMING system message")
    args = parser.parse_args()
    system_prompt = DOMAIN_FRAMING if args.framed else None
    print(f"Condition: {'framed, system prompt: ' + repr(system_prompt) if system_prompt else 'unframed, no system prompt'}")

    meta = model_metadata(args.host, args.model)
    print(f"Ollama {meta['ollama_version']}, model {meta['model']} "
          f"digest {meta['model_digest']} ({meta['parameter_size']}, {meta['quantization_level']})")

    questions = load_split_questions(args.questions, args.split)
    n_adversarial = sum(1 for q in questions if q["expected_source"] == "none")
    print(f"Loaded {len(questions)} {args.split} questions "
          f"({len(questions) - n_adversarial} with a real expected_source, {n_adversarial} adversarial)")

    print("Loading model into memory ...")
    load_seconds = warm_up(args.host, args.model)
    print(f"  loaded in {load_seconds:.1f}s")

    client = OpenAI(
        base_url=args.host + "/v1",
        api_key="ollama",
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )

    results = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    run_start = time.perf_counter()
    for i, q in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['question'][:80]}", flush=True)
        try:
            response = ask_closed_book(client, args.model, q["question"], system_prompt)
            error = None
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}")
            response = {"answer": None, "finish_reason": "error", "prompt_tokens": None,
                        "completion_tokens": None, "latency_seconds": None}
        else:
            print(f"  {response['completion_tokens']} tokens in {response['latency_seconds']}s, "
                  f"finish_reason={response['finish_reason']}")
            if response["finish_reason"] == "length":
                print(f"  WARNING: answer truncated at max_tokens={MAX_TOKENS}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "expected_source": q["expected_source"],
            "expected_answer": q["expected_answer"],
            "model_answer": response["answer"],
            "finish_reason": response["finish_reason"],
            "prompt_tokens": response["prompt_tokens"],
            "completion_tokens": response["completion_tokens"],
            "latency_seconds": response["latency_seconds"],
            "error": error,
        })
    total_seconds = time.perf_counter() - run_start

    n_errors = sum(1 for r in results if r["model_answer"] is None)
    n_truncated = sum(1 for r in results if r["finish_reason"] == "length")

    out = {
        "run": {
            **meta,
            "split": args.split,
            "condition": "framed" if args.framed else "unframed",
            "system_prompt": system_prompt,
            "n_questions": len(questions),
            "temperature": TEMPERATURE,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "model_load_seconds": round(load_seconds, 1),
            "total_generation_seconds": round(total_seconds, 1),
            "n_errors": n_errors,
            "n_truncated": n_truncated,
            "started_at": started_at,
        },
        "answers": results,
    }
    path = output_path(args.model, args.framed, args.split)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(results)} answers to {path}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project-5-rag-retrieval")
    run_suffix = "-framed" if args.framed else ""
    with mlflow.start_run(run_name=f"closed-book-{args.split}-{args.model}{run_suffix}"):
        mlflow.log_params({
            "model": args.model,
            "model_digest": meta["model_digest"],
            "ollama_version": meta["ollama_version"],
            "quantization_level": meta["quantization_level"],
            "backend": "ollama",
            "split": args.split,
            "condition": "framed" if args.framed else "unframed",
            "system_prompt": system_prompt or "",
            "n_questions": len(questions),
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "seed": SEED,
        })
        mlflow.log_metric("n_errors", n_errors)
        mlflow.log_metric("n_truncated", n_truncated)
        mlflow.log_metric("total_generation_seconds", total_seconds)
        mlflow.log_artifact(str(path))

    print(f"{n_errors} error(s), {n_truncated} truncated at max_tokens={MAX_TOKENS}, "
          f"{total_seconds / 60:.1f} min generation time.")
    print(f"Correctness is not scored automatically -- grade {path} by hand.")
    if n_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
