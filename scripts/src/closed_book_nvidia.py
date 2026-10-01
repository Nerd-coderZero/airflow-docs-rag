"""
Closed-book baseline: asks the 15 dev questions to the model chosen for
generation (google/gemma-4-31b-it, via NVIDIA's free API catalog), with
no retrieval and no corpus context -- just the question, exactly as a
user would type it with no RAG pipeline behind it at all.

Why this exists: Airflow's documentation is public and very likely already
in this model's pretraining data. If the model answers well with no
retrieval, hit@5 alone does not tell us the RAG pipeline is useful --
it may just be measuring retrieval mechanics on a topic the model already
knows. This run has to happen before the real generation layer is built,
so the framing of every later result accounts for what the model already
knows on its own.

This script logs no hit@k/MRR (there is no retrieval to score). Instead it
saves every raw answer, alongside the question's expected_answer, to
results/closed_book_dev.json for manual review, and logs one summary
MLflow run so it sits next to the six retrieval runs already recorded.
Automated string-matching against expected_answer is deliberately NOT used
to score correctness here -- these are free-text answers, and a naive
substring/overlap check would misrepresent both false negatives (a
correct answer phrased differently) and false positives (a wrong answer
that happens to share words with the expected one). Correctness is judged
by reading the saved answers, not by a script guessing at them.
"""

import json
import os
import sys
import time
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from openai import OpenAI

MODEL_ID = "google/gemma-4-31b-it"
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

QUESTIONS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../eval/questions.json")
ENV_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../../.env")
MLFLOW_TRACKING_URI = "sqlite:///" + str(Path("../../mlflow.db").resolve())
OUT_PATH = Path("../../results/closed_book_dev.json")

MAX_TOKENS = 768
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 2.0


def load_dev_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        questions = json.load(f)
    return [q for q in questions if q.get("split") == "dev"]


def load_api_key(env_path: Path) -> str:
    if not env_path.exists():
        raise SystemExit(
            f".env file not found at {env_path}. Create a file at that path "
            f"containing a line NVIDIA_API_KEY=your-key-here, or pass its "
            f"path as the second command-line argument. This key is never "
            f"written into this script or committed to the repository."
        )
    load_dotenv(env_path)
    key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            f"{env_path} was found but has no NVIDIA_API_KEY=... line, or it is empty."
        )
    return key


def ask_closed_book(client: OpenAI, question: str) -> dict:
    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": question}],
        temperature=0,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    choice = completion.choices[0]
    message = choice.message
    return {
        "answer": (message.content or "").strip(),
        "finish_reason": choice.finish_reason,
        "reasoning_content_present": bool(getattr(message, "reasoning_content", None)),
    }


def main() -> None:
    dev_questions = load_dev_questions(QUESTIONS_PATH)
    print(f"Loaded {len(dev_questions)} dev questions "
          f"({sum(1 for q in dev_questions if q['expected_source'] != 'none')} with a real expected_source, "
          f"{sum(1 for q in dev_questions if q['expected_source'] == 'none')} adversarial)")

    api_key = load_api_key(ENV_PATH)
    client = OpenAI(
        base_url=NVIDIA_API_BASE,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project-5-rag-retrieval")

    results = []
    truncated_count = 0
    reasoning_present_count = 0

    for i, q in enumerate(dev_questions, start=1):
        print(f"[{i}/{len(dev_questions)}] {q['id']}: {q['question'][:80]}")
        try:
            response = ask_closed_book(client, q["question"])
        except Exception as exc:
            print(f"  ERROR: {exc}")
            response = {"answer": None, "finish_reason": "error", "reasoning_content_present": False}

        if response["finish_reason"] == "length":
            truncated_count += 1
            print("  WARNING: answer truncated at max_tokens")
        if response["reasoning_content_present"]:
            reasoning_present_count += 1
            print("  WARNING: reasoning_content was present -- unexpected for this model")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "expected_source": q["expected_source"],
            "expected_answer": q["expected_answer"],
            "model_answer": response["answer"],
            "finish_reason": response["finish_reason"],
            "reasoning_content_present": response["reasoning_content_present"],
        })

        time.sleep(SLEEP_BETWEEN_CALLS)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} answers to {OUT_PATH}")

    with mlflow.start_run(run_name="closed-book-dev"):
        mlflow.log_param("model", MODEL_ID)
        mlflow.log_param("split", "dev")
        mlflow.log_param("n_questions", len(dev_questions))
        mlflow.log_param("max_tokens", MAX_TOKENS)
        mlflow.log_param("temperature", 0)
        mlflow.log_metric("n_truncated", truncated_count)
        mlflow.log_metric("n_reasoning_content_present", reasoning_present_count)
        mlflow.log_metric("n_errors", sum(1 for r in results if r["model_answer"] is None))
        mlflow.log_artifact(str(OUT_PATH))

    print(f"\n{truncated_count} answer(s) truncated at max_tokens={MAX_TOKENS}.")
    print(f"{reasoning_present_count} answer(s) had unexpected reasoning_content.")
    print("Correctness is not scored automatically -- read results/closed_book_dev.json "
          "and compare each model_answer against its expected_answer by hand.")


if __name__ == "__main__":
    main()
