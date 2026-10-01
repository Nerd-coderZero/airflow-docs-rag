"""
Checks that a long prompt reaches the model intact at the context window
used for the RAG run. A prompt of real corpus text (about 3,000 words, well
above common default windows and below 8192 tokens) is sent with one
verification code at its start and another at its end. The run passes only
if the reply contains both codes and prompt_eval_count leaves room for the
answer inside num_ctx.

With --also-default, the same prompt is sent once more without num_ctx, to
record what the server's default window does to it.

Usage (from scripts/src, Ollama running):
    python context_check.py --model qwen2.5:7b-instruct-q4_K_M
"""

import argparse
import json
import sys
import time
from pathlib import Path

from ollama_native import DEFAULT_HOST, chat, model_metadata

NUM_CTX = 8192
NUM_PREDICT_CHECK = 60
FILLER_WORDS = 3000
CODE_START = "KESTREL-4471"
CODE_END = "MARIGOLD-9036"
SYSTEM_PROMPT = "You are answering questions about Apache Airflow."
CORPUS_DOCS = Path("../../corpus/core_docs/airflow-core/docs")
OUT_DIR = Path("../../results")
TIMEOUT_SECONDS = 1800


def filler_text(n_words: int) -> str:
    words = []
    for path in sorted(CORPUS_DOCS.rglob("*.rst")):
        words.extend(path.read_text(encoding="utf-8", errors="replace").split())
        if len(words) >= n_words:
            break
    if len(words) < n_words:
        raise SystemExit(f"Only {len(words)} words found under {CORPUS_DOCS}")
    return " ".join(words[:n_words])


def build_messages() -> list[dict]:
    body = (
        f"The first verification code is {CODE_START}.\n\n"
        f"{filler_text(FILLER_WORDS)}\n\n"
        f"The second verification code is {CODE_END}.\n\n"
        "Reply with only the first verification code and the second verification code, "
        "separated by a space."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}]


def run_once(host: str, model: str, messages: list[dict], num_ctx: int | None) -> dict:
    options = {"temperature": 0, "seed": 42, "num_predict": NUM_PREDICT_CHECK}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    label = f"num_ctx={num_ctx}" if num_ctx else "server default num_ctx"
    print(f"Sending check prompt ({label}) ...", flush=True)
    r = chat(host, model, messages, options, TIMEOUT_SECONDS)
    found_start = CODE_START in r["content"]
    found_end = CODE_END in r["content"]
    room_ok = num_ctx is None or (r["prompt_eval_count"] or 0) + NUM_PREDICT_CHECK <= num_ctx
    r.update({"num_ctx": num_ctx, "found_start_code": found_start, "found_end_code": found_end,
              "passed": found_start and found_end and room_ok})
    print(f"  prompt_eval_count={r['prompt_eval_count']}  prompt eval {r['prompt_eval_seconds']}s "
          f"({(r['prompt_eval_count'] or 0) / max(r['prompt_eval_seconds'], 1e-9):.1f} tok/s)")
    print(f"  reply: {r['content'][:120]!r}")
    print(f"  start code found: {found_start}, end code found: {found_end}, passed: {r['passed']}")
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="Context window check against a local Ollama model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--also-default", action="store_true")
    args = parser.parse_args()

    meta = model_metadata(args.host, args.model)
    messages = build_messages()
    results = [run_once(args.host, args.model, messages, NUM_CTX)]
    if args.also_default:
        results.append(run_once(args.host, args.model, messages, None))

    safe = args.model.replace(":", "_").replace("/", "_")
    out = OUT_DIR / f"context_check_{safe}.json"
    OUT_DIR.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        **meta,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "filler_words": FILLER_WORDS,
        "codes": [CODE_START, CODE_END],
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    if not results[0]["passed"]:
        print(f"FAILED at num_ctx={NUM_CTX}. Do not start the RAG run.")
        sys.exit(1)
    print(f"PASSED at num_ctx={NUM_CTX}.")


if __name__ == "__main__":
    main()
