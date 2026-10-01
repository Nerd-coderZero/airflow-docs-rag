"""
Minimal client for Ollama's native HTTP API. The native /api/chat endpoint
is used for runs that need an explicit context window, because num_ctx is
passed through its "options" object; the OpenAI-compatible endpoint used by
closed_book.py has no field for it.
"""

import json
import time
import urllib.request

DEFAULT_HOST = "http://localhost:11434"


def get_json(host: str, path: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(host + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(host: str, path: str, payload: dict, timeout: float) -> dict:
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
        version = get_json(host, "/api/version")["version"]
    except OSError as exc:
        raise SystemExit(f"Ollama is not reachable at {host} ({exc}). Start the Ollama app.")
    tags = get_json(host, "/api/tags").get("models", [])
    match = next((m for m in tags if m.get("name") == model or m.get("model") == model), None)
    if match is None:
        raise SystemExit(f"Model {model} is not pulled. Run: ollama pull {model}")
    details = match.get("details", {})
    return {
        "ollama_version": version,
        "model": model,
        "model_digest": match.get("digest"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
    }


def chat(host: str, model: str, messages: list[dict], options: dict, timeout: float,
         keep_alive: str = "60m") -> dict:
    t0 = time.perf_counter()
    resp = post_json(host, "/api/chat", {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "keep_alive": keep_alive,
    }, timeout)
    elapsed = time.perf_counter() - t0
    return {
        "content": (resp.get("message", {}).get("content") or "").strip(),
        "done_reason": resp.get("done_reason"),
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "prompt_eval_seconds": round((resp.get("prompt_eval_duration") or 0) / 1e9, 2),
        "eval_seconds": round((resp.get("eval_duration") or 0) / 1e9, 2),
        "load_seconds": round((resp.get("load_duration") or 0) / 1e9, 2),
        "latency_seconds": round(elapsed, 2),
    }
