"""
Ping candidate models in parallel and report which respond right now,
and whether the reply looks clean (no reasoning trace).

Usage (from scripts/src):
    python probe_models.py ../../.env
"""
import concurrent.futures as cf
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(sys.argv[1] if len(sys.argv) > 1 else "../../.env")

CANDIDATES = [
    "google/gemma-4-31b-it",
    "google/diffusiongemma-26b-a4b-it",   # hybrid: needs enable_thinking False
    "nvidia/nemotron-3.5-lightning-30b-a3b",  # hybrid: try enable_thinking False
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/muse-glimmer-30b",
    "moonshotai/kimi-k3",
    "z-ai/glm-5.3-flash",
    "poolside/laguna-xs-2.1",
]
HYBRIDS = {"google/diffusiongemma-26b-a4b-it", "nvidia/nemotron-3.5-lightning-30b-a3b"}

client = OpenAI(base_url=os.environ["NVIDIA_BASE_URL"],
                api_key=os.environ["NVIDIA_API_KEY"], timeout=45, max_retries=0)


def probe(model):
    t = time.time()
    kwargs = {}
    if model in HYBRIDS:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Which Airflow operator runs a Bash command?"}],
            temperature=0, max_tokens=200, **kwargs)
        msg = r.choices[0].message
        extra = msg.model_extra or {}
        content = (msg.content or "").strip()
        reasoning = (getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                     or extra.get("reasoning") or extra.get("reasoning_content"))
        if reasoning or "<think>" in content:
            verdict = "REASONING"
        elif not content:
            verdict = "EMPTY"
        else:
            verdict = "CLEAN"
        return model, f"{verdict} finish={r.choices[0].finish_reason} ({time.time()-t:.1f}s) {content[:50]!r}"
    except Exception as e:
        return model, f"ERROR {type(e).__name__}: {str(e)[:60]} ({time.time()-t:.1f}s)"


with cf.ThreadPoolExecutor(4) as ex:
    futs = [ex.submit(probe, m) for m in CANDIDATES]
    for f in cf.as_completed(futs):
        m, out = f.result()
        print(f"{m}: {out}", flush=True)
