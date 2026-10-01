"""
Diagnose the NVIDIA API: is it the key, the network, the model endpoint,
or the request size/settings in closed_book.py?

Usage (from scripts/src):
    python diagnose_api.py ../../.env
Needs: pip install openai python-dotenv requests
Every call has a short timeout, so Ctrl+C / the script never hangs long.
"""
import os
import sys
import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(sys.argv[1] if len(sys.argv) > 1 else "../../.env")

KEY = os.environ.get("NVIDIA_API_KEY", "")
BASE = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.environ.get("MODEL_ID", "google/gemma-4-31b-it")
Q = "Which Airflow operator runs a Bash command?"

client = OpenAI(base_url=BASE, api_key=KEY, timeout=40, max_retries=0)
results = {}


def step(name, fn):
    t = time.time()
    try:
        out = fn()
        results[name] = True
        print(f"[PASS] {name} ({time.time() - t:.1f}s) {out}", flush=True)
    except Exception as e:
        results[name] = False
        print(f"[FAIL] {name} ({time.time() - t:.1f}s) {type(e).__name__}: {str(e)[:120]}", flush=True)


def check_key():
    assert KEY.startswith("nvapi-"), "key missing or doesn't start with nvapi-"
    assert "PASTE" not in KEY, "still the placeholder key"
    return f"len={len(KEY)}"


def check_network_auth():
    r = requests.get(f"{BASE}/models", headers={"Authorization": f"Bearer {KEY}"}, timeout=20)
    r.raise_for_status()  # 401/403 = key problem, timeout = network problem
    return f"HTTP {r.status_code}, {len(r.json().get('data', []))} models listed"


def chat(max_tokens, prompt=Q, stream=False):
    def run():
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=max_tokens, stream=stream)
        if stream:
            text = "".join((c.choices[0].delta.content or "") for c in r if c.choices)
            return repr(text[:50])
        return repr((r.choices[0].message.content or "")[:50])
    return run


print(f"Model: {MODEL}\nBase:  {BASE}\n")
step("1 key looks valid", check_key)
step("2 network + auth (list models)", check_network_auth)
step("3 tiny chat, max_tokens=20", chat(20))
step("4 tiny chat again (consistency)", chat(20))
step("5 medium chat, max_tokens=256", chat(256))
step(f"6 script-size chat, max_tokens={os.environ.get('MAX_TOKENS', '1024')}",
     chat(int(os.environ.get("MAX_TOKENS", "1024"))))
step("7 streaming, max_tokens=1024", chat(1024, stream=True))

print("\n--- Verdict ---")
if not results.get("1 key looks valid"):
    print("Fix the key in .env first.")
elif not results.get("2 network + auth (list models)"):
    print("Key/network problem (401/403 = bad key; timeout = firewall/VPN/internet).")
elif not results.get("3 tiny chat, max_tokens=20"):
    print("Auth and network are fine but the model endpoint itself is down/overloaded.")
    print("Not your script. Wait, or pick another model.")
elif not results.get("5 medium chat, max_tokens=256") or not results.get(
        [k for k in results if k.startswith("6")][0]):
    print("Tiny calls work but longer generations time out at the gateway.")
    if results.get("7 streaming, max_tokens=1024"):
        print("Streaming works: switch closed_book.py to stream=True.")
    else:
        print("Lower MAX_TOKENS (e.g. 400) and retry.")
else:
    print("API looks healthy. If closed_book.py still fails, the problem is in the script")
    print("(prompt size, model ID, or how the .env is loaded).")
