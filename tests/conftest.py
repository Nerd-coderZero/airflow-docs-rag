import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "src"))


def _stub_if_missing(name: str, attrs: dict) -> None:
    if importlib.util.find_spec(name) is None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("stub: package not installed in this environment")


_stub_if_missing("mlflow", {})
_stub_if_missing("rank_bm25", {"BM25Okapi": _Unavailable})
_stub_if_missing("sentence_transformers", {"SentenceTransformer": _Unavailable})
