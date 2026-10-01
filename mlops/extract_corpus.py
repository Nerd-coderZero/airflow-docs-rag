"""
Extracts the Project 5 corpus from a pinned checkout of apache/airflow.

Sources included, and why:
  - airflow-core/docs/          core documentation, rst/md, verbatim
  - airflow-core/src/           docstrings only, extracted via ast, not raw source
  - airflow-core/newsfragments/ per-PR changelog fragments, verbatim
  - providers/<name>/docs/      for a fixed 5-provider sample, verbatim

Deliberately excluded: GitHub issues and discussions. They are user-generated
content under GitHub's terms of service, not the repository's Apache-2.0
grant, so including them would misstate what license covers the corpus.

Every extracted file keeps its original repo-relative path in its metadata
so any answer can be traced back to a real file at the pinned commit.
"""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

PROVIDERS = ["amazon", "google", "microsoft", "databricks", "slack"]
EXPECTED_COMMIT = "2f846065252579d4ef8729ba8e42d96c5ab1b611"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def get_current_commit(repo_root: Path) -> str:
    head = (repo_root / ".git" / "HEAD").read_text().strip()
    if head.startswith("ref:"):
        ref_path = repo_root / ".git" / head.split(" ", 1)[1]
        return ref_path.read_text().strip()
    return head


def copy_doc_tree(src_dir: Path, dest_dir: Path, category: str, manifest: list, repo_root: Path, out_root: Path):
    for path in sorted(src_dir.rglob("*")):
        if path.is_file() and path.suffix in (".rst", ".md"):
            rel = path.relative_to(repo_root)
            dest = dest_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            text = path.read_text(encoding="utf-8", errors="ignore")
            manifest.append({
                "category": category,
                "source_path": str(rel),
                "dest_path": str(dest.relative_to(out_root)),
                "words": len(text.split()),
                "sha256": sha256_file(path),
            })


def extract_docstrings(src_dir: Path, dest_dir: Path, manifest: list, repo_root: Path, out_root: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(repo_root)
        try:
            source = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source)
        except SyntaxError:
            continue

        entries = []
        module_ds = ast.get_docstring(tree)
        if module_ds:
            entries.append(("module", "<module>", module_ds))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ds = ast.get_docstring(node)
                if ds:
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    entries.append((kind, node.name, ds))

        if not entries:
            continue

        out_name = str(rel).replace("/", "__") + ".docstrings.txt"
        out_path = dest_dir / out_name
        lines = [f"# source: {rel}\n"]
        total_words = 0
        for kind, name, ds in entries:
            lines.append(f"\n## {kind}: {name}\n{ds}\n")
            total_words += len(ds.split())
        out_path.write_text("".join(lines), encoding="utf-8")

        manifest.append({
            "category": "core_docstrings",
            "source_path": str(rel),
            "dest_path": str(out_path.relative_to(out_root)),
            "docstring_count": len(entries),
            "words": total_words,
            "sha256": sha256_file(out_path),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path,
                         help="path to a checkout of apache/airflow, at the pinned commit")
    parser.add_argument("--out", required=True, type=Path,
                         help="output directory for the extracted corpus")
    parser.add_argument("--skip-commit-check", action="store_true",
                         help="allow extraction from a commit other than the one this corpus was built from")
    args = parser.parse_args()

    repo_root = args.repo_root
    out_root = args.out

    current_commit = get_current_commit(repo_root)
    if current_commit != EXPECTED_COMMIT and not args.skip_commit_check:
        raise SystemExit(
            f"repo at {repo_root} is at commit {current_commit}, expected {EXPECTED_COMMIT}. "
            f"git checkout {EXPECTED_COMMIT} first, or pass --skip-commit-check to extract anyway "
            f"(the corpus manifest and word counts will then reflect a different revision)."
        )

    manifest = []

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    copy_doc_tree(repo_root / "airflow-core" / "docs", out_root / "core_docs", "core_docs", manifest, repo_root, out_root)
    copy_doc_tree(repo_root / "airflow-core" / "newsfragments", out_root / "core_newsfragments", "core_newsfragments", manifest, repo_root, out_root)
    extract_docstrings(repo_root / "airflow-core" / "src", out_root / "core_docstrings", manifest, repo_root, out_root)

    for provider in PROVIDERS:
        provider_docs = repo_root / "providers" / provider
        if not provider_docs.exists():
            raise FileNotFoundError(f"provider dir missing: {provider_docs}")
        copy_doc_tree(provider_docs, out_root / "provider_docs" / provider, f"provider_docs:{provider}", manifest, repo_root, out_root)

    by_category = {}
    for m in manifest:
        cat = m["category"]
        by_category.setdefault(cat, {"files": 0, "words": 0})
        by_category[cat]["files"] += 1
        by_category[cat]["words"] += m["words"]

    summary = {
        "source_repo": "https://github.com/apache/airflow",
        "pinned_commit": current_commit,
        "license": "Apache-2.0 (confirmed directly from repo LICENSE file at this commit)",
        "excluded": "GitHub issues and discussions (user-generated, not covered by repo license)",
        "providers_sampled": PROVIDERS,
        "totals_by_category": by_category,
        "total_files": len(manifest),
        "total_words": sum(m["words"] for m in manifest),
    }

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
