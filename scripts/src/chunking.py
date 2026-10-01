"""
Two chunking strategies compared against each other, not assumed.

structure_aware_chunks: splits RST/docstring files on their own heading
hierarchy (or the "## kind: name" markers in extracted docstring files),
then further splits any section that is still too large by paragraph
boundaries. A section is never split mid-paragraph unless a single
paragraph alone exceeds the cap, in which case it falls back to a word-
count split as a last resort.

fixed_window_chunks: a naive fixed-size sliding window over the raw text,
ignoring all document structure. This is the default most RAG tutorials
ship with, and is the baseline structure-aware chunking is measured
against.

Chunk size is measured in words, not tokens. This is an approximation
(the embedding model's own tokenizer would count differently), but it is
consistent across both strategies, which is what the comparison needs.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    text: str
    word_count: int = field(init=False)

    def __post_init__(self):
        self.word_count = len(self.text.split())


def _chunk_id(source_path: str, index: int, strategy: str) -> str:
    h = hashlib.sha256(f"{strategy}:{source_path}:{index}".encode()).hexdigest()[:12]
    return h


RST_HEADING_CHARS = "=-~^'\"`#*+.:_"


def _split_rst_sections(text: str) -> list[tuple[str, str]]:
    """
    Splits RST text into (heading, body) sections using Sphinx's underline
    convention: a line of repeated punctuation directly below a text line
    marks that text line as a heading. Text before the first heading is
    kept as a section with an empty heading.
    """
    lines = text.split("\n")
    sections = []
    current_heading = ""
    current_body = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        is_heading = False
        if i + 1 < n:
            next_line = lines[i + 1].strip()
            this_stripped = line.strip()
            if (
                this_stripped
                and next_line
                and len(next_line) >= max(3, len(this_stripped) - 2)
                and len(set(next_line)) == 1
                and next_line[0] in RST_HEADING_CHARS
            ):
                is_heading = True

        if is_heading:
            if current_body:
                sections.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.strip()
            current_body = []
            i += 2
            continue

        current_body.append(line)
        i += 1

    if current_body:
        sections.append((current_heading, "\n".join(current_body).strip()))

    return [(h, b) for h, b in sections if b.strip()]


def _split_docstring_file(text: str) -> list[tuple[str, str]]:
    """
    Docstring files (from mlops/extract_corpus.py) use their own marker
    format: '## kind: name' followed by the docstring body. Split on that.
    """
    parts = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
    sections = []
    if parts[0].strip():
        sections.append(("", parts[0].strip()))
    for j in range(1, len(parts) - 1, 2):
        heading = parts[j].strip()
        body = parts[j + 1].strip()
        if body:
            sections.append((heading, body))
    return sections


def _split_paragraphs(text: str) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if p.strip()]


def _word_split(text: str, max_words: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _merge_undersized_sections(sections: list[tuple[str, str]], min_words: int = 15) -> list[tuple[str, str]]:
    """
    Airflow's docs commonly nest a bare subsection heading directly above an
    RST target directive (e.g. "Operators" followed only by
    ".. _howto/operator:X:"), with the real content living under the next
    heading down. Left alone, this produces near-empty chunks that carry no
    answerable content. Any section under min_words is merged forward into
    the next section, carrying its heading along as context rather than
    being dropped or emitted standalone.
    """
    if not sections:
        return sections

    merged = []
    pending_heading = None
    pending_body = ""

    for heading, body in sections:
        combined_heading = f"{pending_heading} / {heading}" if pending_heading and heading else (pending_heading or heading)
        combined_body = f"{pending_body}\n\n{body}".strip() if pending_body else body
        word_count = len(f"{combined_heading} {combined_body}".split())

        if word_count < min_words:
            pending_heading = combined_heading
            pending_body = combined_body
            continue

        merged.append((combined_heading, combined_body))
        pending_heading, pending_body = None, ""

    if pending_heading or pending_body:
        if merged:
            last_heading, last_body = merged[-1]
            new_heading = f"{last_heading} / {pending_heading}" if pending_heading else last_heading
            new_body = f"{last_body}\n\n{pending_body}".strip()
            merged[-1] = (new_heading, new_body)
        else:
            merged.append((pending_heading or "", pending_body))

    return merged


def structure_aware_chunks(source_path: str, text: str, max_words: int = 250, min_words: int = 15) -> list[Chunk]:
    is_docstring_file = source_path.endswith(".docstrings.txt")
    sections = _split_docstring_file(text) if is_docstring_file else _split_rst_sections(text)

    if not sections:
        sections = [("", text)]

    sections = _merge_undersized_sections(sections, min_words=min_words)

    chunks = []
    idx = 0
    for heading, body in sections:
        prefixed = f"{heading}\n\n{body}" if heading else body
        if len(prefixed.split()) <= max_words:
            pieces = [prefixed]
        else:
            paras = _split_paragraphs(body)
            pieces = []
            buf = f"{heading}\n\n" if heading else ""
            buf_words = len((heading or "").split())
            for para in paras:
                para_words = len(para.split())
                if para_words > max_words:
                    if buf.strip():
                        pieces.append(buf.strip())
                        buf, buf_words = "", 0
                    for sub in _word_split(para, max_words):
                        pieces.append(sub)
                    continue
                if buf_words + para_words > max_words and buf.strip():
                    pieces.append(buf.strip())
                    buf, buf_words = "", 0
                buf += para + "\n\n"
                buf_words += para_words
            if buf.strip():
                pieces.append(buf.strip())

        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(Chunk(
                chunk_id=_chunk_id(source_path, idx, "structure"),
                source_path=source_path,
                text=piece.strip(),
            ))
            idx += 1

    return chunks


def fixed_window_chunks(source_path: str, text: str, window_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks = []
    idx = 0
    step = max(1, window_words - overlap_words)
    i = 0
    while i < len(words):
        window = words[i:i + window_words]
        if not window:
            break
        chunks.append(Chunk(
            chunk_id=_chunk_id(source_path, idx, "fixed"),
            source_path=source_path,
            text=" ".join(window),
        ))
        idx += 1
        if i + window_words >= len(words):
            break
        i += step

    return chunks


def load_corpus_files(corpus_root: Path) -> list[tuple[str, str]]:
    """
    Returns (source_path, text) pairs for every corpus file, using the
    manifest so source_path exactly matches expected_source in questions.json.
    """
    manifest_path = corpus_root / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    files = []
    for entry in manifest:
        dest = corpus_root / entry["dest_path"]
        source_path = entry["source_path"]
        text = dest.read_text(encoding="utf-8", errors="ignore")
        files.append((source_path, text))
    return files


def build_chunk_set(corpus_root: Path, strategy: str) -> list[Chunk]:
    files = load_corpus_files(corpus_root)
    all_chunks = []
    for source_path, text in files:
        if strategy == "structure":
            all_chunks.extend(structure_aware_chunks(source_path, text))
        elif strategy == "fixed":
            all_chunks.extend(fixed_window_chunks(source_path, text))
        else:
            raise ValueError(f"unknown strategy: {strategy}")
    return all_chunks


if __name__ == "__main__":
    import sys
    corpus_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus_rebuild")

    for strategy in ("structure", "fixed"):
        chunks = build_chunk_set(corpus_root, strategy)
        sizes = [c.word_count for c in chunks]
        print(f"{strategy}: {len(chunks)} chunks, "
              f"avg {sum(sizes)/len(sizes):.1f} words, "
              f"min {min(sizes)}, max {max(sizes)}")
