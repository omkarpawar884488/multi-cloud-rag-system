"""
Chunk documents for RAG ingestion.

- Reads deduplicated documents (docs_registry_dedup.jsonl)
- Applies structure-aware chunking:
  * GitHub / Markdown → heading-aware chunking
  * Web docs → paragraph-aware chunking
- Writes chunk-level records with metadata and chunk_hash
"""

from pathlib import Path
import json
import re
import hashlib
from .settings import MAX_CHARS, OVERLAP

# =========================
# Paths & Constants
# =========================


BASE_DIR = Path(__file__).resolve().parents[2]
#BASE_DIR = Path(r"E:/GenAi/project/Rag_cloud_platforms")

DEDUP_PATH = BASE_DIR / "data/processed/registry/docs_registry_dedup.jsonl"
CHUNKS_PATH = BASE_DIR / "data/processed/chunks/chunks.jsonl"
CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

#MAX_CHARS = 1800
#OVERLAP = 200


# =========================
# Utilities
# =========================

def sha(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =========================
# Chunking Logic
# =========================

def chunk_markdown(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    """
    Chunk markdown text by headings (##, ###).
    Falls back to sliding window with overlap if a section is too large.
    """
    sections = re.split(r"\n(?=##\s|###\s)", text)
    chunks = []

    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        if len(sec) <= max_chars:
            chunks.append(sec)
        else:
            # Forced split → sliding window with overlap
            start = 0
            while start < len(sec):
                end = min(start + max_chars, len(sec))
                chunks.append(sec[start:end].strip())
                if end == len(sec):
                    break
                start = max(0, end - overlap)

    return chunks


def chunk_web(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    """
    Chunk web text by paragraphs.
    Uses overlap only when a single paragraph exceeds max_chars.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for p in paragraphs:
        if len(p) > max_chars:
            # Forced split → sliding window with overlap
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                chunks.append(p[start:end].strip())
                if end == len(p):
                    break
                start = max(0, end - overlap)
            current = ""
        elif len(current) + len(p) <= max_chars:
            current += ("\n\n" + p if current else p)
        else:
            chunks.append(current)
            current = p

    if current:
        chunks.append(current)

    return chunks


def split_into_chunks(doc: dict):
    """
    Route chunking logic based on source_type.
    """
    text = doc["text"]
    source_type = doc.get("source_type", "").lower()

    if "github" in source_type or "markdown" in source_type:
        return chunk_markdown(text)
    else:
        return chunk_web(text)


# =========================
# Main
# =========================

def main():
    n_docs = 0
    n_chunks = 0

    with DEDUP_PATH.open("r", encoding="utf-8") as fin, \
         CHUNKS_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            doc = json.loads(line)
            n_docs += 1

            chunks = split_into_chunks(doc)

            for i, c in enumerate(chunks):
                chunk_text = c.strip()
                chunk_hash = sha(chunk_text)

                fout.write(
                    json.dumps(
                        {
                            "chunk_id": f"{doc['doc_id']}::c{i}",
                            "chunk_hash": chunk_hash,
                            "doc_id": doc["doc_id"],
                            "provider": doc["provider"],
                            "category": doc["category"],
                            "source_type": doc["source_type"],
                            "title": doc.get("title", ""),
                            "url": doc.get("url"),
                            "metadata": doc.get("metadata", {}),
                            "chunk_index": i,
                            "chunk_text": chunk_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                n_chunks += 1

    print(f"Chunked docs: {n_docs}, chunks: {n_chunks} → {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
