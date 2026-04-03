# src/rag/build_registry.py
from __future__ import annotations
from pathlib import Path
import json
import hashlib

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
OUT_PATH = BASE_DIR / "data" / "processed" / "registry" / "docs_registry.jsonl"

INPUT_DIRS = [
    BASE_DIR / "data" / "processed" / "docs" / "github_normalized",
    BASE_DIR / "data" / "processed" / "docs" / "azure_web",
    BASE_DIR / "data" / "processed" / "docs" / "aws_web",
    BASE_DIR / "data" / "processed" / "docs" / "gcp_web",
]

def load_json_files(folder: Path):
    if not folder.exists():
        return
    for fp in folder.glob("*.json"):
        try:
            yield json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue

def clean_text(t: str) -> str:
    return " ".join((t or "").split())

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for folder in INPUT_DIRS:
            for doc in load_json_files(folder):
                provider = doc.get("provider", "unknown")
                source_type = doc.get("source_type", "unknown")
                title = clean_text(doc.get("title", ""))

                # unify content field
                text = doc.get("body_markdown") or doc.get("body_text") or doc.get("text") or ""
                text = clean_text(text)

                # skip very short
                if len(text) < 300:
                    continue

                unified = {
                    "doc_id": doc.get("doc_id") or doc.get("url") or doc.get("metadata", {}).get("repo_rel_path", ""),
                    "provider": provider,
                    "source_type": source_type,  # github/web
                    "category": doc.get("category", "unknown"),
                    "title": title,
                    "text": text,
                    "url": doc.get("url"),
                    "metadata": doc.get("metadata", {}),
                    "text_hash": sha(text),
                }
                out.write(json.dumps(unified, ensure_ascii=False) + "\n")
                count += 1

    print(f"Wrote registry: {OUT_PATH} ({count} docs)")

if __name__ == "__main__":
    main()
