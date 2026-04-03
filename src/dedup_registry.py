# src/rag/dedup_registry.py
from __future__ import annotations
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[2]
IN_PATH = BASE_DIR / "data" / "processed" / "registry" / "docs_registry.jsonl"
OUT_PATH = BASE_DIR / "data" / "processed" / "registry" / "docs_registry_dedup.jsonl"

def main():
    print("Starting deduplication...")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0
    total = 0

    with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            total += 1
            doc = json.loads(line)
            h = doc.get("text_hash")
            if not h or h in seen:
                continue
            seen.add(h)
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Dedup done: {kept}/{total} docs kept → {OUT_PATH}")

if __name__ == "__main__":
    main()
