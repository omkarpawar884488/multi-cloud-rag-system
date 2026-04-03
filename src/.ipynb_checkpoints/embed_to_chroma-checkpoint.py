"""
Embed chunked documents into Chroma vector database.

- Reads chunks.jsonl
- Uses sentence-transformers/all-MiniLM-L6-v2
- Stores embeddings + minimal metadata in Chroma
- Telemetry disabled
"""

from pathlib import Path
import json
import hashlib
import os

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from .settings import COLLECTION_NAME

# =========================
# Paths & Config
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]
#BASE_DIR = Path(r"E:/GenAi/project/Rag_cloud_platforms")

CHUNKS_PATH = BASE_DIR / "data/processed/chunks/chunks.jsonl"
CHROMA_DIR = BASE_DIR / "data/vector_db/chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

#COLLECTION_NAME = "cloud_rag_chunks"
BATCH_SIZE = 64
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# Helpers
# =========================

def sanitize(value):
    """Chroma metadata must be str, int, float, bool (no None)."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# =========================
# Main
# =========================

def main():
    # Disable telemetry explicitly
    client = chromadb.Client(
        Settings(
            persist_directory=str(CHROMA_DIR),
            anonymized_telemetry=False,
        )
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    model = SentenceTransformer(EMBED_MODEL_NAME)

    ids, docs, metas = [], [], []
    total_chunks = 0

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            ch = json.loads(line)

            ids.append(ch["chunk_id"])
            docs.append(ch["chunk_text"])
            metas.append(
                {
                    "doc_id": sanitize(ch["doc_id"]),
                    "provider": sanitize(ch["provider"]),
                    "category": sanitize(ch["category"]),
                    "source_type": sanitize(ch["source_type"]),
                    "title": sanitize(ch["title"]),
                    "url": sanitize(ch["url"]),
                    "chunk_index": ch["chunk_index"],
                }
            )

            if len(ids) >= BATCH_SIZE:
                embeddings = model.encode(
                    docs,
                    show_progress_bar=False
                ).tolist()

                collection.add(
                    ids=ids,
                    documents=docs,
                    metadatas=metas,
                    embeddings=embeddings,
                )

                total_chunks += len(ids)
                ids, docs, metas = [], [], []

    # Final partial batch
    if ids:
        embeddings = model.encode(
            docs,
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

        total_chunks += len(ids)

    print(f"Embedded & stored {total_chunks} chunks in Chroma")
    print(f"Vector DB location: {CHROMA_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Collection count: {collection.count()}")


if __name__ == "__main__":
    main()
