"""
Query ChromaDB using SentenceTransformer embeddings
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# Disable Chroma telemetry BEFORE importing chromadb
# ------------------------------------------------------------------
os.environ["CHROMA_ANONYMIZED_TELEMETRY"] = "FALSE"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------------
BASE_DIR = Path(r"E:/GenAi/project/Rag_cloud_platforms")
CHROMA_DIR = BASE_DIR / "data/vector_db/chroma"

COLLECTION_NAME = "cloud_rag_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------
# Chroma client & collection
# ------------------------------------------------------------------
def get_chroma_collection():
    """Initialize Chroma client and return collection"""
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    collection = client.get_collection(COLLECTION_NAME)
    return collection


# ------------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------------
def load_embedding_model():
    """Load sentence transformer model"""
    return SentenceTransformer(MODEL_NAME)


# ------------------------------------------------------------------
# Query execution
# ------------------------------------------------------------------
def query_collection(collection, model, query_text, n_results=5):
    """Embed query and search Chroma"""
    query_embedding = model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"provider": "aws"}  # optional filter
    )

    return results


# ------------------------------------------------------------------
# Result printing
# ------------------------------------------------------------------
def print_results(query_text, results, max_chars=600):
    print("\nQUERY:", query_text)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        print(f"\n--- {i} ---")
        print("Title:", meta.get("title", "N/A"))
        print("URL:", meta.get("url", "N/A"))
        print(doc[:max_chars], "...")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    collection = get_chroma_collection()
    print("Collection count:", collection.count())

    model = load_embedding_model()

    query = "Explain AWS Well-Architected cost optimization pillar in simple terms"
    results = query_collection(collection, model, query)

    print_results(query, results)


# ------------------------------------------------------------------
# Script execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
