from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # Rag_cloud_platforms

CHROMA_DIR = BASE_DIR / "data" / "vector_db" / "chroma"
COLLECTION_NAME = "cloud_rag_chunks"

LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
TOP_K = 8
MAX_CHARS = 1800
OVERLAP = 200

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
