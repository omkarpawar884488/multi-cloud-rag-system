from __future__ import annotations

from functools import lru_cache
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .settings import CHROMA_DIR, COLLECTION_NAME, TOP_K

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _chroma_client():
    # IMPORTANT: this is the client that sees your 18280 vectors
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=1)
def get_vectorstore():
    return Chroma(
        client=_chroma_client(),                 # ✅ critical
        collection_name=COLLECTION_NAME,
        embedding_function=_embeddings(),        # ✅ same model as ingestion
    )

''' 
# old code
def get_retriever(where: dict | None = None, k: int | None = None):
    vs = get_vectorstore()
    k = TOP_K if k is None else k
    return vs.as_retriever(search_kwargs={"k": k, "filter": where or {}})
'''

def get_retriever(where: dict | None = None, k: int | None = None):
    vs = get_vectorstore()
    k = TOP_K if k is None else k

    search_kwargs = {"k": k}
    if where:  # only add filter when it actually filters something
        search_kwargs["filter"] = where

    return vs.as_retriever(search_kwargs=search_kwargs)

