# scripts/run_pipeline.py

from src.build_registry import main as build_registry
from src.dedup_registry import main as dedup
from src.chunk_docs import main as chunk
from src.embed_to_chroma import main as embed

def run():
    print("Step 1: Build registry")
    build_registry()

    print("Step 2: Deduplicate")
    dedup()

    print("Step 3: Chunk docs")
    chunk()

    print("Step 4: Embed to Chroma")
    embed()

if __name__ == "__main__":
    run()