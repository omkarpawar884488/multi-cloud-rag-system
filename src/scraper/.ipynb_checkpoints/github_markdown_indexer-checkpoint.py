"""
rag_github_markdown_indexer.py

Scan cloned GitHub repos (AWS / Azure / GCP, etc.), find all Markdown files,
and write a JSONL index with basic metadata for each document.
"""

import os
import json
import hashlib
import argparse
from pathlib import Path


def iter_markdown_files(root: Path):
    """
    Yield all markdown files under `root`.
    """
    root = Path(root)
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith((".md", ".markdown")):
                yield Path(dirpath) / fname


def detect_provider(path: Path) -> str:
    """
    Infer cloud provider (azure, gcp, aws) from file path.
    Falls back to 'unknown' if nothing matches.
    """
    p = str(path).lower()

    if "azure" in p:
        return "azure"
    if "gcp" in p or "googlecloudplatform" in p:
        return "gcp"
    if "aws" in p or "amazon" in p:
        return "aws"
    return "unknown"


def make_doc_id(path: Path, root: Path) -> str:
    """
    Make a stable doc id from path relative to `root`.
    """
    rel = Path(path).relative_to(root)
    return rel.as_posix()  # POSIX-style (forward slashes)


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """
    SHA-256 hash of file contents.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_index(root: Path, output_path: Path, overwrite: bool = True) -> None:
    """
    Build a JSONL index of all markdown files under `root`.
    """
    root = Path(root)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            f"Use --overwrite flag if you want to overwrite it."
        )

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for md_path in iter_markdown_files(root):
            doc = {
                "doc_id": make_doc_id(md_path, root),
                "provider": detect_provider(md_path),
                "source_type": "github",
                "repo_rel_path": str(md_path.relative_to(root)),
                "abs_path": str(md_path.resolve()),
                "content_hash": file_hash(md_path),
            }
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} documents to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index markdown files from cloned GitHub repos into a JSONL file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/github"),
        help="Root directory containing the cloned GitHub repositories "
             "(default: data/raw/github)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/docs/github_index.jsonl"),
        help="Path to the output JSONL file "
             "(default: data/processed/docs/github_index.jsonl)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    overwrite = not args.no_overwrite
    build_index(root=args.root, output_path=args.output, overwrite=overwrite)


if __name__ == "__main__":
    main()
