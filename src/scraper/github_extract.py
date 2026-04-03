import json
from pathlib import Path

# You can change these in a notebook cell if needed
INDEX_PATH = Path("data/processed/docs/github_index.jsonl")
OUTPUT_DIR = Path("data/processed/docs/github_normalized")


def strip_front_matter(text: str) -> str:
    """
    Remove leading YAML front matter if present.

    YAML front matter typically looks like:

    ---
    key: value
    ---
    """
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2]
    return text


def extract_title_and_body(md: str):
    """
    Extract the first '# ' heading as title (if any),
    and return the remaining text as body.
    """
    lines = md.splitlines()
    title = ""
    body_lines = []

    for line in lines:
        if not title and line.startswith("# "):
            title = line[2:].strip()
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()
    return title, body


def normalize_markdown_docs(
    index_path: Path = INDEX_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> int:
    """
    Read the JSONL index, load each markdown file, normalize it
    into a JSON structure (title + body + metadata), and save
    one JSON file per document.

    Returns the number of documents written.
    """
    index_path = Path(index_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(index_path, "r", encoding="utf-8") as index_file:
        for line in index_file:
            entry = json.loads(line)

            # Path to the original markdown file
            path = Path(entry["abs_path"])
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                md = f.read()

            # 1) Strip YAML front matter
            md = strip_front_matter(md)

            # 2) Split into title + body
            title, body = extract_title_and_body(md)

            # 3) Build normalized document
            doc = {
                "doc_id": entry["doc_id"],
                "provider": entry["provider"],        # aws / gcp / azure / unknown
                "source_type": entry["source_type"],  # "github"
                "title": title,
                "body_markdown": body,
                "metadata": {
                    "repo_rel_path": entry["repo_rel_path"],
                    "content_hash": entry["content_hash"],
                },
            }

            # 4) Save as JSON (one file per original markdown)
            out_path = output_dir / (entry["doc_id"].replace("/", "_") + ".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as out_f:
                json.dump(doc, out_f, ensure_ascii=False, indent=2)

            count += 1

    print(f"Wrote {count} normalized documents to {output_dir}")
    return count


if __name__ == "__main__":
    normalize_markdown_docs()




