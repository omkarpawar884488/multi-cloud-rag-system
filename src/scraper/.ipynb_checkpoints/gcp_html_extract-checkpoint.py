from pathlib import Path
from bs4 import BeautifulSoup
import json

#RAW_HTML_DIR = Path("data/raw/html/gcp")
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_HTML_DIR = BASE_DIR / "data/raw/html/gcp"

#OUTPUT_DIR = Path("data/processed/docs/gcp_web")
OUTPUT_DIR = BASE_DIR / "data/processed/docs/gcp_web"

def extract_content(html: str):
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")
    if not main:
        main = soup  # fallback

    # Remove navigation and junk if present
    for selector in ["nav", "header", "footer", "aside", ".sidebar", ".breadcrumbs"]:
        for el in main.select(selector):
            el.decompose()

    title_el = main.find("h1")
    title = title_el.get_text(strip=True) if title_el else ""

    blocks = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "table"]):
        text = el.get_text(" ", strip=True)
        if text:
            blocks.append(text)

    body_text = "\n\n".join(blocks)
    return title, body_text

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for meta_path in RAW_HTML_DIR.glob("*.json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        html_path = Path(meta["file"])
        if not html_path.exists():
            continue

        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        title, body = extract_content(html)
        doc = {
            "provider": "gcp",
            "source_type": "web",
            "url": meta["url"],
            "category": meta["category"],
            "title": title,
            "body_text": body,
        }

        out_name = html_path.stem + ".json"
        out_path = OUTPUT_DIR / out_name
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(doc, out_f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
