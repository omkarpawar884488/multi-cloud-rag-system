from pathlib import Path
from bs4 import BeautifulSoup
import json

# Robust base dir (works no matter where you run the script from)
BASE_DIR = Path(__file__).resolve().parents[2]

RAW_HTML_DIR = BASE_DIR / "data" / "raw" / "html" / "aws"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "docs" / "aws_web"

def extract_content(html: str):
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")
    if not main:
        main = soup  # fallback

    # Remove navigation and junk if present
    for selector in ["nav", "header", "footer", "aside", ".sidebar", ".breadcrumbs", ".toc"]:
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
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # ✅ NEW: crawler stores HTML with same stem as meta, but .html suffix
        html_path = meta_path.with_suffix(".html")
        if not html_path.exists():
            print(f"Missing HTML for meta: {meta_path.name}")
            continue

        html = html_path.read_text(encoding="utf-8", errors="ignore")
        title, body = extract_content(html)

        # optional: skip empty/very short extracts
        if not body or len(body) < 300:
            continue

        doc = {
            "provider": "aws",
            "source_type": "web",
            "url": meta.get("url"),
            "category": meta.get("category", "unknown"),  # crawler doesn't set this
            "title": title,
            "body_text": body,
            "metadata": {
                "depth": meta.get("depth"),
                "domain_type": meta.get("domain_type"),
            },
        }

        # ✅ Use meta_path stem so output matches raw files
        out_path = OUTPUT_DIR / f"{meta_path.stem}.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"AWS extraction complete. Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()



'''
## old code
from pathlib import Path
from bs4 import BeautifulSoup
import json

RAW_HTML_DIR = Path("data/raw/html/aws")
OUTPUT_DIR = Path("data/processed/docs/aws_web")

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

        #html_path = Path(meta["file"])
        html_path = meta_path.with_suffix(".html")
        if not html_path.exists():
            continue

        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        title, body = extract_content(html)
        doc = {
            "provider": "aws",
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
'''