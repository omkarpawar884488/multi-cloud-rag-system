from pathlib import Path
from src.config.load_sources import load_sources
from src.scraper.http_client import fetch_html
import hashlib
import json

RAW_HTML_DIR = Path("data/raw/html/azure")

def main():
    cfg = load_sources()
    azure = cfg["providers"]["azure"]
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    for category, urls in azure["web_roots"].items():
        for url in urls:
            html = fetch_html(url)
            if not html:
                print(f"Failed: {url}")
                continue

            h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            fname = RAW_HTML_DIR / f"{category}_{h}.html"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(html)

            meta = {
                "url": url,
                "category": category,
                "provider": "azure",
                "file": str(fname),
            }
            meta_path = RAW_HTML_DIR / f"{category}_{h}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
