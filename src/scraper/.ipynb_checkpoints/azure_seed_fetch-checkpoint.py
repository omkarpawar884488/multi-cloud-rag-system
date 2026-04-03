from pathlib import Path
from src.config.load_sources import load_sources
from src.scraper.http_client import fetch_html
import hashlib
import json
import argparse
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


RAW_HTML_DIR = Path("data/raw/html/azure")

EXCLUDE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip")

def should_keep_url(url: str, allowed_domains: set[str] | None) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False

    if p.scheme not in ("http", "https"):
        return False

    if any((p.path or "").lower().endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return False

    if allowed_domains:
        host = (p.hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in allowed_domains)

    return True

def extract_links_from_html(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        abs_url = urljoin(base_url, href)
        if abs_url.startswith("http://") or abs_url.startswith("https://"):
            links.add(abs_url)

    return links

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--allowed-domains", type=str, default="")
    ap.add_argument("--same-domain-only", action="store_true")
    ap.add_argument("--sleep-s", type=float, default=0.25)
    return ap.parse_args()




def main():
    args = parse_args()
    domains = {d.strip().lower() for d in args.allowed_domains.split(",") if d.strip()} or None

    cfg = load_sources()
    azure = cfg["providers"]["azure"]
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    visited = set()
    written = 0

    # queue items: (url, depth, category, referrer_url, domain_lock)
    queue = []

    for category, urls in azure["web_roots"].items():
        for url in urls:
            if not should_keep_url(url, domains):
                continue
            domain_lock = urlparse(url).hostname if args.same_domain_only else None
            queue.append((url, 1, category, None, domain_lock))

    while queue and written < args.max_pages:
        url, depth, category, referrer_url, domain_lock = queue.pop(0)

        if url in visited:
            continue
        if depth > args.max_depth:
            continue

        # optional domain lock: keep within the same domain as the seed
        if args.same_domain_only and domain_lock:
            host = (urlparse(url).hostname or "")
            if host != domain_lock and not host.endswith("." + domain_lock):
                continue

        visited.add(url)

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
            "depth": depth,
            "referrer_url": referrer_url,
        }
        meta_path = RAW_HTML_DIR / f"{category}_{h}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        written += 1

        # enqueue next depth
        if depth < args.max_depth:
            for nxt in extract_links_from_html(html, url):
                if nxt in visited:
                    continue
                if not should_keep_url(nxt, domains):
                    continue
                queue.append((nxt, depth + 1, category, url, domain_lock))

        time.sleep(args.sleep_s)

    print(f"Saved {written} pages to: {RAW_HTML_DIR}")


if __name__ == "__main__":
    main()
