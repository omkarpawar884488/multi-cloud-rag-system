# src/scraper/aws_crawl.py
from __future__ import annotations
from pathlib import Path
from collections import deque, defaultdict
from urllib.parse import urljoin, urlparse, urlunparse
import json
import re
import hashlib

from bs4 import BeautifulSoup

from src.config.load_sources import load_sources
from src.scraper.http_client import fetch_html

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "html" / "aws"

# ---- AWS-specific allowlists (CRITICAL) ----
DOCS_DOMAIN = "docs.aws.amazon.com"
MARKETING_DOMAIN = "aws.amazon.com"

DOCS_ALLOWED_PREFIXES = [
    "/wellarchitected/latest/",
    "/prescriptive-guidance/latest/",
    "/iam/latest/userguide/",
    "/vpc/latest/userguide/",
    "/kms/latest/developerguide/",
    "/cost-management/",
    "/organizations/latest/userguide/",
    "/controltower/latest/userguide/",
    "/ec2/latest/userguide/",
    "/s3/latest/userguide/",
    "/lambda/latest/dg/",
    "/eks/latest/userguide/",
]

MARKETING_ALLOWED_PREFIXES = [
    "/architecture/",
    "/solutions/",
    "/compliance/",
]

SKIP_EXTENSIONS = (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".mp4", ".xml")
SKIP_SUBSTRINGS = [
    "APIReference", "/api/", "/cli/", "/sdk/", "changelog", "release-notes", "sitemap", "rss", "atom"
]

def canonicalize(url: str) -> str:
    """Normalize URL to avoid duplicates."""
    u = url.strip()
    u = u.split("#")[0]  # drop fragments
    parsed = urlparse(u)

    # drop most query params to reduce duplicates
    # (AWS docs mostly stable without query)
    parsed = parsed._replace(query="")

    # normalize trailing slash
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    parsed = parsed._replace(path=path)

    return urlunparse(parsed)

def is_allowed(url: str) -> bool:
    u = urlparse(url)
    if u.scheme not in ("http", "https"):
        return False

    if any(u.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False

    if any(s.lower() in url.lower() for s in SKIP_SUBSTRINGS):
        return False

    host = u.netloc.lower()

    if host == DOCS_DOMAIN:
        return any(u.path.startswith(p) for p in DOCS_ALLOWED_PREFIXES)

    if host == MARKETING_DOMAIN:
        return any(u.path.startswith(p) for p in MARKETING_ALLOWED_PREFIXES)

    return False

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        full = urljoin(base_url, href)
        links.append(full)
    return links

def url_to_filename(url: str) -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return h

def crawl(start_urls: list[str], max_depth_docs: int = 3, max_depth_marketing: int = 2,
          max_pages_docs: int = 6000, max_pages_marketing: int = 800):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    queue = deque()
    visited = set()

    # Track pages per domain
    counts = defaultdict(int)

    for u in start_urls:
        cu = canonicalize(u)
        if is_allowed(cu):
            queue.append((cu, 0))

    while queue:
        url, depth = queue.popleft()
        if url in visited:
            continue

        host = urlparse(url).netloc.lower()
        if host == DOCS_DOMAIN and counts["docs"] >= max_pages_docs:
            continue
        if host == MARKETING_DOMAIN and counts["marketing"] >= max_pages_marketing:
            continue

        # depth control per domain
        if host == DOCS_DOMAIN and depth > max_depth_docs:
            continue
        if host == MARKETING_DOMAIN and depth > max_depth_marketing:
            continue

        visited.add(url)

        html = fetch_html(url)
        if not html:
            continue

        # save html + metadata
        fname = url_to_filename(url)
        html_path = RAW_DIR / f"{fname}.html"
        meta_path = RAW_DIR / f"{fname}.json"

        html_path.write_text(html, encoding="utf-8", errors="ignore")
        meta_path.write_text(json.dumps({
            "url": url,
            "depth": depth,
            "domain_type": "docs" if host == DOCS_DOMAIN else "marketing",
        }, indent=2), encoding="utf-8")

        if host == DOCS_DOMAIN:
            counts["docs"] += 1
        else:
            counts["marketing"] += 1

        # enqueue next links
        for link in extract_links(html, url):
            cl = canonicalize(link)
            if cl not in visited and is_allowed(cl):
                queue.append((cl, depth + 1))

    print("AWS crawl done.")
    print("Docs pages:", counts["docs"])
    print("Marketing pages:", counts["marketing"])
    print("Saved to:", RAW_DIR)

def main():
    cfg = load_sources()
    aws = cfg["providers"]["aws"]

    # Take all AWS URLs from config as start URLs
    start_urls = []
    for cat, urls in aws.get("web_roots", {}).items():
        start_urls.extend(urls)

    # Key knobs to avoid your earlier issue:
    crawl(
        start_urls=start_urls,
        max_depth_docs=3,         # increase to 4 only if needed
        max_depth_marketing=2,
        max_pages_docs=6000,      # raise to 8000 if needed
        max_pages_marketing=800
    )

if __name__ == "__main__":
    main()
