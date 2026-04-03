# github_link_crawler.py
import re
import json
import time
import hashlib
import argparse
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
AUTO_LINK_RE = re.compile(r"<(https?://[^>]+)>")

EXCLUDE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip")


def detect_provider_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()

    if "docs.aws.amazon.com" in host or "amazonaws.com" in host:
        return "aws"
    if "cloud.google.com" in host or "googlecloud" in host:
        return "gcp"
    if "learn.microsoft.com" in host or "azure" in host:
        return "azure"
    return "unknown"


def stable_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def extract_http_links_from_markdown(md: str) -> set[str]:
    links = set()
    for _, url in MD_LINK_RE.findall(md):
        links.add(url.strip())
    for url in AUTO_LINK_RE.findall(md):
        links.add(url.strip())
    return links


def extract_http_links_from_html(html: str, base_url: str) -> set[str]:
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


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Heuristic: prefer main/article if present
    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        return ""

    text = main.get_text(separator="\n")
    # Normalize blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def should_keep_url(url: str, allowed_domains: set[str] | None) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False

    if p.scheme not in ("http", "https"):
        return False

    # Skip obvious binaries
    if any(p.path.lower().endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return False

    if allowed_domains:
        host = (p.hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in allowed_domains)

    return True


def fetch(url: str, timeout_s: int = 20) -> tuple[str, str]:
    """
    Returns (final_url, html)
    """
    headers = {"User-Agent": "rag-crawler/1.0 (respectful; contact: you@example.com)"}
    r = requests.get(url, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    return (r.url, r.text)


def write_markdown_doc(out_dir: Path, url: str, referrer_doc_id: str, depth: int, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    docname = f"{stable_id(url)}.md"
    out_path = out_dir / docname

    md = (
        "---\n"
        f"source_url: {url}\n"
        f"referrer_doc_id: {referrer_doc_id}\n"
        f"crawl_depth: {depth}\n"
        "---\n\n"
        f"# {url}\n\n"
        f"{text}\n"
    )
    out_path.write_text(md, encoding="utf-8")
    return out_path


def crawl_from_github_index(
    index_jsonl: Path,
    out_dir: Path,
    max_depth: int,
    max_pages: int,
    allowed_domains: set[str] | None,
    same_domain_only: bool,
    sleep_s: float,
):
    """
    BFS crawl:
      seed URLs come from markdown files listed in index_jsonl
    """
    index_jsonl = Path(index_jsonl)
    out_dir = Path(out_dir)

    # Queue entries: (url, depth, referrer_doc_id, root_domain_for_lock)
    queue = []
    visited = set()
    written = 0

    with index_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            md_path = Path(entry["abs_path"])
            doc_id = entry["doc_id"]

            try:
                md = md_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for url in extract_http_links_from_markdown(md):
                if not should_keep_url(url, allowed_domains):
                    continue

                root_domain = urlparse(url).hostname if same_domain_only else None
                queue.append((url, 1, doc_id, root_domain))

    # BFS
    while queue and written < max_pages:
        url, depth, ref_doc_id, domain_lock = queue.pop(0)
        if url in visited:
            continue
        if depth > max_depth:
            continue

        # Domain lock: only follow within the first domain encountered from the markdown link
        if same_domain_only and domain_lock:
            host = (urlparse(url).hostname or "")
            if host != domain_lock and not host.endswith("." + domain_lock):
                continue

        visited.add(url)

        try:
            final_url, html = fetch(url)
            text = html_to_text(html)
            if len(text) < 200:  # skip extremely thin pages
                continue

            #write_markdown_doc(out_dir, final_url, ref_doc_id, depth, text)
            provider = detect_provider_from_url(final_url)
            provider_dir = out_dir / provider
            
            write_markdown_doc(
                provider_dir,
                final_url,
                ref_doc_id,
                depth,
                text
            )
            written += 1

            # Enqueue next layer
            if depth < max_depth:
                for nxt in extract_http_links_from_html(html, final_url):
                    if nxt in visited:
                        continue
                    if not should_keep_url(nxt, allowed_domains):
                        continue
                    queue.append((nxt, depth + 1, ref_doc_id, domain_lock))

            time.sleep(sleep_s)

        except Exception:
            # intentionally swallow; you can log if you want
            continue

    print(f"Saved {written} crawled pages to: {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/processed/docs/github_index.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/github_external"))
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-pages", type=int, default=500)
    ap.add_argument("--allowed-domains", type=str, default="")
    ap.add_argument("--same-domain-only", action="store_true")
    ap.add_argument("--sleep-s", type=float, default=0.25)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    domains = {d.strip().lower() for d in args.allowed_domains.split(",") if d.strip()} or None

    crawl_from_github_index(
        index_jsonl=args.index,
        out_dir=args.out_dir,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        allowed_domains=domains,
        same_domain_only=args.same_domain_only,
        sleep_s=args.sleep_s,
    )
