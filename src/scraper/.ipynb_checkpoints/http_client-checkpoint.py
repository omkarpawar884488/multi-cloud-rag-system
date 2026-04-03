import time
import requests

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "CloudRAGBot/0.1 (for personal RAG experimentation)"
})

def fetch_html(url: str, max_retries: int = 3, delay: float = 0.5) -> str | None:
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code in (429, 503):
                time.sleep(delay * (attempt + 1))
            else:
                return None
        except requests.RequestException:
            time.sleep(delay * (attempt + 1))
    return None
