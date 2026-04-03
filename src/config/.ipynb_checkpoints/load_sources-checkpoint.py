import yaml
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"

'''
def load_sources():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
'''

def load_sources():
     """
    Load source configuration from sources.yaml.
    
    Returns:
        dict: Parsed YAML configuration containing providers, URLs, and GitHub repos.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"sources.yaml not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    cfg = load_sources()
    for provider, data in cfg["providers"].items():
        print(f"Provider: {provider}")
        for section, urls in data.get("web_roots", {}).items():
            print(f"  {section}: {len(urls)} URLs")
        print(f"  GitHub repos: {len(data.get('github_repos', []))}")
