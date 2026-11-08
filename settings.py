"""User data lives outside the app folder (default ~/GapFinder, override with GAPFINDER_HOME) so updating the app never loses projects or keys."""
import os, json
from pathlib import Path
KEYS = ["GROQ_API_KEY", "GROQ_MODEL", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE", "OPENALEX_MAILTO", "EMBED_MODEL", "GAPFINDER_USE_NEO4J", "LLM_PROVIDER", "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "LLM_TEMPERATURE", "GEMINI_API_KEY", "GEMINI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"]
def home(): d = Path(os.getenv("GAPFINDER_HOME") or Path.home() / "GapFinder"); d.mkdir(parents=True, exist_ok=True); return d
def projects_dir(): d = home() / "projects"; d.mkdir(parents=True, exist_ok=True); return d
def load():
    f = home() / "settings.json"
    try: return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except ValueError: return {}
def save(d):
    f = home() / "settings.json"; cur = load(); cur.update({k: v for k, v in d.items() if k in KEYS}); f.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    try: os.chmod(f, 0o600)
    except OSError: pass
def apply():
    """Settings saved in the app win over .env / shell variables."""
    for k, v in load().items():
        if v: os.environ[k] = str(v)
