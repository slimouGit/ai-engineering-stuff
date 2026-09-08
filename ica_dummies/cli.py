# python
import os
import sys
import json
import importlib.util
from pathlib import Path
from typing import Any, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def load_config():
    """Try package import, fallback to local config.py, then minimal defaults."""
    try:
        from ica_dummies import config as cfg  # type: ignore
        return cfg
    except Exception:
        cfg_path = Path(__file__).with_name("config.py")
        if cfg_path.exists():
            spec = importlib.util.spec_from_file_location("ica_dummies.config", str(cfg_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            return mod
    class C:
        ICA_TOKEN = ""
        ICA_BASE_URL = "https://api.nextgen-beta.ica.ibm.com/ica/v1"
    return C()

cfg = load_config()
TOKEN = (cfg.ICA_TOKEN or os.getenv("ICA_TOKEN", "")).strip()
BASE = (getattr(cfg, "ICA_BASE_URL", "") or "https://api.nextgen-beta.ica.ibm.com/ica/v1").rstrip("/")

# Normalize accidental embedded paths so final endpoints are predictable.
for tail in ("/chat-models", "/agents", "/agents/chat/completions"):
    if BASE.endswith(tail):
        BASE = BASE[: -len(tail)]
BASE = BASE.rstrip("/")

AGENTS_URL = f"{BASE}/agents"
COMPLETIONS_URL = f"{BASE}/agents/chat/completions"

if not TOKEN:
    sys.exit("ICA_TOKEN not found in `ica_dummies.config` or env var ICA_TOKEN")

if not TOKEN.lower().startswith("bearer "):
    TOKEN = f"Bearer {TOKEN}"

HEADERS = {
    "Accept": "application/json",
    "Authorization": TOKEN,
    "Content-Type": "application/json",
    "User-Agent": "ica-dummies-client/1.0",
}

def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def fetch_agents(sess: requests.Session) -> List[Any]:
    print("Fetching agents from:", AGENTS_URL)
    r = sess.get(AGENTS_URL, headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()
    candidates = None
    if isinstance(data, dict):
        candidates = data.get("agents") or data.get("items") or data.get("results")
    if not candidates:
        candidates = data
    if isinstance(candidates, dict):
        candidates = list(candidates.values())
    return candidates if isinstance(candidates, list) else []

def safe_field(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        return obj.get(key)
    if isinstance(obj, (list, tuple)):
        for it in obj:
            if isinstance(it, dict) and key in it:
                return it.get(key)
    return None

def choose_agent(agents: List[Any], desired: Optional[str]) -> Optional[Any]:
    if desired:
        target = desired.strip().lower()
        for a in agents:
            for k in ("id", "name", "displayName", "model"):
                v = safe_field(a, k)
                if v and str(v).lower() == target:
                    return a
    return None

def show_agents(agents: List[Any]) -> None:
    print("Available agents (id / name / displayName):")
    for a in agents[:200]:
        idv = safe_field(a, "id") or "<no-id>"
        namev = safe_field(a, "name") or "<no-name>"
        disp = safe_field(a, "displayName") or "<no-displayName>"
        if idv == "<no-id>" and not isinstance(a, dict):
            idv = str(a)
        print("-", idv, "/", namev, "/", disp)

def call_completion(sess: requests.Session, model_value: str, prompt: str = "Summarise our Q1 spend."):
    payload = {
        "model": model_value,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    print("Calling completions URL:", COMPLETIONS_URL)
    r = sess.post(COMPLETIONS_URL, headers=HEADERS, json=payload, timeout=20)
    print("Status Code:", r.status_code)
    try:
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(r.text)
    r.raise_for_status()

def main():
    desired = os.getenv("ICA_AGENT", "").strip()
    sess = make_session()
    try:
        agents = fetch_agents(sess)
    except requests.RequestException as e:
        print("Failed to fetch agents:", e)
        sys.exit(1)

    if not agents:
        print("No agents returned; inspect the endpoint or token permissions.")
        sys.exit(1)

    chosen = choose_agent(agents, desired)
    if not chosen:
        show_agents(agents)
        if not desired:
            sys.exit("Set the `ICA_AGENT` env var and re-run.")
        sys.exit("Provided ICA_AGENT not found; choose one from the list above.")

    model_value = safe_field(chosen, "name") or safe_field(chosen, "id")
    if not model_value:
        print("Chosen agent lacks name/id.")
        sys.exit(1)

    try:
        call_completion(sess, model_value)
    except requests.RequestException as e:
        print("Completion request failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()