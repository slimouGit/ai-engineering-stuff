# python
# File: `ica_dummies/ica-completions.py`
import os
import sys
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# config (unchanged)
try:
    from ica_dummies.config import ICA_TOKEN
except Exception:
    ICA_TOKEN = os.getenv("ICA_TOKEN")
if not ICA_TOKEN:
    sys.exit("ICA_TOKEN not found in `ica_dummies.config` or env var ICA_TOKEN")
if not ICA_TOKEN.strip().lower().startswith("bearer "):
    ICA_TOKEN = f"Bearer {ICA_TOKEN.strip()}"

BASE = "https://api.nextgen-beta.ica.ibm.com/ica/v1"
AGENTS_URL = f"{BASE}/agents"
COMPLETIONS_URL = f"{BASE}/agents/chat/completions"

headers = {
    "Accept": "application/json",
    "Authorization": ICA_TOKEN,
    "Content-Type": "application/json",
    "User-Agent": "ica-completions-client/1.0",
}

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET", "POST"])
session.mount("https://", HTTPAdapter(max_retries=retries))

def list_agents():
    resp = session.get(AGENTS_URL, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    agents = data.get("agents") or data.get("items") or data.get("results") or data
    if isinstance(agents, dict):
        agents = [v for v in agents.values()]
    return agents if isinstance(agents, list) else []

def get_agent_field(agent, key):
    """Safely extract `key` from an agent that may be a dict, a list/tuple containing dicts, or a primitive."""
    if isinstance(agent, dict):
        return agent.get(key)
    if isinstance(agent, (list, tuple)):
        # prefer first dict that contains the key
        for item in agent:
            if isinstance(item, dict) and key in item:
                return item.get(key)
    # fallback: if agent itself is a primitive and the key requested is common, try converting
    return None

DESIRED = os.getenv("ICA_AGENT", "").strip()

try:
    agents = list_agents()
except requests.RequestException as e:
    print("Failed to list agents:", e)
    try:
        print("Raw response:", e.response.text)
    except Exception:
        pass
    sys.exit(1)

if not agents:
    print("No agents returned. Response may have a different shape; inspect the raw listing above.")
    sys.exit(1)

chosen = None
if DESIRED:
    for a in agents:
        for k in ("id", "name", "displayName", "model"):
            val = get_agent_field(a, k)
            if val and str(val).lower() == DESIRED.lower():
                chosen = a
                break
        if chosen:
            break

if not chosen:
    print("No matching agent for ICA_AGENT; available agents (id / name / displayName):")
    for a in agents[:50]:
        idv = get_agent_field(a, "id") or "<no-id>"
        namev = get_agent_field(a, "name") or "<no-name>"
        disp = get_agent_field(a, "displayName") or "<no-displayName>"
        # if agent is a primitive or list without dicts, provide a string fallback
        if idv == "<no-id>" and not isinstance(a, dict):
            idv = str(a)
        print("-", idv, "/", namev, "/", disp)
    if not DESIRED:
        sys.exit("Set the `ICA_AGENT` env var to one of the above identifiers and re-run.")
    else:
        sys.exit("Provided ICA_AGENT not found. Update ICA_AGENT to a valid id/name/displayName from the list above.")

# select model_value using the safe accessor
model_value = get_agent_field(chosen, "name") or get_agent_field(chosen, "id")
if not model_value:
    print("Chosen agent has no usable `name` or `id` fields, aborting.")
    sys.exit(1)

payload = {
    "model": model_value,
    "messages": [{"role": "user", "content": "Summarise our Q1 spend."}],
    "stream": False,
}

try:
    resp = session.post(COMPLETIONS_URL, headers=headers, json=payload, timeout=15)
    print("Status Code:", resp.status_code)
    try:
        parsed = resp.json()
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    except ValueError:
        print(resp.text)
    resp.raise_for_status()
except requests.HTTPError as e:
    print("Request failed:", e)
    try:
        print("Error payload:", resp.json())
    except Exception:
        print("Error text:", resp.text)
    sys.exit(1)
except requests.RequestException as e:
    print("Request failed:", e)
    sys.exit(1)