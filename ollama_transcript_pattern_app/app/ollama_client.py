import json
import os
from typing import Any, Dict

import requests


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.3:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_MAX_OUTPUT_TOKENS = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "2048"))


class OllamaError(RuntimeError):
    """Diese Klasse beschreibt einen Fehler bei der Kommunikation mit Ollama."""

    pass


def chat_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    # Sendet den Analyseprompt an Ollama und erwartet eine JSON-Antwort.
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": OLLAMA_MAX_OUTPUT_TOKENS,
        },
    }

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            f"Ollama ist nicht erreichbar: {exc}. Läuft 'ollama serve' und ist das Modell vorhanden?"
        ) from exc

    try:
        content = response.json()["message"]["content"]
        return json.loads(content)
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        raise OllamaError("Ollama hat keine gültige JSON-Antwort geliefert.") from exc


def health() -> Dict[str, Any]:
    # Prüft Ollama ohne eine Analyse zu starten.
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return {"ok": True, "model": OLLAMA_MODEL}
    except requests.RequestException as exc:
        return {"ok": False, "model": OLLAMA_MODEL, "detail": str(exc)}
