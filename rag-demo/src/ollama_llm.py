import requests
from base import LLM
from settings import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_URL,
    LLM_TIMEOUT
)


class OllamaLLM(LLM):
    """
    LLM-Adapter für Ollama.
    Standard-URL: http://localhost:11434/api/generate
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        url: str = LLM_URL,
        timeout: int = LLM_TIMEOUT
    ):
        self.model = model
        self.url = url
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(
            self.url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()