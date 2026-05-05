import requests
from base import LLM


class OllamaLLM(LLM):
    """
    LLM-Adapter für Ollama.
    Standard-URL: http://localhost:11434/api/generate
    """

    def __init__(
        self,
        model: str = "llama3.2",
        url: str = "http://localhost:11434/api/generate",
        timeout: int = 120
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