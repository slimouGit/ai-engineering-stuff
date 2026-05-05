import requests
from base import LLM


class LMStudioLLM(LLM):
    """
    LLM-Adapter für LM Studio über OpenAI-kompatible Chat-Completions-API.

    Typische lokale URL:
    http://localhost:1234/v1/chat/completions
    """

    def __init__(
        self,
        model: str = "local-model",
        url: str = "http://localhost:1234/v1/chat/completions",
        timeout: int = 120
    ):
        self.model = model
        self.url = url
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Du bist ein sachlicher Assistent."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2
        }

        response = requests.post(
            self.url,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"].strip()