import requests


OLLAMA_URL = "http://localhost:11434"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"


def generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


def create_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embedding"]
