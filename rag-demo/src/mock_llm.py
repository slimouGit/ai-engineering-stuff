from base import LLM


class MockLLM(LLM):
    """
    Fake-LLM für Tests.

    Vorteil:
    Die RAG-App kann getestet werden, ohne dass Ollama,
    LM Studio oder Internet verfügbar sind.
    """

    def generate(self, prompt: str) -> str:
        return (
            "MOCK-ANTWORT:\n"
            "Ein echtes LLM würde hier auf Basis des Prompts antworten.\n\n"
            "Prompt-Ausschnitt:\n"
            f"{prompt[:500]}..."
        )