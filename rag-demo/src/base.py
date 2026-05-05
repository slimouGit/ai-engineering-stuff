from abc import ABC, abstractmethod


class LLM(ABC):
    """
    Gemeinsames Interface für alle LLM-Anbieter.

    Egal ob Ollama, LM Studio, OpenAI, Azure oder Mock:
    Jede Implementierung muss nur generate(prompt) anbieten.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass