# python
import os
import sys

# Sicherstellen, dass das aktuelle Verzeichnis (`src`) in sys.path ist,
# damit lokale Paket-Imports wie `llm.base` funktionieren.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from document_loader import load_txt_documents, build_chunks
from retriever import TfidfRetriever
from rag_service import RAGService

from mock_llm import MockLLM
from ollama_llm import OllamaLLM
from lm_studio_llm import LMStudioLLM


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def create_llm(provider: str):
    """
    Factory-Methode für das gewünschte LLM.

    Nur hier entscheidest du, welches Modell genutzt wird.
    Die restliche App bleibt unverändert.
    """

    if provider == "mock":
        return MockLLM()

    if provider == "ollama":
        return OllamaLLM(
            model="llama3.2"
        )

    if provider == "lmstudio":
        return LMStudioLLM(
            model="local-model"
        )

    raise ValueError(f"Unbekannter LLM-Provider: {provider}")


def main():
    provider = input("LLM wählen [mock / ollama / lmstudio]: ").strip().lower()

    if not provider:
        provider = "mock"

    llm = create_llm(provider)

    documents = load_txt_documents(DATA_DIR)
    chunks = build_chunks(documents)

    retriever = TfidfRetriever(chunks)
    rag = RAGService(retriever=retriever, llm=llm)

    print("\nGenerische RAG Demo")
    print("-------------------")
    print(f"LLM-Provider: {provider}")
    print("Mit 'exit' beenden.\n")

    while True:
        question = input("Frage: ")

        if question.lower() in ["exit", "quit", "q"]:
            print("App beendet.")
            break

        if not question.strip():
            print("Bitte eine Frage eingeben.\n")
            continue

        result = rag.ask(question)

        print("\nAntwort:")
        print(result["answer"])

        print("\nQuellen:")
        for source in result["sources"]:
            print(f"- {source['source']} | Score: {source['score']}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()