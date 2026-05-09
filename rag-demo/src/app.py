import os

from settings import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_URL,
    LLM_TIMEOUT
)

from document_loader import load_documents, build_chunks
from retriever import TfidfRetriever
from rag_service import RAGService

from mock_llm import MockLLM
from ollama_llm import OllamaLLM
from lm_studio_llm import LMStudioLLM


# Projekt-Basisordner bestimmen
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pfad zum data-Ordner
DATA_DIR = os.path.join(BASE_DIR, "data")


def create_llm():
    """
    Factory-Methode für die Erstellung des konfigurierten LLMs.

    Die konkrete Implementierung wird ausschließlich
    über die Konfiguration bestimmt.

    Dadurch bleibt die restliche Anwendung unabhängig
    vom verwendeten Modellanbieter.
    """

    # Mock-LLM für lokale Tests ohne echtes Modell
    if LLM_PROVIDER == "mock":
        return MockLLM()

    # Ollama-Implementierung
    if LLM_PROVIDER == "ollama":
        return OllamaLLM(
            model=LLM_MODEL,
            url=LLM_URL,
            timeout=LLM_TIMEOUT
        )

    # LM Studio Implementierung
    if LLM_PROVIDER == "lmstudio":
        return LMStudioLLM(
            model=LLM_MODEL,
            url=LLM_URL,
            timeout=LLM_TIMEOUT
        )

    # Fehler bei unbekannter Konfiguration
    raise ValueError(
        f"Unbekannter LLM-Provider: {LLM_PROVIDER}"
    )


def main():
    """
    Haupteinstiegspunkt der Anwendung.

    Ablauf:
    1. Dokumente laden
    2. Dokumente in Chunks zerlegen
    3. Retriever initialisieren
    4. Konfiguriertes LLM erstellen
    5. RAG-Service starten
    6. Nutzerfragen verarbeiten
    """

    print("Starte RAG-Anwendung...")
    print(f"LLM-Provider: {LLM_PROVIDER}")
    print(f"LLM-Modell: {LLM_MODEL}")
    print()

    # Dokumente laden (.txt und .pdf)
    documents = load_documents(DATA_DIR)

    print(f"{len(documents)} Dokument(e) geladen.")

    # Dokumente in kleinere Chunks zerlegen
    chunks = build_chunks(documents)

    print(f"{len(chunks)} Chunk(s) erstellt.")
    print()

    # Retriever initialisieren
    retriever = TfidfRetriever(chunks)

    # Konfiguriertes LLM erzeugen
    llm = create_llm()

    # RAG-Service erzeugen
    rag = RAGService(
        retriever=retriever,
        llm=llm
    )

    print("RAG-System bereit.")
    print("Mit 'exit' beenden.")
    print("-" * 50)

    # Hauptschleife
    while True:

        question = input("\nFrage: ").strip()

        # Anwendung beenden
        if question.lower() in ["exit", "quit", "q"]:
            print("Anwendung beendet.")
            break

        # Leere Eingaben verhindern
        if not question:
            print("Bitte eine Frage eingeben.")
            continue

        try:
            # RAG-Abfrage ausführen
            result = rag.ask(question)

            print("\nAntwort:")
            print(result["answer"])

            print("\nQuellen:")

            for source in result["sources"]:
                print(
                    f"- {source['source']} "
                    f"(Score: {source['score']})"
                )

        except Exception as e:
            print("\nFehler bei der Verarbeitung:")
            print(str(e))

        print("\n" + "-" * 50)


# Direkter Programmeinstieg
if __name__ == "__main__":
    main()