from typing import List, Dict

from base import LLM
from retriever import TfidfRetriever


class RAGService:
    """
    Zentrale RAG-Logik.

    Diese Klasse ist unabhängig vom konkreten LLM-Anbieter.
    Sie erwartet nur ein Objekt, das dem LLM-Interface entspricht.
    """

    def __init__(self, retriever: TfidfRetriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        """
        Baut aus Frage + gefundenen Textstellen den Prompt.
        """

        context = "\n\n".join(
            f"Quelle: {chunk['source']}\nText: {chunk['text']}"
            for chunk in chunks
        )

        return f"""
Beantworte die Frage ausschließlich anhand des folgenden Kontexts.

Wenn die Antwort nicht im Kontext steht, antworte:
"Diese Information steht nicht in den bereitgestellten Dokumenten."

Kontext:
{context}

Frage:
{question}

Antwort:
""".strip()

    def ask(self, question: str) -> Dict:
        """
        Führt den kompletten RAG-Ablauf aus.
        """

        chunks = self.retriever.retrieve(question)

        if not chunks:
            return {
                "answer": "Ich habe keine passende Stelle in den Dokumenten gefunden.",
                "sources": []
            }

        prompt = self.build_prompt(question, chunks)
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": chunks
        }