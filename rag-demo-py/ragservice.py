# python
import json
import math
from database import get_connection
from documentservice import extract_text_from_file, extract_text_from_url
from ollamaservice import OllamaService
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


class RagService:
    def __init__(self):
        self.ollama = OllamaService()
        # defaults - may be overridden by available models or set_* methods
        self.current_chat_model = None
        self.current_embedding_model = None

        models = self.ollama.list_models()

        embedding_terms = ("embed", "embedding", "bge", "minilm")
        embedding_models = [m for m in models if any(t in m.lower() for t in embedding_terms)]
        chat_models = [m for m in models if not any(t in m.lower() for t in embedding_terms)]

        self.current_embedding_model = embedding_models[0] if embedding_models else (models[0] if models else None)
        self.current_chat_model = chat_models[0] if chat_models else (models[0] if models else None)

    def get_ollama_status(self) -> dict:
        models = self.ollama.list_models()

        return {
            "available_models": models,
            "current_chat_model": self.current_chat_model,
            "current_embedding_model": self.current_embedding_model
        }

    def set_chat_model(self, model_name: str) -> None:
        status = self.get_ollama_status()
        if model_name not in status["available_models"]:
            raise ValueError("Model not available: {}".format(model_name))
        self.current_chat_model = model_name

    def set_embedding_model(self, model_name: str) -> None:
        status = self.get_ollama_status()
        if model_name not in status["available_models"]:
            raise ValueError("Model not available: {}".format(model_name))
        self.current_embedding_model = model_name

    def ingest_document(self, document_name: str, file_path: str) -> int:
        text = extract_text_from_file(file_path)
        return self._ingest_text(document_name, text)

    def ingest_url(self, url: str) -> tuple[str, int]:
        extracted = extract_text_from_url(url)
        document_name = f"URL: {extracted['normalized_url']}"
        chunk_count = self._ingest_text(document_name, extracted["text"])
        return document_name, chunk_count

    def _ingest_text(self, document_name: str, text: str) -> int:
        chunks = self._split_into_chunks(text)

        if not chunks:
            raise ValueError("Das Dokument enthält keinen lesbaren Text.")

        with get_connection() as conn:
            conn.execute(
                "DELETE FROM document_chunks WHERE document_name = ?",
                (document_name,)
            )

            for index, chunk in enumerate(chunks):
                embedding = self.ollama.embed(chunk, model=self.current_embedding_model)

                conn.execute("""
                    INSERT INTO document_chunks (
                        document_name,
                        chunk_index,
                        content,
                        embedding
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    document_name,
                    index,
                    chunk,
                    json.dumps(embedding)
                ))

        return len(chunks)

    def ask(self, document_name: str, question: str) -> dict:
        question_embedding = self.ollama.embed(question, model=self.current_embedding_model)

        chunks = self._load_chunks(document_name)

        if not chunks:
            raise ValueError("Dokument wurde nicht gefunden oder noch nicht indexiert.")

        scored_chunks = []

        for chunk in chunks:
            similarity = self._cosine_similarity(
                question_embedding,
                chunk["embedding"]
            )

            scored_chunks.append({
                "chunk_id": chunk["id"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "score": similarity
            })

        relevant_chunks = sorted(
            scored_chunks,
            key=lambda item: item["score"],
            reverse=True
        )[:TOP_K]

        context = "\n\n---\n\n".join(
            chunk["content"] for chunk in relevant_chunks
        )

        answer = self.ollama.generate_answer(context, question, model=self.current_chat_model)

        return {
            "answer": answer,
            "chunks": relevant_chunks,
            "used_models": {
                "chat_model": self.current_chat_model,
                "embedding_model": self.current_embedding_model
            }
        }

    def get_documents(self) -> list[str]:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT document_name
                FROM document_chunks
                ORDER BY document_name
            """).fetchall()

        return [row[0] for row in rows]

    def _load_chunks(self, document_name: str) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT id, chunk_index, content, embedding
                FROM document_chunks
                WHERE document_name = ?
                ORDER BY chunk_index
            """, (document_name,)).fetchall()

        return [
            {
                "id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "embedding": json.loads(row[3])
            }
            for row in rows
        ]

    def _split_into_chunks(self, text: str) -> list[str]:
        cleaned = (
            text.replace("\r", "")
            .replace("\t", " ")
            .strip()
        )

        if not cleaned:
            return []

        chunks = []
        start = 0

        while start < len(cleaned):
            end = min(start + CHUNK_SIZE, len(cleaned))
            chunk = cleaned[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == len(cleaned):
                break

            start = max(0, end - CHUNK_OVERLAP)

        return chunks

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)