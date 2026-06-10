import json
import math
from database import get_connection
from documentservice import extract_text_from_file
from ollamaservice import OllamaService
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K


class RagService:
    def __init__(self):
        self.ollama = OllamaService()

    def ingest_document(self, document_name: str, file_path: str) -> int:
        text = extract_text_from_file(file_path)
        chunks = self._split_into_chunks(text)

        if not chunks:
            raise ValueError("Das Dokument enthält keinen lesbaren Text.")

        with get_connection() as conn:
            conn.execute(
                "DELETE FROM document_chunks WHERE document_name = ?",
                (document_name,)
            )

            for index, chunk in enumerate(chunks):
                embedding = self.ollama.embed(chunk)

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
        question_embedding = self.ollama.embed(question)
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

        answer = self.ollama.generate_answer(context, question)

        return {
            "answer": answer,
            "chunks": relevant_chunks
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
                SELECT content, embedding
                FROM document_chunks
                WHERE document_name = ?
                ORDER BY chunk_index
            """, (document_name,)).fetchall()

        return [
            {
                "content": row[0],
                "embedding": json.loads(row[1])
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