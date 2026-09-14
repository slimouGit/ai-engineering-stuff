import os

import numpy as np

from ollama_client import create_embedding


class RagService:

    def __init__(self):
        self.documents = []
        self.embeddings = []

        self._load_documents()

    def _load_documents(self):

        docs_path = os.path.join(
            os.path.dirname(__file__),
            "docs"
        )

        for filename in os.listdir(docs_path):

            if not filename.endswith(".md"):
                continue

            file_path = os.path.join(
                docs_path,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

            embedding = create_embedding(content)

            self.documents.append({
                "filename": filename,
                "content": content
            })

            self.embeddings.append(embedding)

    def search(self, query: str) -> dict:

        query_embedding = create_embedding(query)

        similarities = []

        for document_embedding in self.embeddings:

            similarity = self._cosine_similarity(
                query_embedding,
                document_embedding
            )

            similarities.append(similarity)

        best_index = int(
            np.argmax(similarities)
        )

        document = self.documents[best_index]

        return {
            "filename": document["filename"],
            "content": document["content"],
            "score": float(
                similarities[best_index]
            )
        }

    @staticmethod
    def _cosine_similarity(a, b) -> float:

        a = np.array(a)
        b = np.array(b)

        denominator = (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )
