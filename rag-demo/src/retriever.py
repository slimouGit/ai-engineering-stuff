from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """
    Sucht relevante Textstellen anhand semantischer bzw. statistischer Ähnlichkeit.

    Diese Klasse kennt kein LLM.
    Sie macht nur Retrieval.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [chunk["text"] for chunk in chunks]

        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Gibt die relevantesten Chunks zur Frage zurück.
        """

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for index in top_indices:
            score = float(similarities[index])

            if score > 0:
                results.append({
                    "source": self.chunks[index]["source"],
                    "text": self.chunks[index]["text"],
                    "score": round(score, 4)
                })

        return results