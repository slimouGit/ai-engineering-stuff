import os
from typing import List, Dict


def load_txt_documents(data_dir: str) -> List[Dict]:
    """
    Lädt alle .txt-Dateien aus einem Ordner.
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data-Ordner nicht gefunden: {data_dir}")

    documents = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "source": filename,
                "text": text
            })

    if not documents:
        raise ValueError("Keine .txt-Dateien gefunden.")

    return documents


def chunk_text(text: str, chunk_size: int = 120) -> List[str]:
    """
    Zerlegt Text in kleinere Abschnitte.
    """

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def build_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Erstellt Chunks und behält die jeweilige Quelle.
    """

    chunks = []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return chunks