import os
from typing import List, Dict

from pypdf import PdfReader


def load_pdf_document(path: str) -> str:
    """
    Extrahiert Text aus einer PDF-Datei.
    Funktioniert bei PDFs mit echtem Text.
    Gescannte PDFs benötigen OCR und werden hier nicht unterstützt.
    """

    reader = PdfReader(path)
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text_parts.append(page_text)

    return "\n".join(text_parts)


def load_txt_document(path: str) -> str:
    """
    Lädt den Inhalt einer TXT-Datei.
    """

    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_documents(data_dir: str) -> List[Dict]:
    """
    Lädt alle unterstützten Dokumente aus dem data-Ordner.

    Unterstützte Dateitypen:
    - .txt
    - .pdf

    Rückgabe:
    Liste von Dictionaries:
    [
        {
            "source": "dateiname.pdf",
            "text": "extrahierter Text"
        }
    ]
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data-Ordner nicht gefunden: {data_dir}")

    documents = []

    for filename in os.listdir(data_dir):
        path = os.path.join(data_dir, filename)

        if filename.lower().endswith(".txt"):
            text = load_txt_document(path)

        elif filename.lower().endswith(".pdf"):
            text = load_pdf_document(path)

        else:
            continue

        if text.strip():
            documents.append({
                "source": filename,
                "text": text
            })

    if not documents:
        raise ValueError("Keine unterstützten Dokumente gefunden. Erlaubt: .txt, .pdf")

    return documents


def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
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
    Erstellt Chunks aus allen geladenen Dokumenten.
    Jeder Chunk behält die ursprüngliche Quelle.
    """

    chunks = []

    for doc in documents:
        for chunk in chunk_text(doc["text"]):
            chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return chunks