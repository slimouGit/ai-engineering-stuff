from pypdf import PdfReader


def extract_text_from_file(file_path: str) -> str:
    lower_path = file_path.lower()

    if lower_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    if lower_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    raise ValueError("Nur TXT- und PDF-Dateien werden unterstützt.")