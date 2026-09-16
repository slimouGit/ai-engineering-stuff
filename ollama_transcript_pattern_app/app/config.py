import os
import re


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "granite3.3:8b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_MAX_OUTPUT_TOKENS = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "2048"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "6000"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "2"))
ABBREVIATION_PATTERN = re.compile(
    r"\b(?:Dr|Prof|Herr|Frau|z\.\s*B|d\.\s*h|bzw|usw|etc|vgl)\.",
    re.IGNORECASE,
)
PERIOD_MARKER = "__PERIOD__"
