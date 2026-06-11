import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "rag.db")

OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 4

WEB_REQUEST_TIMEOUT_SECONDS = 10
WEB_MAX_CONTENT_BYTES = 2_000_000
WEB_USER_AGENT = "rag-demo-bot/1.0 (+local-dev)"