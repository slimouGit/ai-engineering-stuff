"""
Zentrale Anwendungskonfiguration.
"""

# Welcher Provider soll genutzt werden?
LLM_PROVIDER = "ollama"

# Unterstützte Werte:
# - ollama
# - lmstudio
# - mock

# Modellname
LLM_MODEL = "llama3.2"

# API-URL
LLM_URL = "http://localhost:11434/api/generate"

# Timeout in Sekunden
LLM_TIMEOUT = 120

# Retrieval-Konfiguration
TOP_K_RESULTS = 5

# Chunking
CHUNK_SIZE = 200