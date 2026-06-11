# python
# File: `api.py`
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ragservice import RagService

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # im Prod auf die FE-Origin beschränken
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

rag = RagService()


class ModelSelection(BaseModel):
    model: str


@app.get("/api/ollama/status")
def ollama_status():
    """
    Gibt die verfügbaren lokalen Modelle und die aktuell gesetzten Modelle zurück.
    """
    return rag.get_ollama_status()


@app.post("/api/ollama/chat-model")
def set_chat_model(sel: ModelSelection):
    """
    Setzt das Chat-/Generatormodell. Fehler bei nicht verfügbarem Modell.
    """
    try:
        rag.set_chat_model(sel.model)
        return {"ok": True, "current_chat_model": rag.current_chat_model}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/ollama/embedding-model")
def set_embedding_model(sel: ModelSelection):
    """
    Setzt das Embedding-Modell. Fehler bei nicht verfügbarem Modell.
    """
    try:
        rag.set_embedding_model(sel.model)
        return {"ok": True, "current_embedding_model": rag.current_embedding_model}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Optional: expose which models were used by the last ask (falls benötigt)
@app.get("/api/ollama/used")
def used_models():
    return {
        "chat_model": rag.current_chat_model,
        "embedding_model": rag.current_embedding_model
    }


# Frontend-Beispiele (JS \- fetch):
# fetch('/api/ollama/status').then(r=>r.json()).then(console.log)
# fetch('/api/ollama/chat-model', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({model: 'llama3.2:latest'}) })
# fetch('/api/ollama/embedding-model', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({model: 'nomic-embed-text:latest'}) })
#
# Starten: uvicorn api:app --reload --port 8000