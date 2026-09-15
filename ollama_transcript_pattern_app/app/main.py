from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .analyzer import analyze_transcript
from .ollama_client import OllamaError, health
from .schemas import AnalysisResponse


BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_PATH = BASE_DIR / "data" / "interview.txt"
app = FastAPI(title="Transcript Pattern Analyzer", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index():
    # Liefert die Browser-Oberfläche aus dem templates-Ordner.
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


def read_transcript() -> str:
    # Liest das feste Interview und verhindert eine Analyse ohne Text.
    try:
        transcript = TRANSCRIPT_PATH.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Das Interview-Transkript konnte nicht gelesen werden.") from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="Das Interview-Transkript ist leer.")
    return transcript


@app.get("/health")
def health_check():
    # Prüft, ob Ollama erreichbar ist.
    return health()


@app.get("/transcript")
def transcript():
    # Stellt das Interview für die Oberfläche bereit.
    return {"transcript": read_transcript()}


@app.post("/analyze-interview", response_model=AnalysisResponse)
def analyze_interview():
    # Analysiert das vorhandene Interview mit den Standardmustern.
    try:
        return analyze_transcript(read_transcript())
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


