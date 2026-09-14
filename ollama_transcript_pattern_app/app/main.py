from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .analyzer import DEFAULT_PATTERNS, analyze_transcript
from .ollama_client import OllamaError, health
from .schemas import AnalysisResponse, TextAnalysisRequest


BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_PATH = BASE_DIR / "data" / "interview.txt"
app = FastAPI(title="Transcript Pattern Analyzer", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health_check():
    return health()


@app.get("/patterns")
def patterns():
    return [p.model_dump() for p in DEFAULT_PATTERNS]


@app.get("/transcript")
def transcript():
    try:
        return {"transcript": TRANSCRIPT_PATH.read_text(encoding="utf-8")}
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Das Interview-Transkript konnte nicht gelesen werden.") from exc


@app.post("/analyze-interview", response_model=AnalysisResponse)
def analyze_interview():
    try:
        transcript_text = TRANSCRIPT_PATH.read_text(encoding="utf-8").strip()
        if not transcript_text:
            raise HTTPException(status_code=422, detail="Das Interview-Transkript ist leer.")
        return analyze_transcript(transcript_text)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Das Interview-Transkript konnte nicht gelesen werden.") from exc


@app.post("/analyze-text", response_model=AnalysisResponse)
def analyze_text(request: TextAnalysisRequest):
    try:
        return analyze_transcript(request.transcript, request.patterns)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


