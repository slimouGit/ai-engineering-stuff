from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .analyzer import analyze_transcript
from .evaluation import evaluate_matches, load_ground_truth
from .ollama_client import OllamaError, health
from .schemas import AnalysisResponse


BASE_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPT_PATH = BASE_DIR / "data" / "interview.txt"
app = FastAPI(title="Transcript Pattern Analyzer", version="1.0.0")
latest_analysis: AnalysisResponse | None = None


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
    global latest_analysis
    latest_analysis = None
    try:
        latest_analysis = analyze_transcript(read_transcript())
        return latest_analysis
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/evaluate")
def evaluate_current_analysis():
    """Bewertet ausschließlich das zuletzt gespeicherte Analyseergebnis."""
    if latest_analysis is None:
        raise HTTPException(
            status_code=409,
            detail="Bitte zuerst die Analyse ausführen.",
        )

    ground_truth = load_ground_truth()
    predictions = [
        {"pattern": match.pattern, "evidence": match.evidence}
        for match in latest_analysis.matches
    ]

    return {
        "model": latest_analysis.model,
        "ground_truth_loaded": bool(ground_truth),
        "evaluation": evaluate_matches(predictions, ground_truth),
    }


