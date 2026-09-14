# File: `app/main.py`
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "model.joblib"

def _try_load_model():
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception as e:
            print(f"Failed to load model at {MODEL_FILE}: {e}")
    return None

model = _try_load_model()

app = FastAPI(
    title="Document Classification API",
    description="Einfaches ML-Beispielprojekt zur Vorbereitung auf eine AI/Data-Science-Rolle.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="`model.joblib` not found. Run `python train.py` to create it."
        )

    # prefer predict_proba, fallback to predict
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([request.text])[0]
        best_index = int(probabilities.argmax())
        label = str(model.classes_[best_index])
        confidence = round(float(probabilities[best_index]), 4)
    else:
        label = str(model.predict([request.text])[0])
        confidence = 1.0

    return PredictionResponse(label=label, confidence=confidence)


if __name__ == "__main__":
    import uvicorn
    # when running `python app/main.py` start the server directly (no auto-reload)
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)