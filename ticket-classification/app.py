from fastapi import FastAPI
from pydantic import BaseModel

from model import TicketClassifier


app = FastAPI()

classifier = TicketClassifier("model.joblib")


class TicketRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(request: TicketRequest):
    category = classifier.predict(request.text)

    return {
        "category": category
    }