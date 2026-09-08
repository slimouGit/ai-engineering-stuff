import joblib


class TicketClassifier:

    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        return self.model.predict([text])[0]