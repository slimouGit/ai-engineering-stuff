"""
Lädt ein trainiertes Modell und führt Vorhersagen für neue SMS durch.
"""

import os
import joblib

# Projektbasis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pfad zum gespeicherten Modell
MODEL_PATH = os.path.join(BASE_DIR, "models", "sms_spam_model.joblib")


def load_model():
    """
    Lädt das gespeicherte Modell aus der Datei.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modell nicht gefunden: {MODEL_PATH}\n"
            "Bitte zuerst train_model.py ausführen."
        )

    return joblib.load(MODEL_PATH)


def predict_sms(text: str):
    """
    Führt eine Vorhersage für eine einzelne SMS durch.

    Parameter:
    text → Eingabetext

    Rückgabe:
    Dictionary mit Ergebnis
    """

    model = load_model()

    # Vorhersage (0 = ham, 1 = spam)
    prediction = model.predict([text])[0]

    # Wahrscheinlichkeit (für Spam)
    probability = model.predict_proba([text])[0]

    return {
        "text": text,
        "prediction": "SPAM" if prediction == 1 else "HAM",
        "spam_probability": round(float(probability[1]), 4)
    }


# Direkt ausführbar machen
if __name__ == "__main__":
    sms = input("SMS eingeben: ")

    result = predict_sms(sms)

    print("\nErgebnis:")
    print("Text:", result["text"])
    print("Vorhersage:", result["prediction"])
    print("Spam-Wahrscheinlichkeit:", result["spam_probability"])