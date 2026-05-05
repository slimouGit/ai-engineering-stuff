"""
Trainiert ein Machine-Learning-Modell zur Spam-Erkennung.

Pipeline:
1. Daten laden
2. Daten vorbereiten
3. Training/Test Split
4. Modell trainieren
5. Modell evaluieren
6. Modell speichern
"""

import os
import pandas as pd
import joblib

# ML-Komponenten aus scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Projekt-Basisverzeichnis
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pfade für Daten und Modell
DATA_PATH = os.path.join(BASE_DIR, "data", "sms_spam.tsv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "sms_spam_model.joblib")


def load_data():
    """
    Lädt den Datensatz und bereitet ihn vor.

    Rückgabe:
    → DataFrame mit numerischen Labels
    """

    # Prüfen, ob Datei existiert
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Datensatz nicht gefunden: {DATA_PATH}\n"
            "Bitte zuerst download_data.py ausführen."
        )

    # Datei laden (hat Header: label, text)
    df = pd.read_csv(DATA_PATH, sep="\t")

    # Labels von Text → Zahlen umwandeln
    # ham = 0, spam = 1
    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Zeilen mit fehlenden Werten entfernen
    df = df.dropna(subset=["label", "text"])

    return df


def train():
    """
    Führt das vollständige Training durch.
    """

    # Daten laden
    df = load_data()

    # Features (Text) und Zielvariable (Label) trennen
    X = df["text"]
    y = df["label"]

    # Aufteilen in Trainings- und Testdaten
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,     # 20% Testdaten
        random_state=42,   # reproduzierbare Ergebnisse
        stratify=y         # gleiche Verteilung von Spam/Ham
    )

    # Pipeline:
    # 1. Text → TF-IDF Features
    # 2. Klassifikation → Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Modell trainieren
    model.fit(X_train, y_train)

    # Vorhersagen auf Testdaten
    predictions = model.predict(X_test)

    # Bewertung ausgeben
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nDetailbericht:")
    print(classification_report(y_test, predictions))

    # Modell speichern
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nModell gespeichert unter:")
    print(MODEL_PATH)


if __name__ == "__main__":
    train()