"""
Phase 3 – Confidence und Thresholds
===================================

REINE KONSOLEN-APP – kein Frontend, kein Streamlit.

Ziel:
    Verstehen, wie aus einer Modell-Wahrscheinlichkeit
    über einen Threshold eine finale Klassifikation wird.

Ablauf:
    Text
      -> TF-IDF
      -> Logistic Regression
      -> predict_proba()
      -> Confidence für "relevant"
      -> Threshold
      -> relevant / nicht_relevant

Start in PyCharm:
    Einfach diese Datei mit "Run" starten.

Benötigte Pakete:
    pip install pandas scikit-learn
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# 1. DATEN LADEN
# ============================================================

# Die CSV liegt im Unterordner "data".
DATA_PATH = Path(__file__).parent / "data" / "text_samples.csv"

df = pd.read_csv(DATA_PATH)

# X = Eingabetexte
X = df["text"]

# y = bekannte richtige Klasse / Ground Truth
y = df["label"]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

# 75 % Training
# 25 % Test
#
# random_state=42:
# gleiche Aufteilung bei jedem Start
#
# stratify=y:
# Verhältnis der Klassen bleibt ähnlich
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. MODELL
# ============================================================

# TF-IDF:
# Wandelt Text in Zahlen um.
#
# LogisticRegression:
# Lernt aus diesen Zahlen, welche Texte eher
# "relevant" oder "nicht_relevant" sind.
#
# Pipeline:
# Führt beide Schritte automatisch hintereinander aus.
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# ============================================================
# 4. MODELL TRAINIEREN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# 5. INDEX DER POSITIVEN KLASSE BESTIMMEN
# ============================================================

# predict_proba() liefert Wahrscheinlichkeiten für ALLE Klassen.
#
# Beispiel:
# ["nicht_relevant", "relevant"]
#
# Wir wollen gezielt die Wahrscheinlichkeit für "relevant".
classes = model.named_steps["classifier"].classes_

positive_class = "relevant"
positive_index = list(classes).index(positive_class)


# ============================================================
# 6. HILFSFUNKTIONEN
# ============================================================

def predict_with_threshold(text, threshold):
    """
    Gibt für einen Text zurück:
    - Wahrscheinlichkeit für 'relevant'
    - finale Klasse nach Anwendung des Thresholds
    """

    # predict_proba() erwartet mehrere Texte,
    # deshalb wird unser einzelner Text in eine Liste gepackt.
    probability = model.predict_proba([text])[0][positive_index]

    # Eigene Entscheidungsregel:
    # Ist die Wahrscheinlichkeit >= Threshold?
    if probability >= threshold:
        prediction = "relevant"
    else:
        prediction = "nicht_relevant"

    return probability, prediction


def evaluate_threshold(threshold):
    """
    Bewertet das Modell auf den Testdaten
    für einen bestimmten Threshold.
    """

    # Wahrscheinlichkeit für jeden Testtext berechnen
    probabilities = model.predict_proba(X_test)[:, positive_index]

    # Aus Wahrscheinlichkeiten Klassen machen
    predictions = [
        "relevant" if probability >= threshold else "nicht_relevant"
        for probability in probabilities
    ]

    # Precision:
    # Wenn wir "relevant" sagen, wie oft stimmt das?
    precision = precision_score(
        y_test,
        predictions,
        pos_label="relevant",
        zero_division=0
    )

    # Recall:
    # Von allen tatsächlich relevanten Fällen,
    # wie viele wurden gefunden?
    recall = recall_score(
        y_test,
        predictions,
        pos_label="relevant",
        zero_division=0
    )

    # F1:
    # Kombination aus Precision und Recall
    f1 = f1_score(
        y_test,
        predictions,
        pos_label="relevant",
        zero_division=0
    )

    # Confusion Matrix:
    # zeigt richtige und falsche Klassifikationen
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["nicht_relevant", "relevant"]
    )

    return precision, recall, f1, cm


# ============================================================
# 7. KONSOLE
# ============================================================

print("\n======================================")
print("PHASE 3 – CONFIDENCE UND THRESHOLDS")
print("======================================")

print("\nModell wurde trainiert.")
print("Trainingsdaten:", len(X_train))
print("Testdaten:", len(X_test))


# ============================================================
# 8. THRESHOLD ABFRAGEN
# ============================================================

# Standardwert 0.5
threshold_input = input(
    "\nThreshold eingeben (z.B. 0.5) [Enter = 0.5]: "
).strip()

if threshold_input == "":
    threshold = 0.5
else:
    threshold = float(threshold_input)

print(f"\nVerwendeter Threshold: {threshold:.2f}")


# ============================================================
# 9. MODELL BEWERTEN
# ============================================================

precision, recall, f1, cm = evaluate_threshold(threshold)

print("\n--- Modellbewertung ---")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1:        {f1:.2f}")

print("\n--- Confusion Matrix ---")
print("                  Vorhersage")
print("               nicht_rel.  relevant")
print(f"Tats. nicht_rel.     {cm[0][0]:>3}       {cm[0][1]:>3}")
print(f"Tats. relevant       {cm[1][0]:>3}       {cm[1][1]:>3}")


# ============================================================
# 10. MEHRERE THRESHOLDS VERGLEICHEN
# ============================================================

print("\n--- Vergleich verschiedener Thresholds ---")
print("Threshold | Precision | Recall | F1")
print("--------------------------------------")

for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    p, r, f, _ = evaluate_threshold(t)

    print(
        f"{t:>9.2f} | "
        f"{p:>9.2f} | "
        f"{r:>6.2f} | "
        f"{f:>4.2f}"
    )


# ============================================================
# 11. EIGENEN TEXT TESTEN
# ============================================================

print("\n--- Eigenen Text testen ---")

text = input(
    "Text eingeben [Enter = Beispieltext]: "
).strip()

if text == "":
    text = "Ich hatte mehrfach Kontakt zu Mitgliedern der Gruppe."

probability, prediction = predict_with_threshold(
    text,
    threshold
)

print("\nText:")
print(text)

print("\nConfidence für 'relevant':")
print(f"{probability:.2%}")

print("\nThreshold:")
print(f"{threshold:.2f}")

print("\nVorhersage:")
print(prediction)


# ============================================================
# 12. ERKLÄRUNG
# ============================================================

print("\n======================================")
print("MERKSATZ")
print("======================================")

print("""
Confidence:
    Vom Modell berechnete Wahrscheinlichkeit.

Threshold:
    Entscheidungsgrenze.

Beispiel:
    Confidence = 0.72

    Threshold = 0.50
    -> 0.72 >= 0.50
    -> relevant

    Threshold = 0.80
    -> 0.72 < 0.80
    -> nicht_relevant

Wichtig:
    Das Modell wurde dabei NICHT neu trainiert.
    Nur die Entscheidungsgrenze wurde verändert.
""")
