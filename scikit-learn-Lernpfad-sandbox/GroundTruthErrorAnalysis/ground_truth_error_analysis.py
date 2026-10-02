"""
Ground Truth und Error Analysis – Lernbeispiel
==============================================

Ziel:
- Verstehen, was Ground Truth ist.
- Vorhersagen mit der Ground Truth vergleichen.
- False Positives und False Negatives gezielt finden.
- Fehlerfälle einzeln untersuchen.

Benötigte Pakete:
    pip install pandas scikit-learn
"""
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. TRAININGSDATEN
# ============================================================

# Diese Beispiele kennt das Modell beim Training.
train_data = [
    ["Ich möchte meinen Termin ändern.", "nicht_relevant"],
    ["Meine Adresse hat sich geändert.", "nicht_relevant"],
    ["Ich brauche eine Kopie meiner Unterlagen.", "nicht_relevant"],
    ["Ich möchte den Bearbeitungsstand wissen.", "nicht_relevant"],
    ["Ich habe eine Frage zu meinem Schreiben.", "nicht_relevant"],
    ["Mein Reisepass ist abgelaufen.", "nicht_relevant"],

    ["Ich hatte Kontakt zu einer bewaffneten Gruppe.", "relevant"],
    ["Ich nahm an Treffen einer militanten Organisation teil.", "relevant"],
    ["Ich sollte Material für die Gruppe transportieren.", "relevant"],
    ["Ich habe Geld für die Organisation gesammelt.", "relevant"],
    ["Ich stand mit Mitgliedern der Gruppe in Verbindung.", "relevant"],
    ["Ich erhielt Aufträge von Mitgliedern einer bewaffneten Einheit.", "relevant"],
]

train_df = pd.DataFrame(
    train_data,
    columns=["text", "label"]
)

X_train = train_df["text"]
y_train = train_df["label"]


# ============================================================
# 2. TESTDATEN + GROUND TRUTH
# ============================================================

# Diese Texte wurden bewusst NICHT zum Training verwendet.
#
# Die Spalte "label" ist unsere GROUND TRUTH:
# die bekannte richtige Antwort, z. B. durch menschliche Annotation.
#
# Genau damit vergleichen wir später die Modellvorhersage.

test_data = [
    ["Ich hatte nur Kontakt zur Behörde wegen meines Termins.", "nicht_relevant"],
    ["Ich möchte Unterlagen über die Organisation anfordern.", "nicht_relevant"],
    ["Mein Termin bei der Behörde wurde verschoben.", "nicht_relevant"],
    ["Ich kenne Personen aus dieser Gruppe.", "relevant"],
    ["Ich habe früher Nachrichten für Mitglieder weitergegeben.", "relevant"],
    ["Ich sollte Dokumente an Mitglieder der Gruppe bringen.", "relevant"],
]

test_df = pd.DataFrame(
    test_data,
    columns=["text", "label"]
)

X_test = test_df["text"]
y_test = test_df["label"]


# ============================================================
# 3. MODELL ERSTELLEN UND TRAINIEREN
# ============================================================

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

model.fit(X_train, y_train)


# ============================================================
# 4. VORHERSAGEN + CONFIDENCE
# ============================================================

classes = model.named_steps["classifier"].classes_
positive_index = list(classes).index("relevant")

# Wahrscheinlichkeit für die Klasse "relevant"
probabilities = model.predict_proba(X_test)[:, positive_index]

# hart gecodeter Threshold
threshold = 0.5

# Vorhersage selbst aus der Wahrscheinlichkeit bauen
y_pred = [
    "relevant" if prob >= threshold else "nicht_relevant"
    for prob in probabilities
]


# ============================================================
# 5. ERGEBNIS-TABELLE BAUEN
# ============================================================

result = test_df.copy()

result["vorhersage"] = y_pred
result["confidence_relevant"] = probabilities.round(3)

# Stimmt Vorhersage mit Ground Truth überein?
result["korrekt"] = (
    result["label"] == result["vorhersage"]
)


# ============================================================
# 6. FEHLERTYP BESTIMMEN
# ============================================================

def error_type(row):
    """
    TP = tatsächlich relevant, als relevant erkannt
    TN = tatsächlich nicht relevant, als nicht relevant erkannt
    FP = tatsächlich nicht relevant, aber als relevant erkannt
    FN = tatsächlich relevant, aber als nicht relevant erkannt
    """

    if row["label"] == "relevant" and row["vorhersage"] == "relevant":
        return "TP"

    if row["label"] == "nicht_relevant" and row["vorhersage"] == "nicht_relevant":
        return "TN"

    if row["label"] == "nicht_relevant" and row["vorhersage"] == "relevant":
        return "FP"

    if row["label"] == "relevant" and row["vorhersage"] == "nicht_relevant":
        return "FN"


result["fehlertyp"] = result.apply(
    error_type,
    axis=1
)


# ============================================================
# 7. GESAMTERGEBNIS AUSGEBEN
# ============================================================

print("\n======================================")
print("GROUND TRUTH UND ERROR ANALYSIS")
print("======================================")

print("\n--- Alle Testfälle ---")

print(
    result[
        [
            "text",
            "label",
            "vorhersage",
            "confidence_relevant",
            "fehlertyp"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. MODELLBEWERTUNG
# ============================================================

print("\n--- Classification Report ---")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


print("\n--- Confusion Matrix ---")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["nicht_relevant", "relevant"]
)

print("                  Vorhersage")
print("               nicht_rel.  relevant")
print(f"Tats. nicht_rel.     {cm[0][0]:>3}       {cm[0][1]:>3}")
print(f"Tats. relevant       {cm[1][0]:>3}       {cm[1][1]:>3}")


# ============================================================
# 9. FALSE POSITIVES ANALYSIEREN
# ============================================================

false_positives = result[
    result["fehlertyp"] == "FP"
]

print("\n--- FALSE POSITIVES ---")

if false_positives.empty:
    print("Keine False Positives vorhanden.")
else:
    for _, row in false_positives.iterrows():

        print("\nText:")
        print(row["text"])

        print("Ground Truth:")
        print(row["label"])

        print("Modell:")
        print(row["vorhersage"])

        print("Confidence relevant:")
        print(row["confidence_relevant"])

        print(
            "Analyse-Idee: Welche Wörter könnten das Modell "
            "fälschlich in Richtung 'relevant' gelenkt haben?"
        )


# ============================================================
# 10. FALSE NEGATIVES ANALYSIEREN
# ============================================================

false_negatives = result[
    result["fehlertyp"] == "FN"
]

print("\n--- FALSE NEGATIVES ---")

if false_negatives.empty:
    print("Keine False Negatives vorhanden.")
else:
    for _, row in false_negatives.iterrows():

        print("\nText:")
        print(row["text"])

        print("Ground Truth:")
        print(row["label"])

        print("Modell:")
        print(row["vorhersage"])

        print("Confidence relevant:")
        print(row["confidence_relevant"])

        print(
            "Analyse-Idee: Welche relevanten Formulierungen "
            "kennt das Modell vielleicht noch nicht ausreichend?"
        )


# ============================================================
# 11. NUR ALLE FEHLER ANZEIGEN
# ============================================================

errors = result[
    result["korrekt"] == False
]

print("\n--- ALLE FEHLERFÄLLE ---")

if errors.empty:
    print("Keine Fehler vorhanden.")
else:
    print(
        errors[
            [
                "text",
                "label",
                "vorhersage",
                "confidence_relevant",
                "fehlertyp"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 12. MERKSATZ
# ============================================================

print("""
======================================
MERKSATZ
======================================

Ground Truth:
    Die bekannte richtige Antwort.

False Positive:
    Ground Truth = nicht_relevant
    Modell       = relevant

False Negative:
    Ground Truth = relevant
    Modell       = nicht_relevant

Error Analysis:
    Nicht nur zählen, WIE VIELE Fehler auftreten,
    sondern die einzelnen Fehlerfälle anschauen und fragen:

    - Welche Formulierungen führen zum Fehler?
    - Fehlen ähnliche Beispiele im Training?
    - Ist die Ground Truth eindeutig?
    - Ist der Threshold ungünstig?
    - Gibt es systematische Fehlermuster?

Genau daraus entstehen anschließend Verbesserungen
für Daten, Labels, Features, Modell oder Threshold.
""")

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model.joblib"
joblib.dump(model, MODEL_FILE)
print(f"\nModell gespeichert unter: {MODEL_FILE}")

