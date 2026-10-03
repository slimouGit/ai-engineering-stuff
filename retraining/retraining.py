import json
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, f1_score


# ============================================================
# 1. DATEN AUS DATEI LADEN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "retraining_data.json"

with DATA_FILE.open(encoding="utf-8") as file:
    data = json.load(file)


def section_to_dataframe(section_name):
    """Liest einen Datensatz-Abschnitt aus der JSON-Datei und baut ein DataFrame."""
    rows = data[section_name]
    return pd.DataFrame(rows, columns=["text", "label"])


# ============================================================
# 2. TESTDATEN
# ============================================================

# WICHTIG:
# Diese Testdaten bleiben für V1 und V2 identisch.
# Sonst wäre der Vergleich unfair.

test_df = section_to_dataframe("test_data")

X_test = test_df["text"]
y_test = test_df["label"]


# ============================================================
# 3. TRAININGSDATEN FÜR MODELL V1
# ============================================================

# V1 kennt nur wenige Formulierungen.

df_v1 = section_to_dataframe("train_v1")


# ============================================================
# 4. MODELL-FUNKTION
# ============================================================

def create_model():

    return Pipeline([
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
# 5. MODELL V1 TRAINIEREN
# ============================================================

model_v1 = create_model()

model_v1.fit(
    df_v1["text"],
    df_v1["label"]
)

pred_v1 = model_v1.predict(X_test)


print("\n==============================")
print("MODELL V1")
print("==============================")

print(
    classification_report(
        y_test,
        pred_v1,
        zero_division=0
    )
)

f1_v1 = f1_score(
    y_test,
    pred_v1,
    pos_label="relevant"
)

print("F1 V1:", round(f1_v1, 3))


# ============================================================
# 6. FEHLER VON V1 ANALYSIEREN
# ============================================================

result_v1 = test_df.copy()

result_v1["prediction_v1"] = pred_v1

errors_v1 = result_v1[
    result_v1["label"] != result_v1["prediction_v1"]
]

print("\n--- Fehler von V1 ---")
print(errors_v1)


# ============================================================
# 7. NEUE GROUND-TRUTH-DATEN
# ============================================================

# Angenommen, bei der Error Analysis fällt auf:
#
# Das Modell kennt relevante Formulierungen wie
# "Nachrichten weitergeben" oder
# "Informationen übermitteln"
# noch nicht gut genug.
#
# Fachlich geprüfte neue Beispiele werden ergänzt.

new_df = section_to_dataframe("new_ground_truth")


# ============================================================
# 8. TRAININGSDATEN FÜR V2
# ============================================================

# Alte Daten + neue geprüfte Daten

df_v2 = pd.concat(
    [
        df_v1,
        new_df
    ],
    ignore_index=True
)


# ============================================================
# 9. MODELL V2 TRAINIEREN
# ============================================================

model_v2 = create_model()

model_v2.fit(
    df_v2["text"],
    df_v2["label"]
)

pred_v2 = model_v2.predict(X_test)



print("\n==============================")
print("MODELL V2")
print("==============================")

print(
    classification_report(
        y_test,
        pred_v2,
        zero_division=0
    )
)

f1_v2 = f1_score(
    y_test,
    pred_v2,
    pos_label="relevant"
)

print("F1 V2:", round(f1_v2, 3))


# ============================================================
# 10. V1 UND V2 VERGLEICHEN
# ============================================================

comparison = test_df.copy()

comparison["V1"] = pred_v1
comparison["V2"] = pred_v2

print("\n--- Direkter Vergleich ---")
print(comparison)


print("\n--- F1 Vergleich ---")
print("V1:", round(f1_v1, 3))
print("V2:", round(f1_v2, 3))


# ============================================================
# 11. INTERPRETATION
# ============================================================

if f1_v2 > f1_v1:
    print("\nV2 ist auf diesem Testset besser als V1.")

elif f1_v2 == f1_v1:
    print("\nV1 und V2 sind auf diesem Testset gleich gut.")

else:
    print("\nV2 ist auf diesem Testset schlechter als V1.")


print("PREDICT_PROBA MODEL1")
print(model_v1.named_steps["classifier"].classes_)
print(model_v1.predict_proba(X_test))

print("PREDICT_PROBA MODEL2")
print(model_v2.named_steps["classifier"].classes_)
print(model_v2.predict_proba(X_test))