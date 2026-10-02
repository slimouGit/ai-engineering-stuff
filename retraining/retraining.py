import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, f1_score


# ============================================================
# 1. TESTDATEN
# ============================================================

# WICHTIG:
# Diese Testdaten bleiben für V1 und V2 identisch.
# Sonst wäre der Vergleich unfair.

test_data = [
    ["Ich möchte meinen Termin verschieben.", "nicht_relevant"],
    ["Meine Adresse hat sich geändert.", "nicht_relevant"],

    ["Ich musste Nachrichten zwischen Mitgliedern weitergeben.", "relevant"],
    ["Ich sollte Dokumente für die Gruppe transportieren.", "relevant"],
    ["Ich übermittelte Informationen an mehrere Mitglieder.", "relevant"],
]

test_df = pd.DataFrame(
    test_data,
    columns=["text", "label"]
)

X_test = test_df["text"]
y_test = test_df["label"]


# ============================================================
# 2. TRAININGSDATEN FÜR MODELL V1
# ============================================================

# V1 kennt nur wenige Formulierungen.

train_v1 = [
    ["Ich möchte einen Termin ändern.", "nicht_relevant"],
    ["Ich brauche eine Kopie meiner Unterlagen.", "nicht_relevant"],
    ["Ich möchte den Bearbeitungsstand wissen.", "nicht_relevant"],

    ["Ich hatte Kontakt zu einer bewaffneten Gruppe.", "relevant"],
    ["Ich nahm an Treffen der Organisation teil.", "relevant"],
    ["Ich habe Geld für die Gruppe gesammelt.", "relevant"],
]

df_v1 = pd.DataFrame(
    train_v1,
    columns=["text", "label"]
)


# ============================================================
# 3. MODELL-FUNKTION
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
# 4. MODELL V1 TRAINIEREN
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
# 5. FEHLER VON V1 ANALYSIEREN
# ============================================================

result_v1 = test_df.copy()

result_v1["prediction_v1"] = pred_v1

errors_v1 = result_v1[
    result_v1["label"] != result_v1["prediction_v1"]
]

print("\n--- Fehler von V1 ---")
print(errors_v1)


# ============================================================
# 6. NEUE GROUND-TRUTH-DATEN
# ============================================================

# Angenommen, bei der Error Analysis fällt auf:
#
# Das Modell kennt relevante Formulierungen wie
# "Nachrichten weitergeben" oder
# "Informationen übermitteln"
# noch nicht gut genug.
#
# Fachlich geprüfte neue Beispiele werden ergänzt.

new_ground_truth = [
    ["Ich gab Nachrichten an andere Mitglieder weiter.", "relevant"],
    ["Ich übermittelte Informationen zwischen mehreren Personen.", "relevant"],
    ["Ich brachte Dokumente zu Mitgliedern der Gruppe.", "relevant"],
    ["Ich war für die Weitergabe von Nachrichten verantwortlich.", "relevant"],

    # Auch neue Negativbeispiele ergänzen,
    # damit das Modell nicht nur eine Seite lernt.
    ["Ich möchte Informationen über meinen Termin erhalten.", "nicht_relevant"],
    ["Ich habe Unterlagen an die Behörde geschickt.", "nicht_relevant"],
]

new_df = pd.DataFrame(
    new_ground_truth,
    columns=["text", "label"]
)


# ============================================================
# 7. TRAININGSDATEN FÜR V2
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
# 8. MODELL V2 TRAINIEREN
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
# 9. V1 UND V2 VERGLEICHEN
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
# 10. INTERPRETATION
# ============================================================

if f1_v2 > f1_v1:
    print("\nV2 ist auf diesem Testset besser als V1.")

elif f1_v2 == f1_v1:
    print("\nV1 und V2 sind auf diesem Testset gleich gut.")

else:
    print("\nV2 ist auf diesem Testset schlechter als V1.")