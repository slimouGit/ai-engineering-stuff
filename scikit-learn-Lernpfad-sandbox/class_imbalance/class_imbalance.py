import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. DATEN MIT STARKER KLASSEN-UNWUCHT
# ============================================================

# 95 Fälle sind nicht relevant
# 5 Fälle sind relevant
#
# Das ist Class Imbalance:
# Eine Klasse kommt viel häufiger vor als die andere.

daten = []

for i in range(95):
    daten.append([
        i % 10,          # feature_1
        (i * 2) % 7,    # feature_2
        "nicht_relevant"
    ])

# Seltene positive Klasse
positive_faelle = [
    [8, 6, "relevant"],
    [9, 5, "relevant"],
    [8, 5, "relevant"],
    [9, 6, "relevant"],
    [7, 6, "relevant"],
]

daten.extend(positive_faelle)

df = pd.DataFrame(
    daten,
    columns=["feature_1", "feature_2", "label"]
)

print("--- Klassenverteilung ---")
print(df["label"].value_counts())
print("--- Daten ---")
print(df)


# ============================================================
# 2. X UND y
# ============================================================

X = df[["feature_1", "feature_2"]]
y = df["label"]


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

# stratify=y ist bei unausgeglichenen Klassen besonders wichtig.
# Dadurch bleibt das Klassenverhältnis in Train und Test ähnlich.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. NORMALES MODELL
# ============================================================

model_normal = LogisticRegression(
    max_iter=1000
)

model_normal.fit(
    X_train,
    y_train
)

y_pred_normal = model_normal.predict(X_test)


print("\n===================================")
print("OHNE class_weight='balanced'")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred_normal,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_normal,
        labels=["nicht_relevant", "relevant"]
    )
)


# ============================================================
# 5. MODELL MIT CLASS WEIGHT
# ============================================================

# class_weight="balanced":
# Die seltene Klasse bekommt beim Training automatisch
# ein höheres Gewicht.
#
# Fehler bei "relevant" zählen dadurch stärker.

model_balanced = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model_balanced.fit(
    X_train,
    y_train
)

y_pred_balanced = model_balanced.predict(X_test)


print("\n===================================")
print("MIT class_weight='balanced'")
print("===================================")

print(
    classification_report(
        y_test,
        y_pred_balanced,
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred_balanced,
        labels=["nicht_relevant", "relevant"]
    )
)

print("\n--- Vorhersage ---")
print(model_balanced.predict([[0, 0]]))  # relevant

print("\n--- Gewichte ohne class_weight='balanced' ---")
print("Klassen:", model_normal.classes_)
print("Koeffizienten:", model_normal.coef_)
print("Intercept:", model_normal.intercept_)

print("\n--- Gewichte mit class_weight='balanced' ---")
print("Klassen:", model_balanced.classes_)
print("Koeffizienten:", model_balanced.coef_)
print("Intercept:", model_balanced.intercept_)