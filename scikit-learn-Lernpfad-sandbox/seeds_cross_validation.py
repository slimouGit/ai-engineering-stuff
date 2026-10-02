import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score


# ============================================================
# 1. DATEN
# ============================================================

daten = [
    ["Ich habe starke Kopfschmerzen.", "symptom"],
    ["Mir ist schwindelig.", "symptom"],
    ["Ich habe Bauchschmerzen.", "symptom"],
    ["Ich habe seit gestern Fieber.", "symptom"],
    ["Mein Rücken tut weh.", "symptom"],
    ["Ich habe Halsschmerzen.", "symptom"],

    ["Ich nehme täglich Ibuprofen.", "medikament"],
    ["Ich nehme morgens Metformin.", "medikament"],
    ["Ich benutze ein Asthmaspray.", "medikament"],
    ["Ich nehme Paracetamol.", "medikament"],
    ["Ich nehme Blutdrucktabletten.", "medikament"],
    ["Ich benutze regelmäßig Nasenspray.", "medikament"],

    ["Ich habe keine Schmerzen.", "negation"],
    ["Ich nehme keine Medikamente.", "negation"],
    ["Ich habe keine Allergien.", "negation"],
    ["Ich rauche nicht.", "negation"],
    ["Ich habe kein Fieber.", "negation"],
    ["Ich habe keine Vorerkrankungen.", "negation"],
]

df = pd.DataFrame(
    daten,
    columns=["text", "label"]
)

X = df["text"]
y = df["label"]


# ============================================================
# 2. MODELL
# ============================================================

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# ============================================================
# 3. UNTERSCHIEDLICHE SEEDS
# ============================================================

print("\n--- Unterschiedliche Seeds ---")

for seed in [1, 2, 3, 4, 5]:

    # Durch den anderen Seed entstehen andere Train/Test-Daten.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.33,
        random_state=seed,
        stratify=y
    )

    # Modell für diesen Split neu trainieren
    model.fit(X_train, y_train)

    # Vorhersage
    y_pred = model.predict(X_test)

    # F1 berechnen
    f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    print(
        "Seed:",
        seed,
        "| F1:",
        round(f1, 3)
    )


# ============================================================
# 4. CROSS-VALIDATION
# ============================================================

print("\n--- 5-Fold Cross-Validation ---")

# Der Datensatz wird in 5 Teile aufgeteilt.
#
# Durchlauf 1:
# 4 Teile Training
# 1 Teil Test
#
# Durchlauf 2:
# anderer Teil ist Test
#
# usw.

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1_macro"
)

print("F1 pro Fold:", scores)

print(
    "Durchschnittlicher F1:",
    round(scores.mean(), 3)
)

print(
    "Standardabweichung:",
    round(scores.std(), 3)
)