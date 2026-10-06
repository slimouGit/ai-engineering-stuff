import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data import testdaten


# ============================================================
# 1. DATEN LADEN
# ============================================================

df = pd.DataFrame(
    testdaten,
    columns=["text", "label"]
)

X = df["text"]      # Eingabetexte
y = df["label"]     # Ground Truth / richtige Klasse


# ============================================================
# 2. TRAIN / TEST AUFTEILEN
# ============================================================

# Das Testset bleibt bis zur finalen Bewertung unangetastet.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. PIPELINE
# ============================================================

# Text -> TF-IDF -> Klassifikationsmodell
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression())
])


# ============================================================
# 4. MODELLE + HYPERPARAMETER
# ============================================================

# GridSearchCV vergleicht:
# - Logistic Regression
# - Linear SVC
# - verschiedene C-Werte
# - einzelne Wörter vs. Wörter + Wortpaare

parameter_grid = [
    {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier": [
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        ],
        "classifier__C": [0.1, 1, 10]
    },
    {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier": [
            LinearSVC(
                random_state=42,
                max_iter=2000
            )
        ],
        "classifier__C": [0.1, 1, 10]
    },
    {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier": [
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        ],
        "classifier__max_depth": [5, 10, None]
    },
    {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier": [
            MultinomialNB()
        ],
    "classifier__alpha": [0.001, 0.01, 0.1, 1]
    }
]


# ============================================================
# 5. CROSS-VALIDATION + MODELLVERGLEICH
# ============================================================

# Nur die Trainingsdaten werden für die Modellauswahl verwendet.
grid_search = GridSearchCV(
    pipeline,
    parameter_grid,
    cv=5,
    scoring="f1_macro"
)

grid_search.fit(
    X_train,
    y_train
)

print("\n--- Bestes Setup ---")
print("Parameter:", grid_search.best_params_)
print("CV-F1:", round(grid_search.best_score_, 3))


# ============================================================
# 6. FINALES MODELL
# ============================================================

# GridSearchCV trainiert das beste Setup automatisch
# noch einmal auf allen Trainingsdaten.
best_model = grid_search.best_estimator_


# ============================================================
# 7. TESTSET VORHERSAGEN
# ============================================================

y_pred = best_model.predict(X_test)

vergleich = pd.DataFrame({
    "Text": X_test,
    "Tatsächlich": y_test,
    "Vorhersage": y_pred
})

print("\n--- Testdaten ---")
print(vergleich.to_string(index=False))


# ============================================================
# 8. MODELL VALIDIEREN
# ============================================================

print("\nAccuracy:")
print(round(accuracy_score(y_test, y_pred), 3))

print("\nPrecision / Recall / F1:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 9. EIGENE TEXTE VORHERSAGEN
# ============================================================

texte = [
    "Seit heute Morgen habe ich starke Schmerzen.",
    "Ich nehme jeden Abend Aspirin.",
    "Ich habe keine bekannten Allergien."
]

vorhersagen = best_model.predict(texte)

print("\n--- Eigene Vorhersagen ---")

for text, prediction in zip(texte, vorhersagen):
    print(text)
    print("→", prediction)


# ============================================================
# 10. CONFIDENCE
# ============================================================

# LogisticRegression hat predict_proba().
# LinearSVC standardmäßig nicht.

classifier = best_model.named_steps["classifier"]

if hasattr(classifier, "predict_proba"):

    wahrscheinlichkeiten = best_model.predict_proba(texte)
    confidence_threshold = 0.5

    print("\n--- Confidence ---")

    for text, probs in zip(texte, wahrscheinlichkeiten):

        index = probs.argmax()
        label = best_model.classes_[index]
        confidence = probs[index]

        print("\n", text)
        print("→", label if confidence >= confidence_threshold else "unsicher")
        print("Confidence:", round(confidence, 3))

else:

    print("\nDas beste Modell unterstützt kein predict_proba().")