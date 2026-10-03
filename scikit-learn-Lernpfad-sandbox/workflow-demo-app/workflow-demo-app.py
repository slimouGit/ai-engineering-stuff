import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import testdaten


# ============================================================
# Aus der Liste ein DataFrame erstellen.
# ============================================================
df = pd.DataFrame(
    testdaten,
    columns=["text", "label"]
)
print("\n--- Datensatz ---")
print(df)

# ============================================================
# X UND y DEFINIEREN
# ============================================================

# X = Texte, aus denen das Modell lernen soll.
X = df["text"]

# y = richtige Klasse zu jedem Text.
y = df["label"]

# ============================================================
# TF-IDF + LOGISTIC REGRESSION
# ============================================================

# Ein ML-Modell kann mit normalen Sätzen nicht direkt rechnen.
#
# TF-IDF wandelt Wörter in Zahlen um.
#
# Vereinfacht:
#
# "Ich habe Kopfschmerzen"
#
# wird zu einem Zahlenvektor wie:
#
# [0.0, 0.4, 0.8, 0.0, ...]
#
# Wörter, die für bestimmte Texte besonders charakteristisch
# sind, bekommen dabei ein höheres Gewicht.

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer()
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# ============================================================
# TRAIN UND TEST AUFTEILEN
# ============================================================

# 70 % Training
# 30 % Test
#
# stratify=y sorgt dafür, dass alle Klassen ungefähr
# gleichmäßig auf Train und Test verteilt bleiben.
for seed in [1, 2, 3, 4, 5]:
    X_train, X_test, y_train, y_test = train_test_split(
    X,
            y,
        test_size=0.3,
        random_state=seed,
        stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SEED:", seed)
    print(classification_report(y_test, y_pred))
