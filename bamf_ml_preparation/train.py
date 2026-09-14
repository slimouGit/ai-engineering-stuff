from pathlib import Path
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "documents.csv"
MODEL_FILE = BASE_DIR / "model.joblib"

if not DATA_FILE.exists():
    print(f"Data file not found: {DATA_FILE}")
    print("Place your CSV at that path or update the path in `train.py`.")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)

# Detect and fix swapped columns (short keys in text, long sentences in label)
if df["text"].nunique() < df["label"].nunique():
    df = df.rename(columns={"text": "_tmp", "label": "text"})
    df["label"] = df["_tmp"]
    df = df.drop(columns=["_tmp"])

print("Datensatz:")
print(df.head())
print("\nKlassenverteilung:")
print(df["label"].value_counts())

# Group rare labels so stratified split works
min_samples_per_class = 2
label_counts = df["label"].value_counts()
df["label_grouped"] = df["label"].where(
    df["label"].map(label_counts) >= min_samples_per_class, "other"
)

print("\nGruppierte Klassenverteilung:")
print(df["label_grouped"].value_counts())

# If grouping results in only one group, skip stratify
stratify_arg = df["label_grouped"] if df["label_grouped"].nunique() > 1 else None

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.25,
    random_state=42,
    shuffle=True,
    stratify=stratify_arg,
)

model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

labels_for_matrix = sorted(set(y_test) | set(predictions))
if labels_for_matrix:
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=labels_for_matrix))
else:
    print("\nConfusion Matrix: no labels in test/predictions")

joblib.dump(model, MODEL_FILE)
print(f"\nModell gespeichert unter: {MODEL_FILE}")