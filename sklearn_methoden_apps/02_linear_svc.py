"""
LINEAR SVC
Klassifikation von Texten.
Gut geeignet für hochdimensionale TF-IDF-Daten.
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

texte = [
    "Ich möchte einen Termin verschieben.",
    "Ich brauche eine neue Bescheinigung.",
    "Meine Adresse hat sich geändert.",
    "Ich hatte Kontakt zu einer bewaffneten Gruppe.",
    "Ich transportierte Material für eine militante Organisation.",
    "Ich nahm an Treffen einer gewaltbereiten Gruppe teil.",
    "Mein Name ist falsch geschrieben.",
    "Ich möchte den Bearbeitungsstand wissen.",
    "Ich gab Geld an Mitglieder der Gruppe weiter.",
    "Ich war bei mehreren Treffen der Organisation."
]
labels = ["nicht_relevant","nicht_relevant","nicht_relevant","relevant","relevant",
          "relevant","nicht_relevant","nicht_relevant","relevant","relevant"]

X_train, X_test, y_train, y_test = train_test_split(
    texte, labels, test_size=0.3, random_state=42, stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LinearSVC())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
print("Neue Vorhersage:", model.predict(
    ["Ich sollte Nachrichten für die Gruppe weitergeben."]
)[0])
