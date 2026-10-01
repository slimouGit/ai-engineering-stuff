"""
LOGISTIC REGRESSION
Klassifikation von Texten: relevant / nicht_relevant.
Zeigt zusätzlich predict_proba().
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

texte = [
    "Ich möchte meinen Termin ändern.",
    "Meine Adresse ist falsch.",
    "Ich brauche eine Bescheinigung.",
    "Ich hatte Kontakt zu einer bewaffneten Gruppe.",
    "Ich nahm an Treffen einer militanten Organisation teil.",
    "Ich sollte Material für eine bewaffnete Gruppe transportieren.",
    "Mein Reisepass ist abgelaufen.",
    "Ich möchte den Bearbeitungsstand wissen.",
    "Ich habe Geld für eine gewaltbereite Gruppe gesammelt.",
    "Ich stand mit Mitgliedern einer bewaffneten Einheit in Verbindung."
]
labels = ["nicht_relevant","nicht_relevant","nicht_relevant","relevant","relevant",
          "relevant","nicht_relevant","nicht_relevant","relevant","relevant"]

X_train, X_test, y_train, y_test = train_test_split(
    texte, labels, test_size=0.3, random_state=42, stratify=labels
)

# Text -> TF-IDF-Zahlen -> Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))

text = ["Ich hatte mehrfach Kontakt zu Mitgliedern der Gruppe."]
print("Vorhersage:", model.predict(text)[0])
print("Klassen:", model.classes_)
print("Wahrscheinlichkeiten:", model.predict_proba(text)[0])
