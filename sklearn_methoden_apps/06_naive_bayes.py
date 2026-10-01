"""
NAIVE BAYES
Klassischer probabilistischer Ansatz für Textklassifikation.
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

texte = [
    "Ich habe starke Kopfschmerzen.",
    "Mir ist schwindelig.",
    "Ich habe Bauchschmerzen.",
    "Ich nehme täglich Ibuprofen.",
    "Ich benutze ein Asthmaspray.",
    "Der Arzt gab mir Paracetamol.",
    "Ich habe keine Schmerzen.",
    "Ich nehme keine Medikamente.",
    "Ich habe keine Allergien.",
    "Seit gestern habe ich Fieber.",
    "Ich nehme morgens Metformin.",
    "Nein, ich habe keine Vorerkrankungen."
]
labels = ["symptom","symptom","symptom","medikament","medikament","medikament",
          "negation","negation","negation","symptom","medikament","negation"]

X_train, X_test, y_train, y_test = train_test_split(
    texte, labels, test_size=0.25, random_state=42, stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
