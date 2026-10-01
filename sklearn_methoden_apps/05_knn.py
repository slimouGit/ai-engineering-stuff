"""
K-NEAREST NEIGHBORS (KNN)
Neue Daten werden anhand ihrer nächsten bekannten Nachbarn klassifiziert.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.DataFrame({
    "alter":[22,25,28,35,38,42,45,50,55,60,62,30],
    "einkommen":[25,28,30,45,48,55,60,65,70,72,75,35],
    "typ":["A","A","A","B","B","B","B","C","C","C","C","A"]
})

X = df[["alter","einkommen"]]
y = df["typ"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Skalierung ist bei Distanzverfahren wie KNN wichtig.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, zero_division=0))
