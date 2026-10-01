"""
GRADIENT BOOSTING
Viele kleine Modelle werden nacheinander gebaut.
Jedes neue Modell versucht Fehler der bisherigen Modelle zu verbessern.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

df = pd.DataFrame({
    "prioritaet":[1,1,2,2,3,3,4,4,2,3,4,1,2,3,4,1],
    "versuche":[1,2,1,3,2,4,3,5,2,3,4,1,2,4,5,1],
    "dauer":[30,45,70,120,180,240,300,420,90,210,360,25,80,260,400,35],
    "sla_verletzt":[0,0,0,0,0,1,1,1,0,1,1,0,0,1,1,0]
})

X = df[["prioritaet","versuche","dauer"]]
y = df["sla_verletzt"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
