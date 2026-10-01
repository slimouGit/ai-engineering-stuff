"""
DECISION TREE
Ein Baum trifft Entscheidungen über Regeln.
Kann für Klassifikation und Regression verwendet werden.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.DataFrame({
    "preis": [1,2,3,4,1,2,4,3,2,1,4,3],
    "sicherheit": [3,3,2,1,2,3,1,3,2,3,2,1],
    "platz": [5,4,4,2,5,5,2,4,4,5,3,2],
    "klasse": ["gut","gut","gut","nicht_gut","gut","gut",
               "nicht_gut","gut","gut","gut","nicht_gut","nicht_gut"]
})

X = df[["preis","sicherheit","platz"]]
y = df["klasse"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# max_depth begrenzt die Baumtiefe.
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

new_car = pd.DataFrame([{"preis":2,"sicherheit":3,"platz":5}])
print("Vorhersage:", model.predict(new_car)[0])
