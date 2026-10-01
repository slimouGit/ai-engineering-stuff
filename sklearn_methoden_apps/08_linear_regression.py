"""
LINEAR REGRESSION
Sagt einen Zahlenwert voraus, keine Klasse.
Beispiel: Bearbeitungsdauer in Minuten.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.DataFrame({
    "komplexitaet":[1,1,2,2,3,3,4,4,5,5,2,3,4,1],
    "versuche":[1,2,1,2,2,3,3,4,4,5,3,4,2,1],
    "dauer_min":[20,30,40,55,75,95,120,145,170,210,60,100,115,18]
})

X = df[["komplexitaet","versuche"]]
y = df["dauer_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R²:", round(r2_score(y_test, y_pred), 2))

for actual, pred in zip(y_test, y_pred):
    print("Ist:", actual, "| Vorhersage:", round(pred,1))
