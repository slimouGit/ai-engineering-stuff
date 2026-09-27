import pandas as pd

# CSV-Datei laden
df = pd.read_csv("car.csv")

# Diese Spalten soll das Modell als Eingabe verwenden
features = [
    "buying",
    "maintenance",
    "doors",
    "persons",
    "lug_boot",
    "safety"
]

# X = Eingabedaten / Merkmale
X = df[features]

# y = Zielwert, den wir vorhersagen wollen
y = df["class"]


from sklearn.model_selection import train_test_split

# Daten aufteilen:
# 80 % zum Lernen
# 20 % zum Testen
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Unsere Daten enthalten Texte wie:
# "low", "high", "small", "big"
#
# LogisticRegression kann mit diesen Texten nicht direkt rechnen.
# OneHotEncoder wandelt sie deshalb in Zahlen um.
#
# Danach wird die LogisticRegression ausgeführt.
model = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ("classifier", LogisticRegression(max_iter=1000))
])


# Modell mit den Trainingsdaten lernen lassen
model.fit(X_train, y_train)

# Vorhersagen für die Testdaten machen
y_pred = model.predict(X_test)
print("Testdaten:\n", X_test.head())

# Die ersten 10 Vorhersagen anzeigen
print(y_pred[:10])

#Eigene vorhersagen machen
new_car = pd.DataFrame([{
    "buying": "low",
    "maintenance": "med",
    "doors": "4",
    "persons": "4",
    "lug_boot": "big",
    "safety": "high"
}])
# Vorhersage treffen
prediction = model.predict(new_car)

print(prediction)

class Car:
    def __init__(self, buying, maintenance, doors, persons, lug_boot, safety):
        self.buying = buying
        self.maintenance = maintenance
        self.doors = doors
        self.persons = persons
        self.lug_boot = lug_boot
        self.safety = safety


def predict_car(model, car):
    data = pd.DataFrame([{
        "buying": car.buying,
        "maintenance": car.maintenance,
        "doors": car.doors,
        "persons": car.persons,
        "lug_boot": car.lug_boot,
        "safety": car.safety
    }])

    return model.predict(data)

car = Car("low", "med", "4", "4", "small", "high")

print(predict_car(model, car))



