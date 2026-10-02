"""
Phase 1 – Klassisches Machine Learning mit scikit-learn
=======================================================

Ziel:
    Mit den Daten aus "car.csv" soll vorhergesagt werden,
    in welche Bewertungsklasse ein Auto fällt.

Zielklassen:
    unacc = unacceptable / nicht akzeptabel
    acc   = acceptable / akzeptabel
    good  = gut
    vgood = sehr gut

Verwendete Schritte:
    1. CSV laden
    2. Eingabedaten X und Zielvariable y festlegen
    3. Trainings- und Testdaten aufteilen
    4. Kategoriale Werte mit OneHotEncoder vorbereiten
    5. LogisticRegression trainieren
    6. Vorhersagen machen
    7. Precision, Recall und F1 auswerten
    8. Eigene Autos vorhersagen

Benötigte Pakete:
    pip install pandas scikit-learn
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


# ============================================================
# 1. DATEN LADEN
# ============================================================

# Die CSV-Datei muss im gleichen Ordner wie dieses Python-Skript liegen.
df = pd.read_csv("car.csv")

# Einen ersten Blick auf die Daten werfen.
print("\n--- Erste 5 Zeilen ---")
print(df.head())

# Größe des Datensatzes anzeigen:
# 1728 Zeilen = 1728 Autos
# 7 Spalten = 6 Merkmale + 1 Zielspalte
print("\n--- Größe des Datensatzes ---")
print(df.shape)

# Prüfen, wie oft jede Zielklasse vorkommt.
# Das ist wichtig, weil die Klassen unterschiedlich häufig sind.
print("\n--- Verteilung der Zielklassen ---")
print(df["class"].value_counts())


# ============================================================
# 2. X UND y DEFINIEREN
# ============================================================

# "features" enthält alle Spalten, die das Modell zur Vorhersage verwenden darf.
features = [
    "buying",       # Kaufpreis: low, med, high, vhigh
    "maintenance",  # Wartungskosten: low, med, high, vhigh
    "doors",        # Anzahl Türen: 2, 3, 4, 5more
    "persons",      # Anzahl Personen: 2, 4, more
    "lug_boot",     # Kofferraumgröße: small, med, big
    "safety"        # Sicherheitsniveau: low, med, high
]

# X = Eingabedaten / Features
# Das Modell bekommt diese Werte später als Eingabe.
X = df[features]

# y = Zielvariable / Target
# Diesen Wert soll das Modell lernen und später vorhersagen.
y = df["class"]

print("\n--- Beispiel für X ---")
print(X.head())

print("\n--- Beispiel für y ---")
print(y.head())


# ============================================================
# 3. TRAININGS- UND TESTDATEN ERSTELLEN
# ============================================================

# Wir teilen die Daten in zwei Teile:
#
# X_train / y_train:
#     80 % der Daten.
#     Mit diesen Daten lernt das Modell.
#
# X_test / y_test:
#     20 % der Daten.
#     Diese Daten sieht das Modell beim Training nicht.
#     Sie werden später verwendet, um die Qualität zu prüfen.
#
# test_size=0.2:
#     20 % Testdaten.
#
# random_state=42:
#     Sorgt dafür, dass bei jedem Programmlauf dieselbe
#     zufällige Aufteilung verwendet wird.
#
# stratify=y:
#     Sorgt dafür, dass die Verteilung von
#     unacc / acc / good / vgood in Train und Test
#     ungefähr gleich bleibt.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n--- Anzahl Trainings- und Testdaten ---")
print("Trainingsdaten:", len(X_train))
print("Testdaten:", len(X_test))


# ============================================================
# 4. PREPROCESSING + MODELL
# ============================================================

# Problem:
# Unsere Eingabedaten bestehen größtenteils aus Textwerten:
#
#     low
#     high
#     small
#     big
#     vhigh
#     5more
#
# LogisticRegression kann mit solchen Texten nicht direkt rechnen.
#
# Deshalb verwenden wir OneHotEncoder.
#
# Vereinfacht wird zum Beispiel:
#
#     safety = low / med / high
#
# zu mehreren numerischen Spalten:
#
#     safety_low    safety_med    safety_high
#          0             0              1
#
#
# Pipeline:
# Eine Pipeline verbindet mehrere Verarbeitungsschritte.
#
# Bei uns:
#
#     Rohdaten
#        ↓
#     OneHotEncoder
#        ↓
#     LogisticRegression
#
# Vorteil:
# Wir müssen den Encoder nicht jedes Mal manuell aufrufen.
# Wenn wir später model.predict(...) verwenden,
# führt die Pipeline automatisch zuerst das Encoding aus.

model = Pipeline([
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# ============================================================
# 5. MODELL TRAINIEREN
# ============================================================

# fit() bedeutet:
# Das Modell lernt aus den Trainingsdaten.
#
# X_train = Eigenschaften der Autos
# y_train = richtige Bewertungsklassen
#
# Beispiel:
#
#     buying = low
#     maintenance = med
#     safety = high
#     ...
#
#     richtige Klasse = good
#
# Aus vielen solchen Beispielen versucht die LogisticRegression,
# Zusammenhänge zwischen den Merkmalen und den Klassen zu lernen.

model.fit(X_train, y_train)

print("\n--- Training abgeschlossen ---")


# ============================================================
# 6. VORHERSAGEN FÜR DIE TESTDATEN
# ============================================================

# predict() bekommt nur X_test.
#
# Das Modell kennt dabei NICHT y_test.
# Es soll selbst entscheiden, zu welcher Klasse jedes Auto gehört.

y_pred = model.predict(X_test)

print("\n--- Erste 10 Vorhersagen ---")
print(y_pred[:10])


# ============================================================
# 7. VORHERSAGE MIT DER RICHTIGEN ANTWORT VERGLEICHEN
# ============================================================

# y_test enthält die tatsächlichen Klassen.
# y_pred enthält die vom Modell vorhergesagten Klassen.
#
# Dadurch können wir direkt sehen, wo das Modell richtig
# oder falsch lag.

vergleich = pd.DataFrame({
    "Tatsächlich": y_test,
    "Vorhersage": y_pred
})

print("\n--- Tatsächliche Werte vs. Vorhersage ---")
print(vergleich.head(15))


# ============================================================
# 8. MODELL EVALUIEREN
# ============================================================

# Accuracy:
# Anteil aller Vorhersagen, die richtig waren.
#
# Beispiel:
# 90 von 100 Vorhersagen richtig
# -> Accuracy = 0.90 = 90 %
#
# Achtung:
# Unser Datensatz hat unterschiedlich häufige Klassen.
# Deshalb reicht Accuracy alleine nicht aus.

accuracy = accuracy_score(y_test, y_pred)

print("\n--- Accuracy ---")
print(f"{accuracy:.4f}")


# classification_report zeigt für jede Klasse:
#
# Precision:
#     Wenn das Modell z.B. "good" vorhersagt:
#     Wie oft ist diese Vorhersage tatsächlich "good"?
#
# Recall:
#     Von allen tatsächlich vorhandenen "good"-Autos:
#     Wie viele hat das Modell als "good" erkannt?
#
# F1-Score:
#     Kombiniert Precision und Recall zu einem Wert.
#
# Support:
#     Anzahl der tatsächlich vorhandenen Beispiele dieser Klasse.

print("\n--- Precision / Recall / F1 ---")
print(classification_report(y_test, y_pred))


# ============================================================
# 9. EIGENE VORHERSAGE MIT EINEM DATAFRAME
# ============================================================

# Jetzt erstellen wir ein komplett neues Auto.
# Dieses Auto stammt nicht aus X_test.
#
# Wichtig:
# Die Spaltennamen müssen genauso heißen wie beim Training.

new_car = pd.DataFrame([{
    "buying": "low",
    "maintenance": "med",
    "doors": "4",
    "persons": "4",
    "lug_boot": "big",
    "safety": "high"
}])

prediction = model.predict(new_car)

print("\n--- Vorhersage für eigenes Auto ---")
print(new_car)
print("Vorhergesagte Klasse:", prediction[0])


# ============================================================
# 10. OPTIONAL: EIGENE CAR-KLASSE
# ============================================================

class Car:
    """
    Ein einfaches Python-Objekt für ein Auto.

    Diese Klasse gehört nicht zu scikit-learn.
    Sie dient nur dazu, eigene Autos bequem zu erstellen.

    Beispiel:
        car = Car(
            "low",
            "med",
            "4",
            "4",
            "big",
            "high"
        )
    """

    def __init__(
        self,
        buying,
        maintenance,
        doors,
        persons,
        lug_boot,
        safety
    ):
        self.buying = buying
        self.maintenance = maintenance
        self.doors = doors
        self.persons = persons
        self.lug_boot = lug_boot
        self.safety = safety

    def to_dataframe(self):
        """
        Wandelt das Car-Objekt in ein pandas DataFrame um.

        Warum ist das nötig?

        model.predict() kann nicht direkt mit unserem selbst
        erstellten Car-Objekt arbeiten.

        Das Modell wurde mit einem DataFrame trainiert.
        Deshalb geben wir ihm für die Vorhersage ebenfalls
        einen DataFrame mit denselben Spalten.
        """

        return pd.DataFrame([{
            "buying": self.buying,
            "maintenance": self.maintenance,
            "doors": self.doors,
            "persons": self.persons,
            "lug_boot": self.lug_boot,
            "safety": self.safety
        }])


# Eigenes Auto als Python-Objekt erstellen.
car1 = Car(
    "low",
    "med",
    "4",
    "4",
    "big",
    "high"
)

# Das Objekt wird vor der Vorhersage in ein DataFrame umgewandelt.
car1_prediction = model.predict(car1.to_dataframe())

print("\n--- Vorhersage für Car-Objekt ---")
print("Auto 1:", car1.to_dataframe().to_dict(orient="records")[0])
print("Vorhergesagte Klasse:", car1_prediction[0])


# Zweites Beispiel.
car2 = Car(
    "high",
    "med",
    "2",
    "4",
    "small",
    "high"
)

car2_prediction = model.predict(car2.to_dataframe())

print("\n--- Zweites Car-Objekt ---")
print("Auto 2:", car2.to_dataframe().to_dict(orient="records")[0])
print("Vorhergesagte Klasse:", car2_prediction[0])


# ============================================================
# ZUSAMMENFASSUNG
# ============================================================

# Der komplette ML-Ablauf dieses Beispiels:
#
# car.csv
#    ↓
# pandas DataFrame
#    ↓
# X = Features
# y = Zielklasse
#    ↓
# train_test_split()
#    ↓
# X_train / y_train
# X_test  / y_test
#    ↓
# OneHotEncoder
#    ↓
# LogisticRegression
#    ↓
# model.fit()
#    ↓
# model.predict()
#    ↓
# Vergleich mit Ground Truth y_test
#    ↓
# Accuracy
# Precision
# Recall
# F1
#
# Damit sind die wichtigsten Grundlagen von Phase 1 abgedeckt.
