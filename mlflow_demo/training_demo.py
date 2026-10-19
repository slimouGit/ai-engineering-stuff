import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. MLFLOW KONFIGURIEREN
# ============================================================

# MLflow läuft bei dir lokal auf Port 5000.
# Alle Runs, Parameter und Metriken werden dorthin geschickt.
mlflow.set_tracking_uri("http://localhost:5000")

# Name des Experiments in der MLflow-Oberfläche.
mlflow.set_experiment("MLflow Model Training Demo")


# ============================================================
# 2. DATENSATZ LADEN
# ============================================================

# Iris ist ein kleiner Beispiel-Datensatz aus sklearn.
# Aufgabe:
# Anhand von 4 Messwerten soll eine von 3 Blumenarten erkannt werden.
data = load_iris()

X = data.data
y = data.target


# ============================================================
# 3. TRAININGS- UND TESTDATEN ERZEUGEN
# ============================================================

# 70 % der Daten werden fürs Training verwendet.
# 30 % bleiben komplett getrennt und werden anschließend
# zur Bewertung des Modells genutzt.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ============================================================
# 4. VERSCHIEDENE MODELLKONFIGURATIONEN TESTEN
# ============================================================

# C ist ein Hyperparameter der Logistic Regression.
#
# Kleines C:
# stärkere Regularisierung
#
# Großes C:
# schwächere Regularisierung
#
# Wir wollen herausfinden, welche Einstellung auf den
# Testdaten am besten funktioniert.
c_values = [0.1, 1.0, 10.0]


for c_value in c_values:

    # --------------------------------------------------------
    # 5. EIN NEUER MLFLOW RUN
    # --------------------------------------------------------

    # Für jeden C-Wert entsteht ein eigener Run.
    #
    # In MLflow siehst du anschließend:
    #
    # C_0.1
    # C_1.0
    # C_10.0
    #
    with mlflow.start_run(
        run_name=f"C_{c_value}"
    ):

        print("\n----------------------------")
        print("Training mit C =", c_value)
        print("----------------------------")


        # ====================================================
        # 6. MODELL ERZEUGEN
        # ====================================================

        model = LogisticRegression(
            C=c_value,
            max_iter=500
        )


        # ====================================================
        # 7. MODELL TRAINIEREN
        # ====================================================

        # Das Modell lernt anhand der Trainingsdaten.
        model.fit(
            X_train,
            y_train
        )


        # ====================================================
        # 8. VORHERSAGEN AUF TESTDATEN
        # ====================================================

        # Jetzt bekommt das Modell Daten, die es beim Training
        # nicht gesehen hat.
        prediction = model.predict(X_test)


        # ====================================================
        # 9. METRIKEN BERECHNEN
        # ====================================================

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        precision = precision_score(
            y_test,
            prediction,
            average="macro"
        )

        recall = recall_score(
            y_test,
            prediction,
            average="macro"
        )

        f1 = f1_score(
            y_test,
            prediction,
            average="macro"
        )

        f2 = fbeta_score(
            y_test,
            prediction,
            beta=2,
            average="macro"
        )


        # ====================================================
        # 10. PARAMETER IN MLFLOW SPEICHERN
        # ====================================================

        # Parameter beschreiben, WIE das Modell trainiert wurde.
        mlflow.log_param(
            "C",
            c_value
        )

        mlflow.log_param(
            "max_iter",
            500
        )

        mlflow.log_param(
            "test_size",
            0.30
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )


        # ====================================================
        # 11. METRIKEN IN MLFLOW SPEICHERN
        # ====================================================

        # Metriken beschreiben, WIE GUT das Modell war.
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "f2",
            f2
        )


        # ====================================================
        # 12. CONFUSION MATRIX ERZEUGEN
        # ====================================================

        # Damit kann man sehen, welche Klassen richtig oder
        # falsch vorhergesagt wurden.
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction
        )

        plt.title(
            f"Confusion Matrix - C={c_value}"
        )

        file_name = (
            f"confusion_matrix_C_{c_value}.png"
        )

        plt.savefig(
            file_name
        )

        plt.close()


        # ====================================================
        # 13. BILD ALS ARTIFACT SPEICHERN
        # ====================================================

        mlflow.log_artifact(
            file_name
        )


        # ====================================================
        # 14. MODELL IN MLFLOW SPEICHERN
        # ====================================================

        mlflow.sklearn.log_model(
            model,
            name="model"
        )


        # ====================================================
        # 15. ERGEBNIS AUF KONSOLE
        # ====================================================

        print("Accuracy :", round(accuracy, 3))
        print("Precision:", round(precision, 3))
        print("Recall   :", round(recall, 3))
        print("F1       :", round(f1, 3))
        print("F2       :", round(f2, 3))


print("\nTraining abgeschlossen.")
print("MLflow öffnen:")
print("http://localhost:5000")