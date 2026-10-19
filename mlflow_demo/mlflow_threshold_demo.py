import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from sklearn.model_selection import train_test_split


# ============================================================
# 1. MLFLOW KONFIGURIEREN
# ============================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment(
    "Threshold Evaluation Demo"
)


# ============================================================
# 2. BINÄREN TESTDATENSATZ ERZEUGEN
# ============================================================

# Klasse 0 = negativ
# Klasse 1 = positiv
#
# 70 % negativ
# 30 % positiv

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.7, 0.3],
    class_sep=1.0,
    random_state=42,
)


# ============================================================
# 3. TRAININGS- UND TESTDATEN
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)


# ============================================================
# 4. THRESHOLDS DEFINIEREN
# ============================================================

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# ============================================================
# 5. MLFLOW RUN STARTEN
# ============================================================

with mlflow.start_run(
    run_name="threshold_test"
):

    # ========================================================
    # 6. MODELL ERZEUGEN
    # ========================================================

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


    # ========================================================
    # 7. MODELL TRAINIEREN
    # ========================================================

    model.fit(
        X_train,
        y_train,
    )


    # ========================================================
    # 8. PARAMETER IN MLFLOW SPEICHERN
    # ========================================================

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_iter",
        1000
    )

    mlflow.log_param(
        "train_samples",
        len(X_train)
    )

    mlflow.log_param(
        "test_samples",
        len(X_test)
    )


    # ========================================================
    # 9. WAHRSCHEINLICHKEITEN BERECHNEN
    # ========================================================

    # Wir brauchen die Wahrscheinlichkeit für Klasse 1.
    #
    # Beispiel:
    #
    # 0.12
    # 0.48
    # 0.91
    #
    # Der Threshold entscheidet anschließend,
    # ab wann daraus "positiv" wird.

    probabilities = model.predict_proba(
        X_test
    )[:, 1]


    # ========================================================
    # 10. LISTEN FÜR DIE GRAFIK
    # ========================================================

    precision_values = []
    recall_values = []
    f1_values = []
    f2_values = []


    # ========================================================
    # 11. BESTEN THRESHOLD MERKEN
    # ========================================================

    best_threshold = 0
    best_f2 = 0


    # ========================================================
    # 12. ALLE THRESHOLDS TESTEN
    # ========================================================

    for step, threshold in enumerate(
        thresholds
    ):

        # Wahrscheinlichkeit in Klasse umwandeln.
        #
        # Beispiel Threshold 0.30:
        #
        # 0.20 -> negativ
        # 0.30 -> positiv
        # 0.80 -> positiv

        prediction = (
            probabilities >= threshold
        ).astype(int)


        # ====================================================
        # METRIKEN BERECHNEN
        # ====================================================

        precision = precision_score(
            y_test,
            prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            prediction,
            zero_division=0,
        )

        f2 = fbeta_score(
            y_test,
            prediction,
            beta=2,
            zero_division=0,
        )


        # ====================================================
        # WERTE FÜR GRAFIK SPEICHERN
        # ====================================================

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        f1_values.append(
            f1
        )

        f2_values.append(
            f2
        )


        # ====================================================
        # METRIKEN IN MLFLOW SPEICHERN
        # ====================================================

        mlflow.log_metric(
            "threshold",
            threshold,
            step=step,
        )

        mlflow.log_metric(
            "precision",
            precision,
            step=step,
        )

        mlflow.log_metric(
            "recall",
            recall,
            step=step,
        )

        mlflow.log_metric(
            "f1",
            f1,
            step=step,
        )

        mlflow.log_metric(
            "f2",
            f2,
            step=step,
        )


        # ====================================================
        # BESTEN F2-WERT SUCHEN
        # ====================================================

        if f2 > best_f2:

            best_f2 = f2
            best_threshold = threshold


        # ====================================================
        # KONSOLE
        # ====================================================

        print(
            f"Threshold={threshold:.2f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f} | "
            f"F2={f2:.3f}"
        )


    # ========================================================
    # 13. BESTEN THRESHOLD SPEICHERN
    # ========================================================

    mlflow.log_param(
        "best_threshold",
        best_threshold
    )

    mlflow.log_metric(
        "best_f2",
        best_f2
    )


    # ========================================================
    # 14. EINE GEMEINSAME GRAFIK
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )


    # Precision
    plt.plot(
        thresholds,
        precision_values,
        marker="o",
        label="Precision"
    )


    # Recall
    plt.plot(
        thresholds,
        recall_values,
        marker="o",
        label="Recall"
    )


    # F1
    plt.plot(
        thresholds,
        f1_values,
        marker="o",
        label="F1"
    )


    # F2
    plt.plot(
        thresholds,
        f2_values,
        marker="o",
        label="F2"
    )


    # ========================================================
    # BESTEN THRESHOLD MARKIEREN
    # ========================================================

    #plt.axvline(
    #    x=best_threshold,
    #    linestyle="--",
    #    label=(
    #        f"Bester Threshold: "
    #        f"{best_threshold:.2f}"
    #    )
    #)


    # ========================================================
    # GRAFIK BESCHRIFTEN
    # ========================================================

    plt.xlabel(
        "Threshold / Schwellenwert"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Precision, Recall, F1 und F2 nach Threshold"
    )

    plt.xlim(
        0,
        1
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # 15. GRAFIK SPEICHERN
    # ========================================================

    plt.savefig(
        "threshold_evaluation.png"
    )

    plt.close()


    # ========================================================
    # 16. GRAFIK IN MLFLOW SPEICHERN
    # ========================================================

    mlflow.log_artifact(
        "threshold_evaluation.png"
    )


    # ========================================================
    # 17. MODELL IN MLFLOW SPEICHERN
    # ========================================================

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )


# ============================================================
# 18. AUSGABE
# ============================================================

print("\n================================")
print("AUSWERTUNG FERTIG")
print("================================")

print(
    "Bester Threshold:",
    best_threshold
)

print(
    "Bester F2:",
    round(
        best_f2,
        3
    )
)

print(
    "\nMLflow:"
)

print(
    "http://localhost:5000"
)