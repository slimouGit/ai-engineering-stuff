from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# KONFIGURATION
# ============================================================

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Threshold Metrics Demo"

MODEL_FILE = Path("saved_model.joblib")
GRAPH_FILE = Path("threshold_metrics.png")
VALUES_FILE = Path("threshold_metrics.csv")

RANDOM_STATE = 42

# Schwellenwerte von 5 % bis 95 %
THRESHOLDS = [x / 100 for x in range(5, 100, 5)]


# ============================================================
# ANSCHAULICHE DATENGRUNDLAGE
# ============================================================
#
# Beispiel: Kundenservice.
#
# Aufgabe des Modells:
# "dringend" oder "nicht_dringend"
#
# Das Modell soll anhand eines kurzen Kundentextes erkennen,
# ob der Fall dringend bearbeitet werden sollte.
#
# Positivklasse = "dringend"
# ============================================================

DATA = [
    # dringend
    ("Meine Karte wurde gestohlen und es wurden bereits Zahlungen durchgeführt.", "dringend"),
    ("Mein Konto ist gesperrt und ich kann keine Überweisung ausführen.", "dringend"),
    ("Ich sehe mehrere Abbuchungen, die ich nicht getätigt habe.", "dringend"),
    ("Meine Kreditkarte wurde verloren, bitte sofort sperren.", "dringend"),
    ("Ich kann nicht auf mein Konto zugreifen und benötige dringend Zugriff.", "dringend"),
    ("Eine unbekannte Person hat Geld von meinem Konto überwiesen.", "dringend"),
    ("Meine Karte wurde am Automaten eingezogen und ich bin im Ausland.", "dringend"),
    ("Mein Onlinebanking wurde offenbar gehackt.", "dringend"),
    ("Ich habe einen Betrugsverdacht bei einer aktuellen Zahlung.", "dringend"),
    ("Mein Konto zeigt plötzlich einen sehr hohen unbekannten Umsatz.", "dringend"),
    ("Meine Karte funktioniert nicht und ich kann meine Reise nicht bezahlen.", "dringend"),
    ("Ich habe mein Handy mit Banking-App verloren.", "dringend"),
    ("Es wurden heute mehrere fremde Lastschriften gebucht.", "dringend"),
    ("Mein Zugang wurde nach mehreren Anmeldeversuchen blockiert.", "dringend"),
    ("Ich brauche sofort Hilfe, weil eine verdächtige Überweisung läuft.", "dringend"),
    ("Meine Karte wurde kopiert und missbräuchlich verwendet.", "dringend"),
    ("Ich habe gerade eine Phishing-Freigabe bestätigt.", "dringend"),
    ("Eine Überweisung an einen Betrüger wurde gerade ausgeführt.", "dringend"),
    ("Mein Geschäftskonto ist gesperrt und Gehälter können nicht bezahlt werden.", "dringend"),
    ("Ich kann meine Karte nach einem Diebstahl nicht selbst sperren.", "dringend"),
    ("Ein hoher Betrag wurde ohne meine Zustimmung abgebucht.", "dringend"),
    ("Mein Konto wurde übernommen und meine Kontaktdaten wurden geändert.", "dringend"),
    ("Ich habe eine verdächtige Push-TAN bestätigt.", "dringend"),
    ("Die Karte meiner Firma wurde gestohlen.", "dringend"),
    ("Ich erkenne eine Überweisung auf meinem Konto nicht.", "dringend"),
    ("Mein Banking-Zugang funktioniert seit einer verdächtigen SMS nicht mehr.", "dringend"),
    ("Jemand versucht gerade mehrfach mit meiner Karte zu bezahlen.", "dringend"),
    ("Ich habe versehentlich Zugangsdaten an eine Phishing-Seite gegeben.", "dringend"),
    ("Meine Karte ist weg und ich sehe schon neue Umsätze.", "dringend"),
    ("Mein Konto wurde wegen eines Sicherheitsproblems eingefroren.", "dringend"),

    # nicht dringend
    ("Wie kann ich meine Adresse ändern?", "nicht_dringend"),
    ("Wo finde ich meine Kontoauszüge?", "nicht_dringend"),
    ("Wie hoch ist das tägliche Überweisungslimit?", "nicht_dringend"),
    ("Ich möchte eine neue PIN bestellen.", "nicht_dringend"),
    ("Wie ändere ich meine Telefonnummer im Profil?", "nicht_dringend"),
    ("Welche Gebühren fallen für eine Auslandsüberweisung an?", "nicht_dringend"),
    ("Wann wird meine neue Karte geliefert?", "nicht_dringend"),
    ("Wie kann ich einen Dauerauftrag einrichten?", "nicht_dringend"),
    ("Wo sehe ich die Umsätze des letzten Jahres?", "nicht_dringend"),
    ("Ich möchte mein Überweisungslimit erhöhen.", "nicht_dringend"),
    ("Wie kann ich den Namen meines Kontos ändern?", "nicht_dringend"),
    ("Welche Kartenmodelle bieten Sie an?", "nicht_dringend"),
    ("Kann ich Kontoauszüge als PDF herunterladen?", "nicht_dringend"),
    ("Wie richte ich einen Freistellungsauftrag ein?", "nicht_dringend"),
    ("Wann werden Überweisungen normalerweise gebucht?", "nicht_dringend"),
    ("Wie ändere ich die Sprache in der App?", "nicht_dringend"),
    ("Ich möchte Informationen zu einem Sparkonto.", "nicht_dringend"),
    ("Kann ich meine Karte auch mit Google Pay verwenden?", "nicht_dringend"),
    ("Wie kann ich einen Empfänger aus meiner Liste löschen?", "nicht_dringend"),
    ("Wo finde ich meine IBAN in der App?", "nicht_dringend"),
    ("Wie bestelle ich eine Ersatzkarte wegen Ablaufdatum?", "nicht_dringend"),
    ("Was kostet eine zweite Karte?", "nicht_dringend"),
    ("Wie kann ich Benachrichtigungen für Umsätze aktivieren?", "nicht_dringend"),
    ("Kann ich mein Konto auch im Browser verwalten?", "nicht_dringend"),
    ("Wie ändere ich mein Passwort freiwillig?", "nicht_dringend"),
    ("Welche Öffnungszeiten hat die Filiale?", "nicht_dringend"),
    ("Wie lange dauert eine normale Überweisung?", "nicht_dringend"),
    ("Ich möchte meine E-Mail-Adresse aktualisieren.", "nicht_dringend"),
    ("Wo kann ich meine persönlichen Daten einsehen?", "nicht_dringend"),
    ("Wie kann ich einen Kontoauszug erneut herunterladen?", "nicht_dringend"),
]


# ============================================================
# DATEN AUFTEILEN
# ============================================================

def get_data():
    """
    Erstellt aus den leicht verständlichen Beispieltexten
    ein DataFrame und teilt es reproduzierbar in Training und Test.
    """
    df = pd.DataFrame(DATA, columns=["text", "label"])

    train_df, test_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ============================================================
# METRIKEN FÜR EINEN SCHWELLENWERT
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    """
    Wandelt die Modellwahrscheinlichkeit anhand des Thresholds
    in die Klassen dringend / nicht_dringend um.

    Beispiel:
        Wahrscheinlichkeit = 0.72
        Threshold = 0.60
        => dringend

        Wahrscheinlichkeit = 0.72
        Threshold = 0.80
        => nicht_dringend
    """

    predictions = [
        "dringend" if probability >= threshold else "nicht_dringend"
        for probability in probabilities
    ]

    precision = precision_score(
        y_true,
        predictions,
        pos_label="dringend",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        pos_label="dringend",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        pos_label="dringend",
        zero_division=0,
    )

    f2 = fbeta_score(
        y_true,
        predictions,
        beta=2,
        pos_label="dringend",
        zero_division=0,
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


# ============================================================
# MODELL TRAINIEREN
# ============================================================

def train_model():
    """
    Trainiert ein kleines Textklassifikationsmodell.

    Pipeline:
        Text
        -> TF-IDF
        -> Logistic Regression
        -> Wahrscheinlichkeit für "dringend"
    """

    train_df, test_df = get_data()

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    model.fit(
        train_df["text"],
        train_df["label"],
    )

    # Das trainierte Modell lokal speichern.
    joblib.dump(model, MODEL_FILE)

    print(f"\nModell gespeichert: {MODEL_FILE}")

    return model, test_df


# ============================================================
# WAHRSCHEINLICHKEIT FÜR "DRINGEND"
# ============================================================

def get_urgent_probabilities(model, texts):
    """
    predict_proba liefert für jede Klasse eine Wahrscheinlichkeit.

    Wir suchen bewusst die Spalte der Klasse "dringend",
    damit die App unabhängig von der internen Klassenreihenfolge ist.
    """

    classes = list(model.classes_)
    urgent_index = classes.index("dringend")

    return model.predict_proba(texts)[:, urgent_index]


# ============================================================
# ALLE SCHWELLENWERTE AUSWERTEN
# ============================================================

def evaluate_all_thresholds(model, test_df):
    """
    Berechnet Precision, Recall, F1 und F2 für jeden Threshold
    von 0.05 bis 0.95.
    """

    probabilities = get_urgent_probabilities(
        model,
        test_df["text"],
    )

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            test_df["label"],
            probabilities,
            threshold,
        )

        results.append(metrics)

    result_df = pd.DataFrame(results)

    # Werte zusätzlich als CSV speichern.
    result_df.to_csv(
        VALUES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return result_df


# ============================================================
# EINE GEMEINSAME GRAFIK
# ============================================================

def create_graph(result_df, selected_threshold=None):
    """
    Erstellt EINE Grafik mit:
        Precision
        Recall
        F1
        F2

    X-Achse:
        Threshold / Schwellenwert

    Y-Achse:
        Score von 0 bis 1
    """

    plt.figure(figsize=(10, 6))

    plt.plot(
        result_df["threshold"],
        result_df["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        result_df["threshold"],
        result_df["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        result_df["threshold"],
        result_df["f1"],
        marker="o",
        label="F1",
    )

    plt.plot(
        result_df["threshold"],
        result_df["f2"],
        marker="o",
        label="F2",
    )

    # Falls der Nutzer einen eigenen Threshold gewählt hat,
    # wird er in derselben Grafik markiert.
    if selected_threshold is not None:
        plt.axvline(
            x=selected_threshold,
            linestyle="--",
            label=f"Gewählt: {selected_threshold:.2f}",
        )

    plt.xlabel("Threshold / Schwellenwert")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 und F2 je Schwellenwert")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(GRAPH_FILE, dpi=150)
    plt.close()

    print(f"Grafik gespeichert: {GRAPH_FILE}")


# ============================================================
# MLFLOW SPEICHERN
# ============================================================

def log_to_mlflow(result_df, selected_threshold=None):
    """
    Speichert Modell, Metrikverläufe, CSV und die eine gemeinsame
    Grafik in MLflow.
    """

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="threshold_evaluation"):

        mlflow.log_param("model", "TF-IDF + LogisticRegression")
        mlflow.log_param("positive_class", "dringend")
        mlflow.log_param("threshold_count", len(THRESHOLDS))

        if selected_threshold is not None:
            mlflow.log_param(
                "selected_threshold",
                selected_threshold,
            )

        # Jede Metrik wird als Verlauf gespeichert.
        for step, row in result_df.iterrows():
            mlflow.log_metric(
                "threshold",
                float(row["threshold"]),
                step=step,
            )
            mlflow.log_metric(
                "precision",
                float(row["precision"]),
                step=step,
            )
            mlflow.log_metric(
                "recall",
                float(row["recall"]),
                step=step,
            )
            mlflow.log_metric(
                "f1",
                float(row["f1"]),
                step=step,
            )
            mlflow.log_metric(
                "f2",
                float(row["f2"]),
                step=step,
            )

        mlflow.log_artifact(str(GRAPH_FILE))
        mlflow.log_artifact(str(VALUES_FILE))
        mlflow.log_artifact(str(MODEL_FILE))


# ============================================================
# EINEN INDIVIDUELLEN THRESHOLD VERGLEICHEN
# ============================================================

def compare_threshold(model, test_df):
    """
    Der Nutzer gibt selbst einen Threshold ein.

    Dieser wird verglichen mit:
        - Standardwert 0.50
        - Threshold mit höchstem F1
        - Threshold mit höchstem F2
    """

    user_input = input(
        "\nIndividuellen Threshold eingeben (z. B. 0.30): "
    ).strip()

    try:
        selected_threshold = float(user_input)
    except ValueError:
        print("Ungültige Zahl.")
        return

    if not 0 <= selected_threshold <= 1:
        print("Threshold muss zwischen 0 und 1 liegen.")
        return

    probabilities = get_urgent_probabilities(
        model,
        test_df["text"],
    )

    result_df = evaluate_all_thresholds(
        model,
        test_df,
    )

    selected = calculate_metrics(
        test_df["label"],
        probabilities,
        selected_threshold,
    )

    default = calculate_metrics(
        test_df["label"],
        probabilities,
        0.50,
    )

    best_f1_row = result_df.loc[
        result_df["f1"].idxmax()
    ]

    best_f2_row = result_df.loc[
        result_df["f2"].idxmax()
    ]

    print("\n========================================================")
    print("VERGLEICH")
    print("========================================================")
    print(
        f"{'Variante':<20}"
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'F2':>10}"
    )

    print(
        f"{'Standard':<20}"
        f"{default['threshold']:>10.2f}"
        f"{default['precision']:>12.3f}"
        f"{default['recall']:>10.3f}"
        f"{default['f1']:>10.3f}"
        f"{default['f2']:>10.3f}"
    )

    print(
        f"{'Bester F1':<20}"
        f"{best_f1_row['threshold']:>10.2f}"
        f"{best_f1_row['precision']:>12.3f}"
        f"{best_f1_row['recall']:>10.3f}"
        f"{best_f1_row['f1']:>10.3f}"
        f"{best_f1_row['f2']:>10.3f}"
    )

    print(
        f"{'Bester F2':<20}"
        f"{best_f2_row['threshold']:>10.2f}"
        f"{best_f2_row['precision']:>12.3f}"
        f"{best_f2_row['recall']:>10.3f}"
        f"{best_f2_row['f1']:>10.3f}"
        f"{best_f2_row['f2']:>10.3f}"
    )

    print(
        f"{'Eigene Wahl':<20}"
        f"{selected['threshold']:>10.2f}"
        f"{selected['precision']:>12.3f}"
        f"{selected['recall']:>10.3f}"
        f"{selected['f1']:>10.3f}"
        f"{selected['f2']:>10.3f}"
    )

    # Grafik neu erzeugen und eigenen Threshold markieren.
    create_graph(
        result_df,
        selected_threshold=selected_threshold,
    )

    # Vergleich als neuer MLflow-Run speichern.
    log_to_mlflow(
        result_df,
        selected_threshold=selected_threshold,
    )


# ============================================================
# EINEN EIGENEN TEXT KLASSIFIZIEREN
# ============================================================

def classify_text(model):
    """
    Nutzt das gespeicherte Modell mit einem selbst gewählten
    Schwellenwert für einen neuen Kundentext.
    """

    text = input(
        "\nKundentext eingeben: "
    ).strip()

    if not text:
        print("Kein Text eingegeben.")
        return

    threshold_text = input(
        "Threshold eingeben (z. B. 0.30): "
    ).strip()

    try:
        threshold = float(threshold_text)
    except ValueError:
        print("Ungültige Zahl.")
        return

    if not 0 <= threshold <= 1:
        print("Threshold muss zwischen 0 und 1 liegen.")
        return

    probability = get_urgent_probabilities(
        model,
        [text],
    )[0]

    prediction = (
        "dringend"
        if probability >= threshold
        else "nicht_dringend"
    )

    print("\n--- Ergebnis ---")
    print("Text:", text)
    print(f"Wahrscheinlichkeit 'dringend': {probability:.3f}")
    print(f"Verwendeter Threshold: {threshold:.2f}")
    print("Klassifikation:", prediction)


# ============================================================
# MODELL LADEN ODER TRAINIEREN
# ============================================================

def load_or_train_model():
    train_df, test_df = get_data()

    if MODEL_FILE.exists():
        print(f"Gespeichertes Modell wird geladen: {MODEL_FILE}")
        model = joblib.load(MODEL_FILE)
    else:
        print("Noch kein Modell vorhanden -> Training wird gestartet.")
        model, test_df = train_model()

    return model, test_df


# ============================================================
# HAUPTMENÜ
# ============================================================

def main():
    model, test_df = load_or_train_model()

    while True:
        print("""
============================================================
THRESHOLD METRICS DEMO
============================================================
1 - Modell neu trainieren + alle Thresholds auswerten
2 - Eigenen Threshold testen und vergleichen
3 - Eigenen Kundentext mit individuellem Threshold testen
4 - Gespeicherte Threshold-Werte anzeigen
0 - Ende
""")

        choice = input("Auswahl: ").strip()

        if choice == "1":
            model, test_df = train_model()

            result_df = evaluate_all_thresholds(
                model,
                test_df,
            )

            create_graph(result_df)
            log_to_mlflow(result_df)

            print("\n--- Ergebnisse ---")
            print(result_df.to_string(index=False))

        elif choice == "2":
            compare_threshold(
                model,
                test_df,
            )

        elif choice == "3":
            classify_text(model)

        elif choice == "4":
            if not VALUES_FILE.exists():
                print(
                    "Noch keine Auswertung vorhanden. "
                    "Bitte zuerst Menüpunkt 1 ausführen."
                )
                continue

            result_df = pd.read_csv(VALUES_FILE)
            print("\n", result_df.to_string(index=False))

        elif choice == "0":
            break

        else:
            print("Ungültige Auswahl.")


if __name__ == "__main__":
    main()
