"""
Lädt den SMS-Spam-Datensatz aus dem Internet herunter
und speichert ihn lokal im Projekt (Ordner /data).

Warum?
→ Damit die App später NICHT von einer Internetverbindung abhängt.
"""

import os
import pandas as pd

# Projekt-Basisordner bestimmen (eine Ebene über /src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Zielordner für Daten
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ziel-Dateipfad
DATA_PATH = os.path.join(DATA_DIR, "sms_spam.tsv")

# Quelle des Datensatzes (TSV = Tab-Separated Values)
URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def download_dataset():
    """
    Lädt den Datensatz von der URL und speichert ihn lokal.

    Schritte:
    1. Ordner /data anlegen (falls nicht vorhanden)
    2. TSV-Datei von URL laden
    3. Datei lokal speichern
    """

    # Sicherstellen, dass der data-Ordner existiert
    os.makedirs(DATA_DIR, exist_ok=True)

    # Datensatz laden (keine Header vorhanden → manuell setzen)
    df = pd.read_csv(
        URL,
        sep="\t",
        header=None,
        names=["label", "text"]
    )

    # Datei lokal speichern
    df.to_csv(DATA_PATH, sep="\t", index=False)

    print("Datensatz gespeichert unter:")
    print(DATA_PATH)

    # Vorschau anzeigen (Debug)
    print("\nBeispieldaten:")
    print(df.head())


# Script direkt ausführbar machen
if __name__ == "__main__":
    download_dataset()