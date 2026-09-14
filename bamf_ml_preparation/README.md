# BAMF ML Preparation Project

Kleines End-to-End-Projekt zur Vorbereitung auf eine Rolle an der Schnittstelle
zwischen Data Science, Machine Learning und Backend-Entwicklung.

## Use Case

Freitext-Dokumente werden automatisch einer Kategorie zugeordnet:

- `antrag`
- `termin`
- `nachweis`
- `allgemeine_anfrage`

Das Projekt ist bewusst klein gehalten. Es demonstriert aber den kompletten ML-Workflow:

`Daten -> Preprocessing -> Training -> Evaluation -> Modell speichern -> REST API`

## Technik

- Python
- Pandas
- scikit-learn
- TF-IDF
- Logistic Regression
- joblib
- FastAPI
- pytest

## 1. Virtuelle Umgebung

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

## 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

## 3. Modell trainieren

```bash
python train.py
```

Dabei passiert Folgendes:

1. CSV mit Pandas laden
2. Train-/Test-Split erzeugen
3. Texte mit TF-IDF in numerische Features umwandeln
4. Logistic-Regression-Modell trainieren
5. Accuracy, Precision, Recall und F1 ausgeben
6. Confusion Matrix anzeigen
7. Modell mit joblib speichern

## 4. API starten

```bash
uvicorn app.main:app --reload
```

Swagger UI:

`http://127.0.0.1:8000/docs`

## Beispielrequest

```json
{
  "text": "Ich möchte meinen Termin auf nächste Woche verschieben."
}
```

Beispielresponse:

```json
{
  "label": "termin",
  "confidence": 0.72
}
```

## 5. Tests

Nachdem `python train.py` ausgeführt wurde:

```bash
pytest
```

## Was du daran fürs Gespräch erklären können solltest

### Warum TF-IDF?

Ein ML-Modell kann Text nicht direkt verarbeiten. TF-IDF wandelt Wörter bzw.
Wortkombinationen in numerische Merkmale um und gewichtet dabei informative
Begriffe stärker.

### Warum Logistic Regression?

Logistic Regression ist ein einfacher, gut verständlicher und häufig starker
Baseline-Algorithmus für Textklassifikation.

### Was sind Features und Labels?

Die TF-IDF-Werte sind die Features. Die Kategorie eines Textes ist das Label.

### Was macht fit()?

`fit()` trainiert die Pipeline anhand der Trainingsdaten.

### Was macht predict()?

`predict()` verwendet das trainierte Modell für neue, unbekannte Texte.

### Warum Train/Test Split?

Damit die Modellqualität auf Daten bewertet wird, die während des Trainings
nicht gesehen wurden.

### Was ist Overfitting?

Das Modell wäre overfitted, wenn es die Trainingsdaten sehr gut erkennt,
aber bei neuen Formulierungen schlecht generalisiert.

### Warum Precision / Recall / F1?

Accuracy allein kann insbesondere bei ungleich verteilten Klassen irreführend
sein. Precision und Recall zeigen unterschiedliche Fehlertypen; F1 kombiniert
beide.

## Sinnvolle Erweiterungen

1. Größeren realistischen Datensatz verwenden
2. Datenbereinigung und Exploratory Data Analysis ergänzen
3. Cross Validation verwenden
4. Hyperparameter mit GridSearchCV optimieren
5. Modellversionierung ergänzen
6. Dockerfile hinzufügen
7. Monitoring / Logging ergänzen
8. Klassischen ML-Ansatz mit BERT/Transformer vergleichen
9. RAG-Komponente ergänzen, um nach der Klassifikation passende Wissensartikel zu suchen
