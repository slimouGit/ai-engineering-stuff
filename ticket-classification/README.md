# Ticket Classification API

Eine kleine FastAPI-Anwendung zur Klassifizierung von Kundentickets anhand des Textinhalts. Die App analysiert eine Nachricht und ordnet sie automatisch in eine Kategorie ein, z. B. `Technik`, `Zugang`, `Abrechnung` oder `Allgemein`.

## Überblick

Das Projekt verwendet einen einfachen Machine-Learning-Workflow:

- Textvorkonditionierung mit `TfidfVectorizer`
- Klassifikation mit `LogisticRegression`
- Persistierung des trainierten Modells mit `joblib`
- Bereitstellung als REST-API mit FastAPI

Damit lassen sich Support-Tickets automatisch einordnen, ohne dass jede Anfrage manuell kategorisiert werden muss.

## Projektstruktur

```text
ticket-classification/
├── app.py                 # FastAPI-Anwendung
├── model.py               # Wrapper für das geladene Modell
├── train.py               # Training des Klassifizierungsmodells
├── model.joblib           # Trainiertes Modell
├── data/
│   └── tickets.csv       # Datensatz mit Tickets und Labels
├── README.md              # Projekt-Dokumentation
└── __pycache__/           # Python-Cache
```

## Voraussetzungen

- Python 3.10 oder höher
- pip
- optional: virtuelle Umgebung (`venv`)

## Installation

1. In das Projektverzeichnis wechseln:

```bash
cd ticket-classification
```

2. Virtuelle Umgebung erstellen:

```bash
python -m venv .venv
```

3. Umgebung aktivieren:

- Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows (CMD):

```cmd
.venv\Scripts\activate.bat
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

4. Abhängigkeiten installieren:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib pydantic
```

## App starten

Startet den lokalen API-Server:

```bash
uvicorn app:app --reload
```

Danach ist die API normalerweise unter dieser Adresse erreichbar:

- http://127.0.0.1:8000

Die Swagger-Dokumentation ist verfügbar unter:

- http://127.0.0.1:8000/docs
- from inside the ticket-classification folder): uvicorn app:app --reload --host 127.0.0.1 --port 8000

## API-Endpunkte

### Gesundheitscheck

```http
GET /health
```

Beispielantwort:

```json
{
  "status": "ok"
}
```

### Vorhersage einer Ticket-Kategorie

```http
POST /predict
```

Request-Body:

```json
{
  "text": "Ich kann mich nicht einloggen und bekomme eine Fehlermeldung"
}
```

Beispielantwort:

```json
{
  "category": "Zugang"
}
```

## Beispiel mit curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Meine Rechnung stimmt nicht und ich erhalte einen Fehler bei der Zahlung"}'
```

Erwartete Ausgabe:

```json
{
  "category": "Abrechnung"
}
```

## Modell trainieren

Falls das vorhandene Modell aktualisiert oder neu trainiert werden soll, kann das Training mit folgendem Befehl gestartet werden:

```bash
python train.py
```

Dabei wird:

- der Datensatz aus `data/tickets.csv` geladen,
- in Trainings- und Testdaten aufgeteilt,
- das Pipeline-Modell trainiert,
- die Klassifikationsmetriken ausgegeben,
- das Modell als `model.joblib` gespeichert.

## Datensatz

Der Datensatz enthält Texte und zugehörige Labels im Format:

```csv
text,label
"Mein Login funktioniert nicht","Technik"
"Ich habe mein Passwort vergessen","Zugang"
"Meine Rechnung ist falsch","Abrechnung"
```

## Hinweise

- Das Modell ist bewusst einfach gehalten und eignet sich gut für kleine, klare Support-Kategorien.
- Für realistischere Anwendungen mit mehr Texten und komplexeren Mustern sollten zusätzliche Daten, bessere Features oder Modelle wie `TF-IDF` mit mehr Parametern, `XGBoost` oder Transformer-basierte Ansätze evaluiert werden.
- Wenn neue Ticket-Kategorien oder neue Typen von Anfragen hinzukommen, muss der Datensatz erweitert und das Modell neu trainiert werden.

## Nutzung

Die App kann als Grundlage für eine interne Support- oder Helpdesk-Lösung dienen, zum Beispiel:

- automatisierte Ticket-Klassifizierung,
- Routing in die richtige Abteilung,
- Priorisierung von Support-Anfragen,
- Vorschläge für schnellere Bearbeitung.

## Fazit

Dieses Projekt zeigt eine vollständige, leicht verständliche Pipeline von Daten, Modelltraining und API-Deployment. Es ist ideal als Einstieg in Textklassifikation mit Python, scikit-learn und FastAPI.
