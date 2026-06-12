# RAG Demo mit Flask, FastAPI, SQLite und Ollama

Lokale RAG-Demo zum Indexieren von TXT- und PDF-Dokumenten oder Webseiten, zur Speicherung von Chunks und Embeddings in SQLite und zur Fragebeantwortung über ein lokales Ollama-Setup.

## Aktueller Stand

Das Projekt besteht aktuell aus zwei Oberflächen auf derselben Codebasis:

- einer Flask-Weboberfläche in [app.py](app.py)
- einer FastAPI-API in [api.py](api.py)

Die Weboberfläche deckt den kompletten Arbeitsablauf ab:

1. Dokument oder URL einlesen
2. Text in Chunks aufteilen
3. Embeddings mit Ollama erzeugen
4. Chunks samt Embeddings in SQLite speichern
5. Fragen gegen die gespeicherten Chunks beantworten
6. Datenbankinhalt und Embedding-Vorschau im Browser anzeigen

Zusätzlich gibt es eine Datenbankansicht, in der Dokumente, Chunk-Listen und einzelne Chunks gelöscht werden können.

## Funktionen

- Upload von TXT- und PDF-Dateien
- Indexierung von Webseiten per URL
- Auswahl von Chat- und Embedding-Modell aus den lokal verfügbaren Ollama-Modellen
- RAG-Antworten auf Basis der relevantesten Chunks
- Datenbankansicht mit Dokumentübersicht und Chunk-Details
- Löschfunktion für Dokumente und einzelne Chunks
- Embedding-Vorschau in der UI
- Schutz vor offensichtlichen lokalen oder privaten Ziel-URLs beim URL-Ingest

## Architektur

### `app.py`

Flask-Webanwendung mit den HTML-Routen für:

- Startseite
- Upload
- URL-Ingest
- Fragebeantwortung
- Modellwechsel
- Datenbankansicht
- Löschen von Dokumenten und Chunks

### `api.py`

FastAPI-API für externe Clients. Die API stellt derzeit vor allem Modellverwaltung und Statusinformationen bereit:

- `GET /api/ollama/status`
- `POST /api/ollama/chat-model`
- `POST /api/ollama/embedding-model`
- `GET /api/ollama/used`

### `ragservice.py`

Kernlogik der RAG-Pipeline:

- Text zerlegen in überlappende Chunks
- Embeddings erzeugen
- Chunks in SQLite ablegen
- Ähnlichkeit per Cosine Similarity berechnen
- relevante Chunks an das Chat-Modell übergeben

### `documentservice.py`

Extraktion von Text aus:

- `.txt`
- `.pdf`
- Webseiten per HTTP(S)

Für URLs sind Timeouts, Größenlimits und einfache SSRF-Schutzmaßnahmen eingebaut.

### `ollamaservice.py`

Kommunikation mit dem lokalen Ollama-Server:

- Modellliste lesen
- Embeddings anfordern
- Antworten generieren

### `database.py`

SQLite-Zugriff für:

- Tabelleninitialisierung
- Dokumentübersicht
- Chunk-Abfragen
- Löschoperationen

### `config.py`

Zentrale Konfiguration für Pfade, Ollama-URL und Chunking-Werte.

## Datenmodell

Die SQLite-Datenbank liegt lokal als `rag.db` im Projektverzeichnis. Aktuell gibt es eine Tabelle:

- `document_chunks`

Spalten:

- `id`
- `document_name`
- `chunk_index`
- `content`
- `embedding`

## Warum Embeddings als JSON gespeichert werden

Die Embeddings werden aktuell als JSON-String in einer Textspalte gespeichert. Das ist hier bewusst so umgesetzt.

Vorteile dieser Lösung im aktuellen Projektstand:

- sehr einfach zu implementieren
- leicht lesbar und direkt in der DB prüfbar
- keine zusätzliche Serialisierungslogik nötig
- gut für einen lokalen Prototypen mit überschaubarer Datenmenge
- kompatibel mit der bisherigen Lade- und Vergleichslogik in Python

Warum nicht direkt BLOB:

- BLOB wäre kompakter und potenziell effizienter
- dafür bräuchte es eine saubere Binärserialisierung und passende Deserialisierung
- für das aktuelle Projekt bringt das noch keinen klaren Mehrwert, weil die Embeddings ohnehin komplett geladen und in Python verarbeitet werden
- Debugging wäre schwieriger, da BLOB nicht direkt lesbar ist

Kurz gesagt: JSON ist hier nicht das effizienteste Format, aber für den jetzigen Stand die pragmatischste Lösung.

## Warum nicht schon eine Vektor-Datenbank

Eine spezialisierte Vektor-Datenbank wäre sinnvoll, wenn:

- deutlich mehr Dokumente gespeichert werden
- Approximate-Nearest-Neighbor-Suche gebraucht wird
- Filter, Metadaten und Suche stärker skaliert werden müssen
- die RAG-Anwendung produktionsnäher wird

Für dieses Projekt ist SQLite mit JSON-Embeddings ausreichend, weil:

- die Datenmenge noch klein ist
- die Suche aktuell vollständig in Python über Cosine Similarity läuft
- die gesamte Lösung lokal, kompakt und nachvollziehbar bleiben soll

## Embedding-Vorschau und negative Werte

In der DB-Ansicht kann die Embedding-Vorschau auch negative Zahlen enthalten. Das ist normal.

Embeddings sind numerische Vektoren im Merkmalsraum, keine Wahrscheinlichkeiten. Einzelne Dimensionen dürfen positiv oder negativ sein. Negative Werte bedeuten daher nicht, dass etwas fehlerhaft ist.

## Voraussetzungen

- Python 3.11 oder neuer wird empfohlen
- lokaler Ollama-Server unter `http://localhost:11434`
- passende Modelle in Ollama installiert, z. B. ein Chat-Modell und ein Embedding-Modell

## Installation

Beispiel mit virtuellem Umfeld:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install flask fastapi uvicorn requests pydantic pypdf
```

Falls du bereits ein eigenes Environment nutzt, installiere die Abhängigkeiten dort entsprechend.

## Starten

### Flask-Weboberfläche

```bash
python app.py
```

Danach läuft die Oberfläche standardmäßig auf `http://127.0.0.1:5000`.

### FastAPI-API

```bash
uvicorn api:app --reload --port 8000
```

## Typischer Ablauf

1. Ollama starten und Modelle bereitstellen.
2. Die Flask-App öffnen.
3. Chat- und Embedding-Modell auswählen.
4. Dokument oder URL indexieren.
5. Eine Frage stellen.
6. Falls nötig, die Datenbankansicht öffnen und Chunks prüfen oder löschen.

## Hinweise zur Weiterentwicklung

- JSON-Embeddings können bei wachsendem Umfang durch BLOB oder eine Vektor-Datenbank ersetzt werden.
- Die Ähnlichkeitssuche ist aktuell linear und damit für kleine Datenmengen ausreichend.
- Die URL-Extraktion ist bewusst defensiv ausgelegt, um lokale/private Ziele zu blockieren.

## Projektstruktur

- `app.py` - Flask UI
- `api.py` - FastAPI API
- `ragservice.py` - RAG-Logik
- `ollamaservice.py` - Ollama-Anbindung
- `documentservice.py` - Text- und URL-Extraktion
- `database.py` - SQLite-Zugriff
- `config.py` - Konfiguration
- `templates/index.html` - Weboberfläche
- `uploads/` - hochgeladene Dateien
