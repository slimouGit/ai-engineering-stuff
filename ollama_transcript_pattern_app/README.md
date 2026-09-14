# Transcript Pattern Analyzer mit Ollama

Kleine lokale FastAPI-App für folgenden Use Case:

1. Das bestehende Interview-Transkript aus `data/interview.txt` laden
2. Das Transkript mit einem lokalen Ollama-Modell analysieren
3. Vordefinierte Muster mit Textbeleg, Begründung und Confidence ausgeben

## Architektur

```text
data/interview.txt
  |
  v
FastAPI
    |
    v
Ollama /api/chat
    |
    v
JSON: pattern + evidence + explanation + confidence
```

## 1. Ollama vorbereiten

Ollama starten:

```bash
ollama serve
```

Modell herunterladen, zum Beispiel:

```bash
ollama pull qwen2.5:7b
```

Wenn du ein anderes Modell verwenden willst:

Windows PowerShell:

```powershell
$env:OLLAMA_MODEL="granite3.3:8b"
```

Linux/macOS:

```bash
export OLLAMA_MODEL="granite3.3:8b"
```

## 2. Python-Umgebung

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Dann:

```bash
pip install -r requirements.txt
```

## 3. App starten

Im Projektordner:

```bash
uvicorn app.main:app --reload
```

Danach öffnen:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Interview analysieren

Die Browser-Oberfläche lädt automatisch `data/interview.txt`. Mit **Interview analysieren** wird genau dieses Transkript nach den definierten Mustern durchsucht.

## Text-API testen

```bash
curl -X POST http://127.0.0.1:8000/analyze-text \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "Seit drei Tagen habe ich starke Rückenschmerzen. Medikamente nehme ich keine."
  }'
```

Beispielantwort:

```json
{
  "transcript": "Seit drei Tagen habe ich starke Rückenschmerzen. Medikamente nehme ich keine.",
  "matches": [
    {
      "pattern": "beschwerden_symptome",
      "evidence": "starke Rückenschmerzen",
      "explanation": "Es wird ein konkretes Symptom genannt.",
      "confidence": 0.98
    },
    {
      "pattern": "zeitangabe_verlauf",
      "evidence": "Seit drei Tagen",
      "explanation": "Die Dauer der Beschwerden wird beschrieben.",
      "confidence": 0.99
    }
  ],
  "model": "qwen2.5:7b"
}
```

## Eigene Muster übergeben

`POST /analyze-text` akzeptiert optional eigene Muster:

```json
{
  "transcript": "...",
  "patterns": [
    {
      "name": "bearbeitungsdauer",
      "description": "Aussagen über lange Warte- oder Bearbeitungszeiten"
    },
    {
      "name": "fehlende_unterlagen",
      "description": "Aussagen über fehlende Dokumente oder Nachweise"
    }
  ]
}
```

Damit kannst du die medizinischen Übungsmuster später durch BAMF-nahe Muster ersetzen.

## Wichtige Dateien

- `app/main.py` – FastAPI-Endpunkte
- `app/ollama_client.py` – Ollama-Anbindung
- `app/analyzer.py` – Prompt und Musterlogik
- `app/schemas.py` – Pydantic-Datenmodelle
- `templates/index.html` – einfache Browser-Oberfläche

## Hinweis für echte Interviewdaten

Bei echten personenbezogenen Interviewdaten sollte die gesamte Verarbeitung lokal bzw. in der freigegebenen Projektumgebung erfolgen. Dieses Beispiel sendet nichts an einen externen Cloud-Dienst; Ollama und Whisper laufen lokal.
