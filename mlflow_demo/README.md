# MLflow-Demos

Dieses Projekt enthält zwei kleine Beispiele dafür, wie MLflow Trainingsläufe, Parameter, Metriken, Modellaufrufe und Traces erfasst. Die Beispiele verwenden einen lokalen MLflow-Tracking-Server unter `http://localhost:5000`.

## Voraussetzungen

- Python 3.10 oder neuer
- Ein OpenAI-API-Schlüssel für den Promptvergleich
- PowerShell unter Windows (die Befehle lassen sich für andere Terminals entsprechend anpassen)

Öffne ein Terminal im Verzeichnis **über** dem Projektordner `mlflow_demo` und installiere die benötigten Pakete:

```powershell
py -m pip install mlflow scikit-learn openai
```

Der Promptvergleich liest den Schlüssel aus `OPENAI_API_KEY`. Setze ihn für das aktuelle PowerShell-Fenster, bevor du das Beispiel startest:

```powershell
$env:OPENAI_API_KEY = "dein-api-schlüssel"
```

Der Schlüssel wird nicht in MLflow protokolliert. Lege ihn nicht in den Quellcode oder in Git ab.

## MLflow-Server starten

Starte den Server im Projektverzeichnis. Halte dieses Terminal während der Beispiele geöffnet:

```powershell
cd .\mlflow_demo
py -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

Öffne anschließend [http://127.0.0.1:5000](http://127.0.0.1:5000), um die MLflow-Oberfläche aufzurufen. Die SQLite-Datenbank und die Artefakte bleiben lokal im Projekt erhalten; lösche `mlflow.db` und `mlartifacts` nicht, wenn du die bisherigen Läufe behalten möchtest.

## Beispiel 1: Iris-Modell mit Trace

Öffne ein zweites Terminal im Verzeichnis **über** `mlflow_demo` und führe das Modul aus:

```powershell
py -m mlflow_demo.mlflow_trace_demo
```

Das Skript lädt den Iris-Datensatz, teilt ihn in Trainings- und Testdaten und trainiert eine logistische Regression für zwei Werte des Parameters `C` (`0.1` und `1.0`). Für jede Variante wird ein MLflow-Run mit eigenem Namen angelegt.

In der MLflow-Oberfläche findest du im Experiment **MLflow Trace Demo**:

- den Parameter `C` und die Metrik `accuracy` für jeden Run
- einen Trace des Aufrufs von `train_model`, einschließlich Modelltraining und Auswertung
- die Konsolenausgabe mit der Accuracy beider Varianten

Dieses Beispiel benötigt keinen API-Schlüssel.

## Beispiel 2: Prompts vergleichen

Setze `OPENAI_API_KEY` und führe das Skript ebenfalls aus dem Verzeichnis **über** `mlflow_demo` aus:

```powershell
py -m mlflow_demo.prompt_vergleich
```

Das Skript sendet drei unterschiedliche Fragen zu Machine Learning an das in `config.py` festgelegte OpenAI-Modell (`o4-mini`). Jeder Prompt wird als eigener Run im Experiment **Prompt Vergleich** erfasst. Der Run enthält den Prompt und den Modellnamen als Parameter, die Antwort als Artefakt `response.txt` sowie den von `mlflow.openai.autolog()` aufgezeichneten OpenAI-Trace.

Die Antworten werden zusätzlich im Terminal ausgegeben. Dieser Aufruf benötigt eine Internetverbindung und kann Kosten bei OpenAI verursachen.

## Wichtige MLflow-Funktionen in den Beispielen

- `mlflow.set_tracking_uri(...)` verbindet die Skripte mit dem lokalen Tracking-Server.
- `mlflow.set_experiment(...)` ordnet neue Runs einem benannten Experiment zu.
- `mlflow.start_run(...)` markiert einen einzelnen Ausführungsdurchlauf und gibt ihm einen Namen.
- `mlflow.log_param(...)` speichert Eingaben oder Einstellungen, zum Beispiel `C`, Prompt und Modell.
- `mlflow.log_metric(...)` speichert einen numerischen Messwert wie die Accuracy.
- `mlflow.log_text(...)` legt Text als Run-Artefakt ab.
- `@mlflow.trace` zeichnet einen Funktionsaufruf auf; `mlflow.openai.autolog()` aktiviert die automatische Erfassung von OpenAI-Aufrufen.

In der Oberfläche kannst du Runs vergleichen, Parameter und Metriken prüfen, Artefakte öffnen und die Details der Traces ansehen.

## Server beenden

Im Terminal, in dem der MLflow-Server läuft, `Ctrl+C` drücken. Die Datenbank und gespeicherten Artefakte bleiben erhalten.