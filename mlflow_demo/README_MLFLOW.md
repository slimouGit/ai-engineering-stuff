# MLflow Demo – Startanleitung

Diese Anleitung beschreibt, wie die lokale MLflow-Umgebung später wieder gestartet wird.

## 1. Projektordner öffnen

In PowerShell:

```powershell
cd C:\Users\SalimOussayfi\devTools\gen-ai\ai-engineering-stuff\mlflow_demo
```

## 2. MLflow-Server starten

```powershell
py -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

Danach läuft MLflow normalerweise unter:

```text
http://127.0.0.1:5000
```

Das PowerShell-Fenster mit dem MLflow-Server geöffnet lassen.

## 3. MLflow UI öffnen

Im Browser:

```text
http://127.0.0.1:5000
```

Dort sind Experimente, Runs, Parameter, Metriken und Traces sichtbar.

## 4. Python-App starten

In einem zweiten PowerShell-Fenster:

```powershell
cd C:\Users\SalimOussayfi\devTools\gen-ai\ai-engineering-stuff\mlflow_demo
py app.py
```

Die App verwendet:

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

Dadurch werden neue Runs an den lokal laufenden MLflow-Server geschickt.

## 5. Sind die Experimente nach einem Neustart noch vorhanden?

Ja. MLflow speichert die Experimente persistent.

Bei diesem Startbefehl:

```powershell
py -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

werden die Daten in der Datei

```text
mlflow.db
```

im aktuellen Projektordner gespeichert.

Solange diese Datei nicht gelöscht wird und der Server wieder mit derselben Datenbank gestartet wird, bleiben die Experimente und Runs erhalten.

Wichtig: Die Datei `mlflow.db` daher nicht löschen.

## 6. Falls bestehende Experimente nicht angezeigt werden

Dann wurde der MLflow-Server wahrscheinlich früher aus einem anderen Ordner gestartet und verwendet dort eine andere `mlflow.db`.

Nach vorhandenen Datenbanken suchen:

```powershell
Get-ChildItem C:\Users\SalimOussayfi -Filter mlflow.db -Recurse -ErrorAction SilentlyContinue
```

Wenn die richtige Datenbank gefunden wurde, den Server aus dem entsprechenden Ordner starten oder diese Datenbank gezielt als Backend verwenden.

Bei älteren MLflow-Setups können Experimente auch in einem Ordner namens

```text
mlruns
```

liegen.

## 7. Server beenden

Im PowerShell-Fenster des MLflow-Servers:

```text
CTRL + C
```

Die gespeicherten Experimente werden dadurch nicht gelöscht.

## Kurzfassung

Starten:

```powershell
cd C:\Users\SalimOussayfi\devTools\gen-ai\ai-engineering-stuff\mlflow_demo
py -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

Browser:

```text
http://127.0.0.1:5000
```

App in zweitem Terminal:

```powershell
cd C:\Users\SalimOussayfi\devTools\gen-ai\ai-engineering-stuff\mlflow_demo
py app.py
```
