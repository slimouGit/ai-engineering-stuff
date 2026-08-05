# Lokale Ollama-MCP-Demo

Diese Demo zeigt den Nutzen von MCP praktisch:

1. Du stellst eine normale Frage an Ollama.
2. Ollama erkennt, dass es Ticketdaten benötigt.
3. Ollama fordert einen Tool-Aufruf an.
4. Der Python-Client ruft das Tool über MCP auf.
5. Der MCP-Server liest oder ändert `tickets.json`.
6. Das Ergebnis geht zurück an Ollama, das die Antwort formuliert.

```text
Benutzer → Ollama → Python-Client → MCP-Server → tickets.json
                ←               ←            ←
```

## Voraussetzungen

- Python 3.10 oder neuer
- Ollama

## Windows: einfachster Start

Doppelklick auf:

```text
start_windows.bat
```

Das Skript erstellt eine virtuelle Umgebung, installiert die Pakete und lädt `qwen3:4b`.

## Manueller Start

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

Danach:

```bash
pip install -r requirements.txt
ollama pull qwen3:4b
python app.py
```

## Beispielfragen

```text
Zeige mir Ticket 1001.
Suche Tickets zum Thema Login.
Setze Ticket 1001 auf RESOLVED.
```

Achte auf diese Ausgabe:

```text
→ MCP-Tool: get_ticket({'ticket_id': 1001})
```

Genau dort wird der Unterschied sichtbar: Ollama antwortet nicht nur aus seinem Modellwissen, sondern ruft eine externe, standardisierte MCP-Funktion auf.

## Dateien

- `app.py`: Ollama-Client und Agentenschleife
- `mcp_server.py`: MCP-Server mit drei Tools
- `tickets.json`: lokale Beispieldaten
- `requirements.txt`: Python-Abhängigkeiten

## Anderes Modell verwenden

Windows PowerShell:

```powershell
$env:OLLAMA_MODEL="qwen3:8b"
python app.py
```

Linux/macOS:

```bash
OLLAMA_MODEL=qwen3:8b python app.py
```

Das Modell muss Tool-Calling unterstützen.
