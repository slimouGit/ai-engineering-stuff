# Multi-Agent Support Demo

Kleine Demo mit:

- 3 spezialisierten KI-Agenten
- lokalem Ollama
- RAG mit `nomic-embed-text`
- MCP-Server für Ticket-Tools
- JSON-Datei als Ticket-Speicher

## Voraussetzungen

- Python 3.10+
- Ollama lokal installiert und gestartet

## Installation

```bash
pip install -r requirements.txt
```

Modelle laden:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Start

```bash
python pandas-app.py
```

Der MCP-Server wird vom Client über stdio als Subprozess gestartet.

## Beispiel

```text
Seit dem letzten Update kann ich mich nicht mehr anmelden.
```

Die Anfrage läuft durch:

1. TicketAgent
2. KnowledgeAgent + RAG
3. ActionAgent + MCP
