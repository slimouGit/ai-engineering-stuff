# Ablauf der App: von Klick bis Ergebnis

Dieses Dokument beschreibt den sequenziellen Ablauf der App, wenn der Nutzer auf den Button "Interview analysieren" klickt.

## 1. Seite wird geladen

1. Der Browser ruft die Startseite auf:
   - GET /
2. FastAPI liefert die HTML-Datei aus:
   - templates/index.html
3. Die Seite enthält:
   - ein readonly-Textfeld für das Interview-Transkript
   - einen Button "Interview analysieren"
   - einen Bereich für die Analyseergebnisse

## 2. Transkript wird beim Seitenaufruf geladen

1. Die JavaScript-Funktion `loadTranscript()` wird beim Öffnen der Seite ausgeführt.
2. Sie sendet einen Request an den Backend-Endpunkt:
   - GET /transcript
3. In [app/main.py](app/main.py) läuft dabei:
   - `transcript()` wird aufgerufen
   - `read_transcript()` liest die Datei `data/interview.txt`
   - Wenn die Datei existiert und nicht leer ist, wird ihr Inhalt zurückgegeben
4. Der Browser setzt den Inhalt in das Textfeld `#transcript`.

## 3. Nutzer klickt auf "Interview analysieren"

1. Die Funktion `analyzeInterview()` im Frontend wird gestartet.
2. Sie setzt den Status auf:
   - "Analysiere Interview..."
3. Danach wird ein HTTP-Request gesendet:
   - POST /analyze-interview

## 4. Backend empfangt den Analyse-Request

1. In [app/main.py](app/main.py) wird der Endpunkt `analyze_interview()` aufgerufen.
2. Dieser ruft erneut `read_transcript()` auf.
3. Das Transkript wird erneut aus `data/interview.txt` gelesen.
4. Wenn das Transkript fehlt oder leer ist, entsteht ein HTTP-Fehler.
5. Wenn alles okay ist, wird die eigentliche Analyse gestartet:
   - `analyze_transcript(read_transcript())`

## 5. Analyse beginnt: Transkript wird in Abschnitte zerlegt

1. In [app/analyzer.py](app/analyzer.py) wird `analyze_transcript(transcript)` aufgerufen.
2. Die Funktion ruft `chunk_text(transcript)` auf.
3. `split_sentences(text)` zerlegt den Text in Sätze.
4. Dabei werden Abkürzungen wie "Dr." oder "z. B." geschützt, damit sie nicht fälschlich als Satzende interpretiert werden.
5. `chunk_text()` baut aus den Sätzen größere Blöcke mit einer festen Maximalgröße und leichtem Overlap.
6. Das Ziel ist:
   - möglichst ganze Sinnabschnitte
   - trotzdem keine zu großen Eingaben für das Modell

## 6. Für jeden Chunk wird eine Modellabfrage gestartet

1. Für jeden Chunk wird `analyze_chunk(chunk)` ausgeführt.
2. `build_user_prompt(chunk)` erzeugt aus den konfigurierten Musterdefinitionen den Prompt für Ollama.
3. `chat_json(system_prompt, user_prompt)` in [app/ollama_client.py](app/ollama_client.py) stellt den Request an Ollama:
   - POST /api/chat
4. Dabei werden gesendet:
   - das System-Prompt
   - der User-Prompt mit dem aktuellen Chunk
   - das gewählte Modell
   - JSON-Format als Antwort erwartet
5. Ollama liefert ein JSON zurück mit Einträgen in der Form:
   - pattern
   - evidence
   - explanation
   - confidence

## 7. Treffer werden validiert

1. Die Antwort wird geparst.
2. In `analyze_chunk()` werden nur Treffer akzeptiert, deren `pattern` zu den erlaubten Mustern gehört.
3. Das verhindert, dass fremde oder ungültige Muster aus der Modellantwort übernommen werden.

## 8. Ergebnisse aus allen Chunks werden zusammengeführt

1. In `analyze_transcript()` werden alle Treffer aus allen Chunks gesammelt.
2. `remove_duplicate_matches()` entfernt doppelte Treffer.
3. Das passiert anhand von:
   - Mustername
   - Textbeleg (normalisiert)
4. Danach bleibt eine bereinigte Trefferliste übrig.

## 9. Antwort wird an das Frontend zurückgegeben

1. `AnalysisResponse` wird erzeugt mit:
   - `transcript`
   - `matches`
   - `model`
2. Der FastAPI-Endpunkt gibt diese Struktur als JSON zurück.
3. Das Frontend erhält die Antwort.

## 10. Frontend rendert das Ergebnis

1. `analyzeInterview()` prüft den HTTP-Status.
2. Wenn kein Fehler vorliegt, setzt es den Status auf:
   - "Fertig. Modell: ..."
3. Danach ruft es `render(data)` auf.
4. `render()` erzeugt für jeden Treffer eine Karte mit:
   - Mustername
   - Treffertext
   - Erklärung
   - Confidence in Prozent
5. Die Ergebnisse werden in `#results` eingefügt.

## 11. Abschluss

Damit ist der komplette Ablauf abgeschlossen:

- Frontend klickt Button
- Backend liest Transkript
- Text wird in Chunks aufgeteilt
- Ollama analysiert jeden Chunk
- Treffer werden zusammengeführt und dedupliziert
- JSON-Ergebnis wird zurückgegeben
- Browser rendert die Matches im UI

## Kurz gesagt

Der gesamte Ablauf ist also:

Frontend Button -> POST /analyze-interview -> read_transcript() -> chunk_text() -> analyze_chunk() -> Ollama /api/chat -> Matches validieren -> deduplizieren -> JSON-Ergebnis -> render() -> UI
