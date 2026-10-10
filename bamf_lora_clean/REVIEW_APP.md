# Review von `app.py`

Kritische Prüfung ohne Codeänderungen. Jeder Punkt hat eine ID und eine Checkbox,
damit er einzeln bearbeitet und abgehakt werden kann.

> Zeilennummern (`app.py`, `config.py`) beziehen sich auf den Stand zum Zeitpunkt der Review und verschieben sich nach Änderungen.

Legende Aufwand: **S** = klein (wenige Zeilen), **M** = mittel, **L** = größer / Umbau.

---

## Übersicht

| ID | Titel | Kategorie | Priorität | Aufwand | Status |
|---|---|---|---|---|---|
| A1 | Prompt wird beim Training abgeschnitten | Problem | Hoch | S | [ ] |
| A2 | Kein Validierungs-/Testsplit | Problem | Hoch | M | [ ] |
| A3 | Vorabprüfungen fehlen vor dem Training | Problem | Hoch | S | [ ] |
| A4 | Destruktive Reihenfolge beim Löschen | Problem | Mittel | S | [ ] |
| A5 | `training_info.json` zu früh geschrieben | Problem | Niedrig | S | [ ] |
| A6 | Ollama-Fehlertext und Antwort-Validierung | Problem | Mittel | S | [ ] |
| A7 | Fragile Prüfung per `ollama list` | Problem | Niedrig | S | [ ] |
| A8 | Kein Traceback im allgemeinen Fehler-Handler | Problem | Niedrig | S | [ ] |
| A9 | Inkonsistenter Modellname | Problem | Niedrig | S | [ ] |
| B1 | Ungenutzter Import `Path` | Redundanz | Niedrig | S | [ ] |
| B2 | Überflüssige Bedingung in `ensure_llama_cpp` | Redundanz | Niedrig | S | [ ] |
| B3 | Überflüssiges Löschen von `training_output` | Redundanz | Niedrig | S | [ ] |
| B4 | `del` + `empty_cache()` mit falscher Begründung | Redundanz | Niedrig | S | [ ] |
| B5 | Parameter, die dem Standard entsprechen | Redundanz | Niedrig | S | [ ] |
| B6 | Wirkungsloses Kürzen / `eos or ""` in `TextDataset` | Redundanz | Niedrig | S | [ ] |
| B7 | Pad-Token-Fallback | Redundanz | Niedrig | S | [ ] |
| B8 | Eigene Zielmodul-Prüfung | Redundanz | Niedrig | S | [ ] |
| B9 | Prompt-Fallback ohne Chat-Template | Redundanz | Niedrig | S | [ ] |
| B10 | `SYSTEM` im Modelfile | Redundanz | Niedrig | S | [ ] |
| B11 | Doppelte Ollama-URL | Redundanz | Niedrig | S | [ ] |
| B12 | Magische Zahlen im Code | Redundanz | Niedrig | S | [ ] |
| C1 | Padding auf feste Länge | Effizienz | Niedrig | M | [ ] |
| C2 | Modell erst auf CPU, dann auf GPU | Effizienz | Mittel | S | [ ] |
| C3 | Hoher Plattenverbrauch (ca. 18 GB) | Effizienz | Mittel | M | [ ] |
| C4 | `train_lora()` macht zu viel | Design | Mittel | L | [ ] |
| C5 | Langer System-Prompt in jedem Trainingsbeispiel | Design | Hoch | M | [ ] |
| C6 | Testfunktionen ohne Modellprüfung / Docstrings | Design | Niedrig | S | [ ] |

Empfohlene Reihenfolge: **A1 → A3 → A4 → A2 → C5 → Rest**.

---

## A. Echte Probleme

### A1 – Prompt wird beim Training abgeschnitten

**Codestellen:** `app.py:123-134`, `config.py:40`, `config.py:53`
```python
# app.py:129-134
# Die Antwort soll niemals komplett durch MAX_LENGTH abgeschnitten werden.
max_prompt_len = max(1, config.MAX_LENGTH - len(answer_ids))
prompt_ids = prompt_ids[-max_prompt_len:]          # schneidet den Prompt VORNE ab

input_ids = (prompt_ids + answer_ids)[: config.MAX_LENGTH]
labels = ([-100] * len(prompt_ids) + answer_ids)[: config.MAX_LENGTH]
```
```python
# config.py:40
MAX_LENGTH = 192
```
- **Priorität / Aufwand:** Hoch / S
- **Fundstelle:** `TextDataset.__init__`, `config.MAX_LENGTH`
- **Befund:** Gemessen: Der Prompt hat **376 Tokens**, `MAX_LENGTH` ist **192**.
  `prompt_ids[-max_prompt_len:]` schneidet am Anfang ab. Dabei gehen der Kopf
  (`<|im_start|>system`) und etwa die erste Hälfte des System-Prompts verloren,
  also auch die Definitionen von `relevant` / `nicht_relevant`.
- **Auswirkung:** Training und Abfrage sehen unterschiedliche Prompts. `ask_ollama`
  sendet den vollständigen Prompt, das Modell wurde aber auf einen gekürzten trainiert.
- **Lösungsansätze:**
  1. `MAX_LENGTH` über die reale Länge erhöhen (z. B. 512), dabei VRAM beobachten.
  2. System-Prompt kürzen (siehe C5).
  3. Beim Start prüfen und warnen/abbrechen, wenn ein Prompt gekürzt würde.
- **Akzeptanzkriterium:** Kein Trainingsbeispiel wird gekürzt (Prüfausgabe: längster Prompt ≤ `MAX_LENGTH`).

### A2 – Kein Validierungs-/Testsplit

**Codestellen:** `app.py:174`, `app.py:229`, `app.py:248`, `app.py:272-289`
```python
# app.py:174
samples = load_samples()                 # alle 500 Beispiele werden zum Training genutzt
# app.py:229
dataset = TextDataset(samples, tokenizer)
# app.py:248
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)   # kein eval_dataset
```
- **Priorität / Aufwand:** Hoch / M
- **Fundstelle:** `train_lora`
- **Befund:** Es wird nur auf allen 500 Beispielen trainiert. `train_loss` = 0,0087
  zeigt Auswendiglernen, nicht Generalisierung. Nirgends wird auf unbekannten Texten gemessen.
- **Lösungsansätze:**
  1. Daten z. B. 80/10/10 aufteilen (mit festem Seed), `eval_dataset` im `Trainer` setzen.
  2. Nach dem Training eine Genauigkeit auf Testdaten über Ollama berechnen und in `training_info.json` speichern.
- **Akzeptanzkriterium:** Ausgabe von Accuracy (idealerweise je Klasse) auf nicht trainierten Daten.

### A3 – Vorabprüfungen fehlen vor dem Training

**Codestellen:** `app.py:174-176` (Trainingsstart), `app.py:306-331` (git/llama.cpp), `app.py:371-372` (Ollama-CLI), `app.py:401-422` (`environment_check`)
```python
# app.py:317 – erst NACH dem Training erreicht
if shutil.which("git") is None:
# app.py:371
if shutil.which("ollama") is None:
# app.py:298-299 – Aufruf der späten Schritte
convert_to_gguf()
import_into_ollama()
```
- **Priorität / Aufwand:** Hoch / S
- **Fundstelle:** `train_lora`, `ensure_llama_cpp`, `import_into_ollama`
- **Befund:** Git, llama.cpp, Ollama-CLI, Ollama-Server und die Python-Abhängigkeiten
  des Konverters werden erst nach Training und Merge (ca. 6+ Minuten) geprüft.
  `environment_check` existiert, wird aber nicht vorgeschaltet.
- **Lösungsansatz:** Eine Funktion `preflight()` am Anfang von `train_lora()` aufrufen,
  die alles Nötige prüft (inkl. freiem Speicherplatz) und früh abbricht.
- **Akzeptanzkriterium:** Fehlende Voraussetzungen führen in unter 5 Sekunden zu einem Abbruch.

### A4 – Destruktive Reihenfolge beim Löschen

**Codestellen:** `app.py:254-255`, `app.py:262-263`, `app.py:343-344`
```python
# app.py:254-255
if config.ADAPTER_DIR.exists():
    shutil.rmtree(config.ADAPTER_DIR)
# app.py:262-263
if config.MERGED_MODEL_DIR.exists():
    shutil.rmtree(config.MERGED_MODEL_DIR)
# app.py:343-344
if config.GGUF_FILE.exists():
    config.GGUF_FILE.unlink()
```
- **Priorität / Aufwand:** Mittel / S
- **Fundstelle:** `train_lora`, `convert_to_gguf`
- **Befund:** `adapter/`, `merged_model/` und die alte GGUF-Datei werden vor dem Erzeugen
  der neuen Ergebnisse gelöscht. Schlägt etwas fehl, ist auch der letzte funktionierende Stand weg.
- **Lösungsansatz:** In temporäre Ordner/Dateien schreiben und erst nach Erfolg ersetzen
  (oder Ergebnisse mit Zeitstempel ablegen).
- **Akzeptanzkriterium:** Nach einem Fehler im Konvertierungsschritt ist der vorherige Stand noch vorhanden.

### A5 – `training_info.json` zu früh geschrieben

**Codestellen:** `app.py:272-289`, `app.py:298-299`
```python
# app.py:272-275 – geschrieben, bevor GGUF/Ollama laufen
config.TRAINING_INFO.write_text(
    json.dumps(
        {
            "trained": True,
```
- **Priorität / Aufwand:** Niedrig / S
- **Fundstelle:** `train_lora`
- **Befund:** `"trained": true` wird vor Konvertierung und Ollama-Import gespeichert.
  Die Datei wird außerdem nirgends gelesen.
- **Lösungsansatz:** Am Ende schreiben (oder Felder `gguf_created`, `ollama_imported` ergänzen),
  oder die Datei entfernen, falls sie nicht gebraucht wird.

### A6 – Ollama-Fehlertext und Antwort-Validierung

**Codestellen:** `app.py:40`, `app.py:71-90`
```python
# app.py:89-90
response.raise_for_status()                                  # Fehlertext von Ollama geht verloren
return response.json()["message"]["content"].strip()         # keine Prüfung gegen VALID_LABELS
# app.py:40
VALID_LABELS = {"relevant", "nicht_relevant"}                # nur in load_samples (app.py:60) genutzt
```
- **Priorität / Aufwand:** Mittel / S
- **Fundstelle:** `ask_ollama`
- **Befund:** `raise_for_status()` verwirft den Fehlertext von Ollama (z. B. „model not found“),
  daher nur „404 Not Found“. Die Antwort wird nicht gegen `VALID_LABELS` geprüft;
  `VALID_LABELS` dient nur der CSV-Prüfung.
- **Lösungsansatz:** Bei Status ≥ 400 `response.text` / `response.json()["error"]` in die Fehlermeldung aufnehmen;
  Antwort normalisieren und mit `VALID_LABELS` abgleichen (ungültig → Hinweis).

### A7 – Fragile Prüfung per `ollama list`

**Codestellen:** `app.py:382-398`
```python
# app.py:395-396
if config.OLLAMA_TRAINED_MODEL.lower() not in result.stdout.lower():
    raise RuntimeError("Ollama create lief durch, aber das Modell erscheint nicht in 'ollama list'.")
```
- **Priorität / Aufwand:** Niedrig / S
- **Fundstelle:** `import_into_ollama`
- **Befund:** Der Substring-Test (`name in stdout`) würde auch bei `trained-modell-v2` bestehen.
  Er ist zudem überflüssig, weil `ollama create` mit `check=True` schon bei Fehlern abbricht.
- **Lösungsansatz:** Entfernen oder exakt je Zeile / über `GET /api/tags` prüfen.

### A8 – Kein Traceback im allgemeinen Fehler-Handler

**Codestellen:** `app.py:484-485`
```python
except Exception as exc:
    print("\nFehler:", exc)
```
- **Priorität / Aufwand:** Niedrig / S
- **Fundstelle:** `main`, `except Exception`
- **Befund:** Nur `str(exc)` wird ausgegeben, was die Fehlersuche erschwert.
- **Lösungsansatz:** `traceback.print_exc()` oder ein Debug-Schalter.

### A9 – Inkonsistenter Modellname

**Codestellen:** `config.py:15`, `config.py:29`, `app.py:376`
```python
# config.py:15
OLLAMA_TRAINED_MODEL = "trained-modell"
# config.py:29
GGUF_FILE = ROOT / "trained-model.gguf"
```
- **Priorität / Aufwand:** Niedrig / S
- **Fundstelle:** `config.OLLAMA_TRAINED_MODEL = "trained-modell"`
- **Befund:** Zwei „l“, während die Datei `trained-model.gguf` heißt. Verwechslungsgefahr.
- **Lösungsansatz:** Einheitlich benennen (dabei bestehende Ollama-Modelle/Dokumentation anpassen).

---

## B. Redundanter oder unnötiger Code

### B1 – Ungenutzter Import `Path`

**Codestelle:** `app.py:29`
```python
from pathlib import Path
```
- **Fundstelle:** `from pathlib import Path` (Zeile 29)
- **Befund:** Wird in `app.py` nicht verwendet.
- **Lösungsansatz:** Zeile löschen.

### B2 – Überflüssige Bedingung in `ensure_llama_cpp`

**Codestelle:** `app.py:308-312`
```python
if config.CONVERT_SCRIPT.exists():
    return

if config.LLAMA_CPP_DIR.exists() and not config.CONVERT_SCRIPT.exists():   # 2. Teil immer wahr
```
- **Fundstelle:** `if config.LLAMA_CPP_DIR.exists() and not config.CONVERT_SCRIPT.exists():`
- **Befund:** Der zweite Teil ist immer wahr, weil oben bei vorhandenem Skript bereits `return` erfolgt.
- **Lösungsansatz:** Auf `if config.LLAMA_CPP_DIR.exists():` vereinfachen.

### B3 – Überflüssiges Löschen von `training_output`

**Codestellen:** `app.py:231-232`, `app.py:235`, `config.py:24`
```python
# app.py:231-232
if config.TRAINING_OUTPUT_DIR.exists():
    shutil.rmtree(config.TRAINING_OUTPUT_DIR)
# app.py:241
save_strategy="no",
```
- **Fundstelle:** `train_lora`, `shutil.rmtree(config.TRAINING_OUTPUT_DIR)`
- **Befund:** Mit `save_strategy="no"` wird dort nichts gespeichert (Ordner ist leer).
- **Lösungsansatz:** Löschen samt Konstante `TRAINING_OUTPUT_DIR`, oder `output_dir` auf einen Temp-Ordner setzen.

### B4 – `del` + `empty_cache()` mit falscher Begründung

**Codestelle:** `app.py:291-296`
```python
# Speicher freigeben, bevor der Konverter startet.
del trainer
del model
del merged_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```
- **Fundstelle:** `train_lora`, Kommentar „bevor der Konverter startet“
- **Befund:** Der Konverter läuft auf der CPU. Ohne `gc.collect()` ist die Freigabe zudem nicht garantiert.
  Sinnvoll ist sie eher für das spätere Laden in Ollama.
- **Lösungsansatz:** Kommentar korrigieren und `gc.collect()` ergänzen, oder entfernen.

### B5 – Parameter, die dem Standard entsprechen

**Codestellen:** `app.py:184`, `app.py:204-205`, `app.py:223`, `app.py:243-245`
```python
trust_remote_code=False,              # app.py:184, 205
low_cpu_mem_usage=True,               # app.py:204
bias="none",                          # app.py:223
use_cpu=not use_gpu,                  # app.py:243
dataloader_pin_memory=use_gpu,        # app.py:244
remove_unused_columns=False,          # app.py:245
```
- **Fundstelle:** `from_pretrained(trust_remote_code=False, low_cpu_mem_usage=True)`, `LoraConfig(bias="none")`,
  `TrainingArguments(use_cpu, dataloader_pin_memory, remove_unused_columns=False)`
- **Befund:** Weitgehend Standardwerte. `remove_unused_columns=False` ist vermutlich ebenfalls entbehrlich (nicht verifiziert).
- **Lösungsansatz:** Je Parameter prüfen und nur behalten, wenn er dokumentierend/sicherheitsrelevant sein soll
  (`trust_remote_code=False` bewusst als Sicherheitsaussage behalten).

### B6 – Wirkungsloses Kürzen / `eos or ""` in `TextDataset`

**Codestellen:** `app.py:121`, `app.py:133-134`, `app.py:187-188`
```python
eos = tokenizer.eos_token or ""                                        # app.py:121
input_ids = (prompt_ids + answer_ids)[: config.MAX_LENGTH]             # app.py:133
labels = ([-100] * len(prompt_ids) + answer_ids)[: config.MAX_LENGTH]  # app.py:134
```
- **Fundstelle:** `input_ids = (...)[: config.MAX_LENGTH]`, `eos = tokenizer.eos_token or ""`
- **Befund:** Nach dem Kürzen des Prompts ist die Länge ohnehin ≤ `MAX_LENGTH`.
  `train_lora` bricht bei fehlendem EOS vorher ab.
- **Lösungsansatz:** Kürzungen/Fallback entfernen oder bewusst als Absicherung kommentieren.

### B7 – Pad-Token-Fallback

**Codestelle:** `app.py:189-190`
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```
- **Fundstelle:** `if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token`
- **Befund:** Beim aktuellen Tokenizer toter Code; als generische Absicherung für andere Modelle aber sinnvoll.
- **Lösungsansatz:** Behalten (Modelle sind austauschbar) und kurz kommentieren.

### B8 – Eigene Zielmodul-Prüfung

**Codestellen:** `app.py:155-163`, `app.py:215`
```python
def ensure_model_compatible(model):          # app.py:155
...
ensure_model_compatible(model)               # app.py:215
```
- **Fundstelle:** `ensure_model_compatible`
- **Befund:** PEFT wirft bei fehlenden Zielmodulen selbst einen Fehler; die eigene Funktion liefert nur eine verständlichere Meldung.
- **Lösungsansatz:** Behalten (bessere Meldung bei Modellwechsel) oder entfernen.

### B9 – Prompt-Fallback ohne Chat-Template

**Codestelle:** `app.py:100-111`
```python
# app.py:107-111
# Fallback für Tokenizer ohne Chat-Template.
return (
    f"System: {config.SYSTEM_PROMPT}\n"
    f"User: {text}\n"
    "Assistant:"
)
```
- **Fundstelle:** `build_prompt`, Zweig `System: … User: … Assistant:`
- **Befund:** Passt zu keinem Ollama-Template; ein Modell ohne Template wäre nach dem Import inkonsistent.
- **Lösungsansatz:** Statt Fallback einen klaren Fehler auslösen oder das Template im Modelfile explizit setzen.

### B10 – `SYSTEM` im Modelfile

**Codestelle:** `app.py:375-379`
```python
config.MODELFILE.write_text(
    f'''FROM ./{config.GGUF_FILE.name}\n\n'''
    f'''PARAMETER temperature {config.TEMPERATURE}\n\n'''
    f'''SYSTEM """{config.SYSTEM_PROMPT}"""\n''',
```
- **Fundstelle:** `import_into_ollama`
- **Befund:** Für den App-Ablauf redundant, weil `ask_ollama` den System-Prompt bei jeder Anfrage mitsendet.
  Nützlich bleibt es für die Nutzung über die Ollama-CLI.
- **Lösungsansatz:** Bewusst entscheiden und dokumentieren.

### B11 – Doppelte Ollama-URL

**Codestellen:** `app.py:418`, `config.py:16`
```python
# app.py:418
response = requests.get("http://localhost:11434/api/tags", timeout=5)
# config.py:16
OLLAMA_URL = "http://localhost:11434/api/chat"
```
- **Fundstelle:** `environment_check`: `"http://localhost:11434/api/tags"`
- **Befund:** Dupliziert den Host aus `config.OLLAMA_URL`.
- **Lösungsansatz:** Basis-URL in `config.py` definieren (`OLLAMA_HOST`) und beide Endpunkte daraus ableiten.

### B12 – Magische Zahlen im Code

**Codestellen:** `app.py:87`, `app.py:240`, `app.py:268`, `app.py:418`
```python
timeout=180,               # app.py:87
logging_steps=5,           # app.py:240
max_shard_size="4GB",      # app.py:268
timeout=5                  # app.py:418
```
- **Fundstelle:** `timeout=180`, `timeout=5`, `logging_steps=5`, `max_shard_size="4GB"`
- **Lösungsansatz:** In `config.py` auslagern.

---

## C. Effizienz und Design

### C1 – Padding auf feste Länge

**Codestellen:** `app.py:137-140`, `config.py:40`
```python
padding = config.MAX_LENGTH - len(input_ids)
input_ids += [tokenizer.pad_token_id] * padding
attention_mask += [0] * padding
labels += [-100] * padding
```
- **Priorität / Aufwand:** Niedrig / M
- **Befund:** Alle Beispiele werden auf `MAX_LENGTH` aufgefüllt.
- **Lösungsansatz:** Dynamisches Padding per Data-Collator (`DataCollatorForSeq2Seq` o. ä.) spart Rechenzeit.

### C2 – Modell erst auf CPU, dann auf GPU

**Codestellen:** `app.py:200-210`
```python
model = AutoModelForCausalLM.from_pretrained(...)    # app.py:200-206, lädt auf die CPU
# Modell explizit auf die GPU legen.
if use_gpu:
    model = model.to("cuda")                         # app.py:210
```
- **Priorität / Aufwand:** Mittel / S
- **Befund:** `from_pretrained(...)` lädt in den RAM, danach folgt `.to("cuda")`.
- **Lösungsansatz:** `device_map="cuda"` direkt beim Laden. Spart RAM und Ladezeit.

### C3 – Hoher Plattenverbrauch (ca. 18 GB)

**Codestellen:** `app.py:262-270` (merged_model), `app.py:334-362` (GGUF), `config.py:31`
```python
# config.py:31
GGUF_OUTTYPE = "f16"
# app.py:352-353
"--outtype",
config.GGUF_OUTTYPE,
```
- **Priorität / Aufwand:** Mittel / M
- **Befund:** `merged_model/` (ca. 6 GB) + GGUF (ca. 6 GB) + Ollama-Kopie (ca. 6 GB).
- **Lösungsansätze:**
  1. `merged_model/` nach erfolgreicher Konvertierung löschen.
  2. Quantisieren (`q8_0` / `q4_K_M`) statt `f16`.
  3. Nur den Adapter konvertieren und per `ADAPTER` im Modelfile nutzen (ohne Merge).

### C4 – `train_lora()` macht zu viel

**Codestellen:** `app.py:166-303` (`train_lora`), `app.py:298-299`, `app.py:447-488` (`main`)
```python
# app.py:298-299 – hängt fest am Ende von train_lora
convert_to_gguf()
import_into_ollama()
# app.py:466-467 – Menü kennt nur Punkt 2 für den gesamten Ablauf
elif choice == "2":
    train_lora()
```
- **Priorität / Aufwand:** Mittel / L
- **Befund:** Training, Merge, Konvertierung und Import in einer Funktion. Ein Fehler spät im Ablauf erzwingt das komplette Neutraining.
- **Lösungsansatz:** Aufteilen in Einzelschritte (`train`, `merge`, `convert`, `import`) und im Menü einzeln ausführbar machen
  (z. B. „nur Konvertierung + Import aus vorhandenem `merged_model/`“).

### C5 – Langer System-Prompt in jedem Trainingsbeispiel

**Codestellen:** `config.py:53-ff` (`SYSTEM_PROMPT`), `app.py:93-112` (`build_prompt`), `app.py:123`
```python
# app.py:123 – jedes Beispiel erhält den vollen System-Prompt
prompt = build_prompt(tokenizer, sample["text"])
```
- **Priorität / Aufwand:** Hoch / M
- **Befund:** Der Prompt wiederholt sich in 500 Beispielen, kostet Tokens/Zeit und ist Auslöser von A1.
- **Lösungsansatz:** Beim Fine-Tuning genügt ein kurzer System-Prompt, da das Verhalten gelernt wird.
  Training und Abfrage müssen aber **denselben** Prompt nutzen. Dazu `config.SYSTEM_PROMPT` kürzen.
- **Hinweis:** Danach die Vorher/Nachher-Vergleiche (Menü 4) neu bewerten.

### C6 – Testfunktionen ohne Modellprüfung / Docstrings

**Codestellen:** `app.py:425-444`, `app.py:447`
```python
def get_test_text():            # app.py:425
def test_model(model_name):     # app.py:432
def compare_models():           # app.py:439
def main():                     # app.py:447
```
- **Priorität / Aufwand:** Niedrig / S
- **Befund:** `test_model` und `compare_models` prüfen nicht, ob das Modell existiert;
  `get_test_text`, `test_model`, `compare_models`, `main` haben keine Docstrings.
- **Lösungsansatz:** Vorab `GET /api/tags` auswerten und eine verständliche Meldung ausgeben
  („Modell X fehlt, bitte `ollama pull X`“).

---

## Arbeitsweise

1. Einen Punkt wählen und Checkbox in der Tabelle oben auf `[x]` setzen, sobald erledigt.
2. Jede Änderung klein halten und danach prüfen (z. B. Menü 5 „Systemcheck“, Menü 4 „Vorher/Nachher“).
3. Bei Änderungen an Prompt, `MAX_LENGTH` oder Datenaufteilung (A1, A2, C5) das Training neu starten.
4. Bei Änderungen an `config.py` (A9, B11, B12) auch `APP_ERKLAERUNG.md` aktualisieren.
