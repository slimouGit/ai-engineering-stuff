# Projektuebersicht: `bamf_lora_pycharm`

Diese Datei listet alle wichtigen Dateien/Ordner und Klassen im Projekt auf und beschreibt ihren Zweck.

## 1) Dateien und Ordner im Projekt

### Root-Ordner

- `app.py`
  - Hauptanwendung (CLI-Menue).
  - Laedt Trainingsdaten, trainiert LoRA-Adapter, testet Basismodell/feinjustiertes Modell und zeigt Trainingsstatus.

- `requirements.txt`
  - Python-Abhaengigkeiten fuer das Projekt.

- `README.md`
  - Kurze Nutzungserklaerung der App.

- `training_info.json` (wird nach Training erzeugt/aktualisiert)
  - Metadaten des letzten Trainings:
    - Zeitpunkt
    - Basismodell
    - Anzahl Trainingsbeispiele
    - Epochen
    - Loss
    - Laufzeit

- `Modelfile` (Legacy/optional)
  - Historischer Rest fuer Ollama-Adapter-Import.
  - Fuer den aktuellen lokalen PEFT-Workflow nicht notwendig.

### Datenordner

- `data/text_samples.csv`
  - Aktiver Trainingsdatensatz (`text,label`).
  - Wird in `app.py` als Quelle fuer das LoRA-Training genutzt.

- `data/train.json`
  - Aelterer Datensatz aus frueherem App-Stand.
  - Wird im aktuellen `app.py` nicht mehr verwendet.

### Modell-/Artefaktordner

- `adapter/`
  - Gespeicherter LoRA-Adapter + Tokenizer-Dateien.
  - Detaillierte Inhalte:
    - `adapter_model.safetensors`
      - Die trainierten LoRA-Gewichte (nur Delta zum Basismodell, nicht das komplette LLM).
      - Ohne diese Datei kann `load_trained_model()` kein trainiertes Modell laden.
    - `adapter_config.json`
      - Technische LoRA-Konfiguration des gespeicherten Adapters.
      - Enthält u. a.:
        - `base_model_name_or_path` (hier: `Qwen/Qwen2.5-0.5B-Instruct`)
        - `peft_type` (`LORA`)
        - `r`, `lora_alpha`, `lora_dropout`
        - `target_modules` (hier `q_proj`, `v_proj`)
      - Muss zur Architektur des Basismodells passen.
    - `tokenizer.json`
      - Vollständiges serialisiertes Tokenizer-Modell.
      - Wird beim Laden des trainierten Modells wiederverwendet.
    - `tokenizer_config.json`
      - Tokenizer-Metadaten und Sondertoken-Konfiguration.
      - In deinem Stand z. B.:
        - `tokenizer_class: Qwen2Tokenizer`
        - `eos_token:  <|im_end|>`
        - `pad_token:  <|endoftext|>`
        - `model_max_length: 131072`
    - `chat_template.jinja`
      - Chat-Template des Tokenizers (Nachrichtenformat mit Rollen wie `system`, `user`, `assistant`).
      - Wichtig, wenn man `apply_chat_template(...)` nutzt oder kompatibles Prompt-Format sicherstellen will.
    - `README.md`
      - Automatisch generierte Model-Card (PEFT/HF-Standard).
      - Beschreibt Basisinfos zum Adapter, aber enthält oft Platzhalter (`[More Information Needed]`).

- `training_output/`
  - Ausgabeordner des `Trainer`-Runs.
  - Kann je nach Training leer bleiben oder Logs/States enthalten.

### Technische Hilfsordner

- `__pycache__/`
  - Automatisch von Python erzeugte Bytecode-Dateien.

- `.pytest_cache/`
  - Cache von `pytest` (falls Tests ausgefuehrt wurden).

## 2) Klassen in `app.py`

- `TextDataset(Dataset)`
  - Aufgabe:
    - Wandelt CSV-Zeilen in tokenisierte Trainingsbeispiele um.
    - Baut pro Beispiel:
      - `input_ids`
      - `attention_mask`
      - `labels` (mit `-100` fuer Prompt-Teile, damit nur die Zielantwort gelernt wird)
  - Wird in `train_lora()` dem HuggingFace `Trainer` uebergeben.

## 3) Wichtige Funktionen in `app.py`

- `load_samples()`
  - Laedt und validiert `data/text_samples.csv`.

- `make_prompt(text)`
  - Baut den Prompt fuer das Modell (System/User/Assistant-Format).

- `normalize_label(text)`
  - Mappt Modellantwort auf Klassen:
    - `relevant`
    - `nicht_relevant`
    - sonst `unsicher`

- `load_base_model()`
  - Laedt Basismodell + Tokenizer und cached sie im Speicher.

- `load_trained_model()`
  - Laedt Basismodell + LoRA-Adapter aus `adapter/` und cached sie.

- `classify_text(model, tokenizer, text)`
  - Fuehrt Inferenz aus und gibt normalisiertes Klassenlabel zurueck.

- `train_lora()`
  - Trainiert LoRA auf den CSV-Daten und schreibt Adapter + `training_info.json`.

- `test_base_model()`
  - Testet Klassifikation mit Basismodell.

- `test_trained_model()`
  - Testet Klassifikation mit trainiertem Adapter.

- `show_training_status()`
  - Zeigt Inhalte aus `training_info.json`.

- `main()`
  - CLI-Menue-Steuerung der App.

## 4) Aktueller technischer Ablauf

1. Basismodell waehlen (`HF_BASE_MODEL` in `app.py`).
2. Trainingsdaten aus `data/text_samples.csv` laden.
3. LoRA trainieren (`train_lora()`).
4. Adapter in `adapter/` speichern.
5. Spaeter Basismodell oder trainiertes Modell fuer Klassifikation testen.
