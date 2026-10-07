# Lokale Textklassifikation mit Qwen + LoRA

Diese App ist ein bewusst kleines Lernbeispiel fuer lokalen LLM-Feinschliff mit LoRA.

Sie macht genau vier Dinge:

1. Basismodell testen
2. LoRA trainieren (auf `data/text_samples.csv`)
3. Trainiertes Modell testen
4. Trainingsstatus anzeigen

## Ziel

Die App soll zeigen, wie ein kleines lokales LLM mit eigenen gelabelten Texten fuer eine einfache Klassifikation genutzt wird.

Ausgabe-Klassen:

- `relevant`
- `nicht_relevant`
- `unsicher` (Fallback, wenn kein klares Label im Modelloutput erkannt wurde)

---

## Projektaufbau

```text
bamf_lora_pycharm/
├── app.py
├── README.md
├── requirements.txt
├── training_info.json            # wird nach Training geschrieben
├── adapter/                      # wird nach Training geschrieben
├── training_output/              # Trainer-Artefakte
└── data/
    └── text_samples.csv          # Trainingsdaten (text,label)
```

---

## Schnellstart

```bash
pip install -r requirements.txt
python app.py
```

---

## Menue und Ablauf

Beim Start von `app.py` laeuft `main()` und zeigt:

```text
1 - Basismodell testen
2 - LoRA trainieren
3 - Trainiertes Modell testen
4 - Trainingsstatus anzeigen
0 - Ende
```

## Option 1: Basismodell testen

### Interner Ablauf

- `test_base_model()`
- `input_text()`
- `load_base_model()`
- `classify_text(model, tokenizer, text)`
- `normalize_label(decoded_text)`

### Was passiert

1. Du gibst einen Testtext ein.
2. Das Basismodell (`HF_BASE_MODEL`) wird geladen (nur beim ersten Mal, danach Cache).
3. Aus dem Text wird mit `make_prompt()` ein Prompt gebaut.
4. Das Modell generiert einen sehr kurzen Output (`max_new_tokens=4`).
5. Der Output wird auf ein Klassenlabel gemappt.

---

## Option 2: LoRA trainieren

### Interner Ablauf

- `train_lora()`
- `load_samples()`
- `TextDataset(rows, tokenizer)`
- `Trainer(...).train()`
- `save_pretrained(ADAPTER_DIR)`
- `training_info.json` schreiben

### Was passiert im Detail

1. **CSV laden und validieren**
   - Quelle: `data/text_samples.csv`
   - Erwartung: `text,label`
   - Nur Labels `relevant` und `nicht_relevant` sind gueltig.

2. **Tokenizer und Basismodell laden**
   - Modellname kommt aus `HF_BASE_MODEL`.

3. **LoRA aufsetzen**
   - Konfiguration in `LoraConfig(...)`:
     - `r=8`
     - `lora_alpha=16`
     - `lora_dropout=0.05`
     - `target_modules=["q_proj", "v_proj"]`

4. **Trainingsbeispiele aufbereiten**
   - Klasse: `TextDataset`
   - Prompt + Ziellabel werden tokenisiert.
   - Prompt-Tokens bekommen `labels=-100` (kein Loss auf Prompt, nur auf Zielantwort).

5. **Training starten**
   - Hyperparameter ueber `TrainingArguments`.
   - Standard aktuell:
     - `num_train_epochs=TRAIN_EPOCHS`
     - `per_device_train_batch_size=1`
     - `gradient_accumulation_steps=4`

6. **Ergebnisse speichern**
   - `adapter/` (LoRA-Gewichte + Tokenizerdateien)
   - `training_info.json` (Metadaten wie Zeit, Loss, Laufzeit)

---

## Option 3: Trainiertes Modell testen

### Interner Ablauf

- `test_trained_model()`
- `load_trained_model()`
- `classify_text(...)`

### Was passiert

1. Es wird geprueft, ob `adapter/adapter_model.safetensors` existiert.
2. Wenn nicht: Hinweis `Bitte zuerst trainieren.`
3. Wenn ja:
   - Basismodell laden
   - LoRA-Adapter drueberladen (`PeftModel.from_pretrained`)
   - Text klassifizieren wie beim Basismodell

---

## Option 4: Trainingsstatus anzeigen

### Interner Ablauf

- `show_training_status()`

### Was passiert

1. `training_info.json` wird gelesen.
2. Folgende Felder werden ausgegeben:
   - `trained`
   - `trained_at`
   - `training_examples`
   - `epochs`
   - `train_loss`
   - `runtime_seconds`

---

## Wichtige Funktionen in `app.py`

- `load_samples()`
  - Liest und validiert den CSV-Datensatz.

- `make_prompt(text)`
  - Baut das Prompt-Format fuer System/User/Assistant.

- `normalize_label(text)`
  - Reduziert freien Modelltext auf Label.

- `load_base_model()`
  - Laedt Basismodell + Tokenizer mit Cache.

- `load_trained_model()`
  - Laedt Basismodell + gespeicherten LoRA-Adapter.

- `classify_text(...)`
  - Fuehrt Inferenz aus und gibt Label zurueck.

- `TextDataset`
  - Wandelt Rohdaten in Trainer-kompatible Tensoren.

- `train_lora()`
  - Kompletter Trainingspfad inkl. Speichern der Artefakte.

---

## Datenformat der CSV

Datei: `data/text_samples.csv`

```csv
text,label
Ich hatte Kontakt ...,relevant
Ich moechte einen Termin ...,nicht_relevant
```

Regeln:

- Header muss `text,label` sein.
- Label nur: `relevant` oder `nicht_relevant`.

---

## Konfiguration in `app.py`

- `HF_BASE_MODEL`
  - HuggingFace-Basismodell, z. B. `Qwen/Qwen2.5-0.5B-Instruct`
- `TRAIN_EPOCHS`
  - Anzahl Trainingsdurchlaeufe
- `MAX_LENGTH`
  - Maximale Sequenzlaenge
- `SYSTEM_PROMPT`
  - Instruktion fuer die Klassifikationsantwort

---

## Typische Fehler und Loesungen

### Fehler: "Kein trainiertes Modell gefunden. Bitte zuerst LoRA trainieren."

- Ursache: Adapter fehlt.
- Loesung: Menuepunkt `2` ausfuehren.

### Fehler: Modell wird nicht gefunden / Downloadfehler

- Ursache: falscher `HF_BASE_MODEL` oder Verbindungsproblem.
- Loesung: Modellname pruefen, Internet pruefen, Pakete aktualisieren.

### Problem: Klassifikation schlecht

- Moegliche Ursachen:
  - zu wenig Daten
  - uneinheitliche Labels
  - zu wenige Epochen
- Loesung:
  - bessere/mehr Daten
  - Labelqualitaet pruefen
  - `TRAIN_EPOCHS` schrittweise erhoehen

---

## Kurzfazit

Diese App ist ein schlankes, praktisches Beispiel fuer lokalen LoRA-Feinschliff auf CPU:

- Datensatz einlesen
- LoRA trainieren
- Adapter wiederladen
- Klassifikation mit Basismodell vs. trainiertem Modell vergleichen
