# app.py – Funktionsweise Schritt für Schritt

Dieses Dokument erklärt die Datei `app.py` so, dass man sie auch ohne Vorwissen über
KI-Training nachvollziehen kann. Es folgt dem Code von oben nach unten und
beschreibt danach den kompletten Ablauf der Anwendung.

> Hinweis: Alle Texte im Projekt sind **synthetisch** (ausgedacht). Das Projekt ist ein Lernbeispiel.

---

## 1. Worum geht es?

Die Anwendung bringt einem kleinen Sprachmodell (beliebig austauschbar, siehe `config.py`) bei,
kurze Texte in genau eine von zwei Klassen einzuordnen:

| Klasse | Bedeutung |
|---|---|
| `relevant` | Der Text beschreibt konkrete eigene Kontakte, Unterstützung oder Beteiligung (z. B. „Ich übermittelte Nachrichten zwischen Mitgliedern“). |
| `nicht_relevant` | Keine konkrete eigene Beteiligung (z. B. „Ich kenne den Namen der Gruppe nur aus den Nachrichten.“). |

Das Modell wird nicht komplett neu trainiert (das wäre für eine 12-GB-Grafikkarte zu groß),
sondern mit **LoRA** nur leicht „nachjustiert“. Danach wird das Ergebnis in
**Ollama** bereitgestellt, damit man es bequem abfragen kann.

### Begriffe in einfachen Worten

| Begriff | Erklärung |
|---|---|
| **Basismodell** | Das fertige, allgemeine Modell, z. B. ein Modell von Hugging Face (liegt als Dateien im Ordner `models`). Welches Modell genutzt wird, bestimmt `LOCAL_TRAIN_MODEL`. |
| **LoRA** | „Low-Rank Adaptation“. Statt alle Milliarden Gewichte zu ändern, werden kleine Zusatz-Matrizen trainiert. Das ist schnell und braucht wenig Speicher. |
| **Adapter** | Das Ergebnis des LoRA-Trainings: eine kleine Datei mit den Zusatzgewichten. |
| **Mergen** | Adapter und Basismodell werden zu **einem** Modell verschmolzen. |
| **GGUF** | Ein Dateiformat, das Ollama / llama.cpp lesen können. Ollama kann die Trainings-Dateien (Safetensors) dieses Modells nicht direkt laden. |
| **Ollama** | Lokaler Modell-Server (Port 11434), den die App per HTTP anspricht. |
| **Prompt** | Der Eingabetext an das Modell (System-Anweisung + Nutzertext). |
| **Token** | Ein Textstück, mit dem das Modell rechnet (Wort oder Wortteil). |
| **Epoche** | Ein kompletter Durchlauf durch alle Trainingsbeispiele. |

---

## 2. Beteiligte Dateien

```
bamf_lora_clean/
├─ app.py                  Hauptprogramm (Menü, Training, Konvertierung, Test)
├─ config.py               Alle Einstellungen an einer Stelle
├─ data/text_samples.csv   Trainingsdaten (Spalten: text,label)
├─ models/<modellname>/    Basismodell (Safetensors, von Hugging Face)
├─ adapter/                Ergebnis: LoRA-Adapter
├─ merged_model/           Ergebnis: Basismodell + Adapter verschmolzen
├─ training_output/        Temporärer Ordner des Trainers
├─ training_info.json      Protokoll des letzten Trainings
├─ llama.cpp/              Werkzeug zur GGUF-Konvertierung (wird bei Bedarf geklont)
├─ trained-model.gguf      Ergebnis: Modell im GGUF-Format
└─ Modelfile               Bauanleitung für Ollama
```

### Die wichtigsten Einstellungen in `config.py`

| Einstellung | Wert | Bedeutung |
|---|---|---|
| `LOCAL_TRAIN_MODEL` | `models/<modellname>` | Basismodell fürs Training |
| `OLLAMA_BASE_MODEL` | `<name>:<tag>` | Vergleichsmodell **in Ollama** (sollte dem Trainingsmodell entsprechen und muss separat per `ollama pull <name>:<tag>` installiert sein) |
| `OLLAMA_TRAINED_MODEL` | `trained-modell` | Name des neuen Modells in Ollama |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Adresse der Ollama-Schnittstelle |
| `EPOCHS` | 3 | Anzahl Durchläufe durch die Daten |
| `BATCH_SIZE` | 1 | Beispiele pro Schritt |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | Es werden 4 Schritte gesammelt, bevor gelernt wird (wirkt wie Batch 4, spart Speicher) |
| `LEARNING_RATE` | 2e-4 | Wie stark pro Schritt gelernt wird |
| `MAX_LENGTH` | 192 | Maximale Tokenzahl pro Trainingsbeispiel |
| `LORA_R` / `LORA_ALPHA` | 8 / 16 | Größe und Stärke des LoRA-Adapters |
| `LORA_TARGET_MODULES` | `q_proj`, `v_proj` | Welche Schichten des Modells angepasst werden |
| `TEMPERATURE` | 0 | 0 = immer die wahrscheinlichste Antwort, keine Zufälligkeit |
| `MAX_NEW_TOKENS` | 12 | Die Antwort darf höchstens 12 Tokens lang sein |
| `SYSTEM_PROMPT` | langer Text | Anweisung an das Modell: nur `relevant` oder `nicht_relevant` antworten |

---

## 3. Gesamtablauf auf einen Blick

```
                    python app.py
                         │
                         ▼
                  ┌─────────────┐
                  │    Menü     │◄─────────────────────────────┐
                  └──────┬──────┘                              │
   ┌──────┬──────┬───────┼───────┬────────┐                    │
   ▼      ▼      ▼       ▼       ▼        ▼                    │
  [1]    [2]    [3]     [4]     [5]      [0]                   │
 Basis- Training Trainier-  Vorher/  System-  Ende             │
 modell  (siehe  tes Modell Nachher  check                     │
 testen  unten)  testen    vergleichen                         │
   └──────┴──────┴───────┴───────┘                             │
                         └─────────────────────────────────────┘
```

Menüpunkt **2** ist der große Ablauf:

```
CSV laden ──► Modell laden ──► LoRA anbringen ──► Trainieren ──► Adapter speichern
                                                                       │
Ollama-Modell ◄── Modelfile ◄── GGUF erzeugen ◄── Mergen & speichern ◄─┘
   erstellt         schreiben      (llama.cpp)
```

---

## 4. Der Code im Detail

### 4.1 Kopf der Datei (Zeilen 1–40)

```python
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
```
Die Bibliothek `transformers` würde sonst versuchen, TensorFlow/Flax zu laden, falls
diese installiert sind. Das kann Fehler verursachen. Mit diesen zwei Zeilen wird
ausdrücklich nur **PyTorch** verwendet. Sie stehen **vor** den Importen, damit sie
rechtzeitig wirken.

Danach werden Bibliotheken geladen:

| Import | Wofür |
|---|---|
| `csv`, `json` | CSV lesen, Trainingsprotokoll schreiben |
| `shutil` | Ordner löschen, Programme im PATH finden (`which`) |
| `subprocess` | Externe Programme starten (`git`, `ollama`, Konverter) |
| `requests` | HTTP-Anfragen an Ollama |
| `torch` | PyTorch, die Rechenbasis für das Training |
| `peft` | Stellt LoRA bereit |
| `transformers` | Lädt Modell/Tokenizer und enthält den `Trainer` |
| `config` | Eigene Einstellungen |

`VALID_LABELS = {"relevant", "nicht_relevant"}` ist die Liste der erlaubten Antworten.

---

### 4.2 `load_samples()` – Trainingsdaten einlesen

**Aufgabe:** Die CSV-Datei lesen und prüfen, ob sie brauchbar ist.

Schritte:
1. Existiert `data/text_samples.csv`? Wenn nicht → Fehler.
2. Datei öffnen mit `utf-8-sig` (verträgt Umlaute und ein evtl. vorhandenes Windows-Zeichen am Dateianfang).
3. Gibt es die Spalten `text` und `label`? Wenn nicht → Fehler.
4. Jede Zeile durchgehen (`start=2`, weil Zeile 1 die Kopfzeile ist – so stimmen die Zeilennummern in Fehlermeldungen mit dem Editor überein):
   - Text und Label bereinigen (`strip()`, Label klein schreiben).
   - Leerer Text → Fehler mit Zeilennummer.
   - Label nicht `relevant`/`nicht_relevant` → Fehler mit Zeilennummer.
5. Gültige Zeilen werden als `{"text": ..., "label": ...}` gesammelt.
6. Keine Beispiele → Fehler.

**Ergebnis:** eine Liste mit (aktuell) 500 Beispielen.

Beispiel einer Zeile:
```
Ich übermittelte Nachrichten zwischen mehreren Mitgliedern der Organisation.,relevant
```

---

### 4.3 `ask_ollama(model_name, text)` – Ollama befragen

**Aufgabe:** Einen Text an ein Modell in Ollama schicken und die Antwort zurückgeben.

Es wird ein HTTP-POST an `http://localhost:11434/api/chat` gesendet:

```json
{
  "model": "<Modellname>",
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    {"role": "user",   "content": "<Text>"}
  ],
  "stream": false,
  "options": {"temperature": 0, "num_predict": 12}
}
```

- `system` = die Regeln für das Modell, `user` = der zu prüfende Text.
- `stream: false` = die Antwort kommt komplett auf einmal, nicht Wort für Wort.
- `num_predict: 12` begrenzt die Antwortlänge.
- `timeout=180` = nach 3 Minuten ohne Antwort wird abgebrochen.
- `raise_for_status()` löst bei HTTP-Fehlern (z. B. **404 = Modell nicht vorhanden**) eine Ausnahme aus.
- Zurückgegeben wird `response.json()["message"]["content"]` ohne Leerzeichen am Rand.

---

### 4.4 `build_prompt(tokenizer, text)` – Eingabe für das Training bauen

**Aufgabe:** Systemanweisung und Text so zusammensetzen, wie das Modell es gewohnt ist.

Jedes Modell hat ein **Chat-Template**, also ein festes Format mit Sondermarkierungen,
das Rollen trennt. Das Format unterscheidet sich je nach Modell. Ein häufiges Beispiel (ChatML-Stil) sieht sinngemäß so aus:

```
<|im_start|>system
...Anweisung...<|im_end|>
<|im_start|>user
...Text...<|im_end|>
<|im_start|>assistant
```

- Hat der Tokenizer ein Template, wird `apply_chat_template(..., add_generation_prompt=True)` genutzt. Das Flag hängt den Beginn der Assistentenantwort an – genau dort soll das Modell weiterschreiben.
- Hat er keins, wird ein einfacher Fallback verwendet (`System: … User: … Assistant:`).

Wichtig: Beim Training und bei Ollama muss dasselbe Format gelten, sonst lernt das
Modell etwas anderes, als später abgefragt wird.

---

### 4.5 `TextDataset` – Trainingsbeispiele in Zahlen verwandeln

Das Modell versteht keinen Text, nur Zahlen (Token-IDs). Diese Klasse wandelt
jedes Beispiel um. Sie ist das Herzstück der Datenvorbereitung.

Für **jedes** Beispiel passiert Folgendes:

1. **Prompt bauen:** `build_prompt(...)` (siehe 4.4).
2. **Antwort bauen:** Label + EOS-Token, z. B. `relevant<EOS>`. Das EOS („End of Sequence“) bringt dem Modell bei, nach dem Label **aufzuhören**.
3. **Tokenisieren:** Prompt und Antwort werden in Token-IDs umgewandelt (`add_special_tokens=False`, weil das Template die Sondertokens schon enthält).
4. **Länge sichern:** Die Gesamtlänge darf `MAX_LENGTH` (192) nicht überschreiten. Ist der Prompt zu lang, wird er **von vorn** gekürzt (`prompt_ids[-max_prompt_len:]`), damit die Antwort immer vollständig bleibt.
5. **Drei Listen erzeugen:**

| Liste | Inhalt | Zweck |
|---|---|---|
| `input_ids` | Prompt-Tokens + Antwort-Tokens | Eingabe ans Modell |
| `labels` | `-100` für den Prompt, echte IDs für die Antwort | Was das Modell vorhersagen soll |
| `attention_mask` | `1` für echte Tokens, `0` für Auffüllung | Welche Positionen zählen |

6. **Auffüllen (Padding):** Alle Beispiele werden auf exakt 192 Tokens gebracht.
   `input_ids` bekommt das Pad-Token, `attention_mask` bekommt `0`, `labels` bekommt `-100`.
7. Alles wird zu PyTorch-Tensoren (`torch.long`).

**Warum `-100`?** Dieser Wert bedeutet „bei der Fehlerberechnung ignorieren“.
So wird das Modell **nur für die Antwort** (`relevant` / `nicht_relevant`) bewertet
und nicht dafür, die Anweisung oder den Eingabetext nachzusprechen.

Veranschaulicht:

```
Position:    [ Prompt-Tokens ............ ][ relevant ][EOS][ Padding ]
input_ids:   [ 151644 872 ... 77091       ][  ids    ][ids ][ pad ... ]
labels:      [ -100  -100 ... -100        ][  ids    ][ids ][ -100 .. ]
attn_mask:   [ 1 1 1 ...................  ][  1 1    ][ 1  ][ 0 0 0 .. ]
```

`__len__` liefert die Anzahl der Beispiele, `__getitem__` das Beispiel Nr. *i*.
Das braucht der Trainer, um die Daten zu durchlaufen.

---

### 4.6 `ensure_model_compatible(model)` – Sicherheitsnetz

**Aufgabe:** Prüfen, ob die in `LORA_TARGET_MODULES` genannten Schichten (`q_proj`, `v_proj`)
im Modell wirklich vorkommen.

- Es werden alle Modulnamen gesammelt (jeweils der letzte Namensteil).
- Fehlt ein Zielmodul, kommt eine verständliche Fehlermeldung statt eines kryptischen
  Fehlers später. Das passiert z. B. beim Wechsel auf ein Modell mit anderen Schichtnamen.

`q_proj` und `v_proj` sind Teile des **Attention-Mechanismus** (Query und Value) –
dort greift LoRA an.

---

### 4.7 `train_lora()` – der Hauptablauf (Menüpunkt 2)

Diese Funktion führt alles nacheinander aus.

#### Schritt A: Vorbereitung
1. Existiert der Modellordner? Sonst Fehler.
2. `load_samples()` → Trainingsdaten laden.
3. Infos ausgeben (Modellpfad, Anzahl Beispiele, Epochen).

#### Schritt B: Tokenizer laden
- `AutoTokenizer.from_pretrained(..., local_files_only=True)` lädt den Tokenizer **nur lokal** (kein Internet).
- `trust_remote_code=False` verbietet das Ausführen fremden Codes aus Modellordnern (Sicherheit).
- Fehlt ein EOS-Token → Abbruch. Fehlt ein Pad-Token → das EOS-Token wird ersatzweise genutzt.

#### Schritt C: Hardware prüfen
- `torch.cuda.is_available()` prüft, ob eine NVIDIA-GPU nutzbar ist.
- Ausgabe: `GPU wird verwendet: NVIDIA GeForce RTX 3060` oder Hinweis auf CPU.

#### Schritt D: Modell laden
```python
AutoModelForCausalLM.from_pretrained(
    ..., dtype=float16 (GPU) / float32 (CPU), low_cpu_mem_usage=True)
```
- `float16` halbiert den Speicherbedarf (wichtig bei 12 GB VRAM); auf der CPU wird `float32` genutzt.
- `low_cpu_mem_usage=True` lädt sparsam in den Arbeitsspeicher.
- Das Laden zeigt den Balken „Loading checkpoint shards 1/2, 2/2“ – das Modell liegt in **zwei** Dateien.
- Danach `model.to("cuda")`: das Modell wird auf die Grafikkarte verschoben.
- `use_cache = False`: Der Zwischenspeicher fürs Generieren wird fürs Training abgeschaltet (spart Speicher, nötig fürs Training).
- `pad_token_id` wird im Modell eingetragen.

#### Schritt E: Kompatibilität prüfen
`ensure_model_compatible(model)` (siehe 4.6).

#### Schritt F: LoRA anbringen
```python
LoraConfig(task_type=CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
           target_modules=["q_proj","v_proj"], bias="none")
model = get_peft_model(model, lora_config)
```
| Parameter | Bedeutung |
|---|---|
| `r=8` | „Rang“: Größe der Zusatzmatrizen. Größer = ausdrucksstärker, aber mehr Speicher. |
| `lora_alpha=16` | Skalierung: wie stark der Adapter gewichtet wird (üblich: 2 × r). |
| `lora_dropout=0.05` | Zufälliges Weglassen von 5 %, gegen Überanpassung. |
| `bias="none"` | Bias-Werte werden nicht trainiert. |

Das Basismodell wird **eingefroren**; nur die kleinen LoRA-Matrizen sind trainierbar.
`print_trainable_parameters()` zeigt, dass das nur einen winzigen Bruchteil (Promille) aller Parameter ausmacht.

#### Schritt G: Dataset und Trainer
- `TextDataset(samples, tokenizer)` → Beispiele in Tensoren (siehe 4.5).
- Ein eventuell vorhandener alter Ordner `training_output` wird gelöscht.
- `TrainingArguments` legt fest, wie trainiert wird:

| Argument | Bedeutung |
|---|---|
| `num_train_epochs` | Anzahl Durchläufe (3) |
| `per_device_train_batch_size=1` | Ein Beispiel pro Rechenschritt |
| `gradient_accumulation_steps=4` | Alle 4 Schritte wird gelernt → effektive Batchgröße 4 |
| `learning_rate=2e-4` | Lerntempo |
| `logging_steps=5` | Alle 5 Schritte wird der Fehlerwert (Loss) ausgegeben |
| `save_strategy="no"` | Keine Zwischenstände speichern (spart Platz/Zeit) |
| `report_to="none"` | Kein Online-Tracking (wandb o. Ä.) |
| `use_cpu` | `True`, wenn keine GPU da ist |
| `dataloader_pin_memory` | Beschleunigt Datentransfer zur GPU |
| `remove_unused_columns=False` | Wichtig: Eigene Spalten (`labels` etc.) nicht verwerfen |

Rechenbeispiel: 500 Beispiele ÷ 4 = **125 Lernschritte pro Epoche**, bei 3 Epochen = **375 Schritte**.

#### Schritt H: Training
```python
start = time.time()
result = trainer.train()
runtime = time.time() - start
```
Hier passiert das eigentliche Lernen. Pro Schritt vergleicht das Modell seine Vorhersage
mit dem Label, berechnet den **Loss** (Fehlerwert) und passt die LoRA-Matrizen an.
Ein **sinkender Loss** zeigt, dass das Modell lernt. `result.training_loss` ist der Durchschnitt,
`runtime` die Dauer in Sekunden.

#### Schritt I: Adapter speichern (Teil 1/4 → 2/4)
Der alte Ordner `adapter/` wird gelöscht, dann werden Adapter und Tokenizer gespeichert.
Der Adapter ist klein (wenige MB).

#### Schritt J: Mergen (2/4)
```python
merged_model = model.merge_and_unload()
```
Die LoRA-Gewichte werden fest ins Basismodell **eingerechnet**, und die LoRA-Hülle
wird entfernt. Es entsteht ein normales, eigenständiges Modell. Gespeichert wird in
`merged_model/` als Safetensors, in Stücken von höchstens 4 GB (`max_shard_size`) – daher zwei Dateien.

#### Schritt K: Protokoll schreiben
`training_info.json` enthält z. B.:

```json
{
  "trained": true,
  "trained_at": "2026-10-10T16:19:21",
  "training_examples": 500,
  "epochs": 3,
  "train_loss": 0.0087,
  "runtime_seconds": 379.72,
  "base_model_path": "...\\models\\<modellname>",
  "lora_r": 8,
  "lora_alpha": 16
}
```
Ein `train_loss` nahe 0 bedeutet: Das Modell hat die Trainingsdaten sehr gut gelernt.
(Das sagt noch nichts über unbekannte Texte aus – dafür gibt es die Testfunktionen.)

#### Schritt L: Speicher freigeben
`del trainer`, `del model`, `del merged_model`, `torch.cuda.empty_cache()`.
Der Grafikspeicher wird frei, bevor der nächste Schritt (externe Konvertierung) startet.

#### Schritt M: Konvertieren und importieren
`convert_to_gguf()` und `import_into_ollama()` – siehe unten.
Zum Schluss: „FERTIG. … verfügbar als: trained-modell“.

---

### 4.8 `ensure_llama_cpp()` – Konverter bereitstellen

**Aufgabe:** Sicherstellen, dass `llama.cpp/convert_hf_to_gguf.py` vorhanden ist.

1. Skript vorhanden → fertig.
2. Ordner `llama.cpp` existiert, aber das Skript fehlt → Fehler (Ordner ist unvollständig, bitte löschen).
3. `git` nicht installiert → Fehler.
4. Sonst: `git clone https://github.com/ggml-org/llama.cpp.git` und erneut prüfen.

---

### 4.9 `convert_to_gguf()` – Schritt 3/4

**Aufgabe:** Das gemergte Modell in das GGUF-Format umwandeln.

1. `merged_model/` muss existieren.
2. `ensure_llama_cpp()`.
3. Eine alte `trained-model.gguf` wird gelöscht.
4. Es wird ein Unterprozess gestartet:
   ```
   python llama.cpp/convert_hf_to_gguf.py merged_model --outfile trained-model.gguf --outtype f16
   ```
   - `sys.executable` = derselbe Python-Interpreter wie die App.
   - `f16` = 16-Bit-Genauigkeit (gute Qualität, ca. 6 GB Dateigröße).
   - `check=True` = bei Fehler wird eine Ausnahme ausgelöst.
5. Prüfung: Datei existiert und ist nicht leer.

---

### 4.10 `import_into_ollama()` – Schritt 4/4

**Aufgabe:** Das GGUF in Ollama registrieren.

1. GGUF-Datei und `ollama`-Programm müssen vorhanden sein.
2. Ein **Modelfile** wird geschrieben (die „Bauanleitung“ für Ollama):
   ```
   FROM ./trained-model.gguf

   PARAMETER temperature 0

   SYSTEM """<SYSTEM_PROMPT>"""
   ```
   `FROM` zeigt bewusst auf die GGUF-Datei – **nicht** auf die Safetensors, die Ollama je nach Modellarchitektur nicht direkt importieren kann.
3. `ollama create trained-modell -f Modelfile` erstellt das Modell.
4. `ollama list` wird abgefragt; steht `trained-modell` darin, war es erfolgreich, sonst Fehler.

---

### 4.11 `environment_check()` – Menüpunkt 5

Zeigt eine Übersicht, ohne etwas zu verändern:

| Eintrag | Geprüft wird |
|---|---|
| Python | Version |
| Trainings-CSV | Datei vorhanden? |
| Lokales HF-Modell | Modellordner vorhanden? |
| Git / Ollama CLI | Programme im PATH? |
| llama.cpp Converter | Vorhanden oder „wird bei Bedarf geklont“ |
| Ollama Server | `GET http://localhost:11434/api/tags` erreichbar? |

---

### 4.12 Testfunktionen

- `get_test_text()` fragt einen Text ab. Bei leerer Eingabe (Enter) wird das Beispiel
  „Ich hatte regelmäßig Kontakt zu einer bewaffneten Gruppe.“ verwendet.
- `test_model(model_name)` zeigt Modell, Text und die Antwort von `ask_ollama`.
- `compare_models()` fragt **dasselbe** erst `OLLAMA_BASE_MODEL` („VOR TRAINING“), dann `OLLAMA_TRAINED_MODEL` („NACH LORA-TRAINING“). So sieht man den Effekt des Trainings direkt.

---

### 4.13 `main()` – das Menü

Eine Endlosschleife (`while True`), die das Menü zeigt und die Eingabe auswertet:

| Eingabe | Aktion |
|---|---|
| `1` | `test_model(OLLAMA_BASE_MODEL)` |
| `2` | `train_lora()` |
| `3` | `test_model(OLLAMA_TRAINED_MODEL)` |
| `4` | `compare_models()` |
| `5` | `environment_check()` |
| `0` | Schleife beenden |
| sonst | „Ungültige Auswahl.“ |

Die Aktionen stehen in einem `try`-Block. Fehler brechen die App **nicht** ab, sondern
werden angezeigt und das Menü erscheint erneut:

| Fehlerart | Ausgabe | Typische Ursache |
|---|---|---|
| `requests.RequestException` | „HTTP/Ollama-Fehler“ | Ollama läuft nicht oder Modell fehlt (404) |
| `subprocess.CalledProcessError` | „Externer Befehl fehlgeschlagen“ + Befehl | Konvertierung/`ollama create` schlug fehl |
| alle anderen | „Fehler: …“ | z. B. fehlende Datei, falsches Label in der CSV |

Am Ende: `if __name__ == "__main__": main()` – `main()` startet nur, wenn die Datei direkt
ausgeführt wird (nicht beim Import).

---

## 5. Ablauf Schritt für Schritt (aus Nutzersicht)

### Variante „Training“ (Menü 2)

| # | Was passiert | Sichtbar als |
|---|---|---|
| 1 | CSV wird geprüft und geladen | `Trainingsbeispiele: 500` |
| 2 | Tokenizer wird geladen | – |
| 3 | GPU wird erkannt | `GPU wird verwendet: …` |
| 4 | Modell wird geladen und auf die GPU verschoben | `Loading checkpoint shards` |
| 5 | LoRA wird angebracht | `trainable params: …` |
| 6 | Training läuft (375 Schritte) | Loss-Ausgaben alle 5 Schritte |
| 7 | Adapter wird gespeichert | Ordner `adapter/` |
| 8 | Adapter wird ins Modell gemergt | `--- 2/4 LoRA mergen ---`, Ordner `merged_model/` |
| 9 | Protokoll wird geschrieben | `training_info.json` |
| 10 | GPU-Speicher wird freigegeben | – |
| 11 | Konvertierung nach GGUF | `--- 3/4 GGUF-Konvertierung ---`, `trained-model.gguf` |
| 12 | Modelfile + `ollama create` | `--- 4/4 Ollama-Import ---` |
| 13 | Fertig | `FERTIG. … trained-modell` |

### Variante „Abfrage“ (Menü 1, 3, 4)

1. Text eingeben (oder Enter für den Beispieltext).
2. `ask_ollama` sendet System-Prompt + Text an Ollama.
3. Das Modell antwortet mit `relevant` oder `nicht_relevant`.
4. Antwort wird ausgegeben.

---

## 6. Wo werden welche Daten gespeichert?

```
CSV ──► (Speicher) ──► adapter/ (klein)
                   └─► merged_model/ (ca. 6 GB)
                            └─► trained-model.gguf (ca. 6 GB)
                                     └─► Ollama-interne Kopie (trained-modell)
```

Bei jedem Training werden `adapter/`, `merged_model/`, `training_output/` und die GGUF-Datei
**gelöscht und neu erzeugt**. Ältere Ergebnisse gehen dabei verloren.

---

## 7. Häufige Probleme

| Problem | Ursache / Lösung |
|---|---|
| `404 Not Found` bei Menü 1 oder 4 | Das in `OLLAMA_BASE_MODEL` eingetragene Modell fehlt in Ollama → `ollama pull <name>:<tag>` oder `OLLAMA_BASE_MODEL` in `config.py` auf ein vorhandenes Modell setzen. |
| `404 Not Found` bei Menü 3 | Das Training wurde noch nicht erfolgreich bis zum Ollama-Import durchgeführt. |
| Ladebalken bleibt bei `1/2` stehen | Die zweite Datei wird noch von der Platte gelesen (beim ersten Start langsam), oder die GPU ist durch einen **anderen Python-Prozess** belegt. Mit `nvidia-smi` prüfen. |
| CUDA out of memory | GPU-Speicher voll: andere Programme beenden, `MAX_LENGTH` verkleinern. |
| `LoRA-Zielmodule fehlen` | `LORA_TARGET_MODULES` passt nicht zum Modell. |
| `Ungültiges Label in CSV-Zeile …` | Nur `relevant` oder `nicht_relevant` sind erlaubt. |
| `llama.cpp existiert, aber … fehlt` | Ordner `llama.cpp` löschen, App klont ihn neu. |
| `ollama nicht gefunden` | Ollama installieren und Terminal neu starten. |

---

## 8. Zusammenfassung in einem Satz

`app.py` liest beschriftete Beispieltexte, trainiert damit per LoRA ein kleines
Sprachmodell nach, verschmilzt das Ergebnis mit dem Basismodell, wandelt es in das
GGUF-Format um, registriert es in Ollama als `trained-modell` und bietet ein
Menü, um Modelle vorher/nachher zu testen und zu vergleichen.
