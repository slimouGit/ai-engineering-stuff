# BAMF / ASM – Ollama + LoRA Lernprojekt

Bewusst kleine Konsolen-App ohne Frontend und ohne zusätzliche Schichten.

## Was die App zeigt

```text
Ollama-Basismodell
        ↓
      testen

CSV-Trainingsdaten
        ↓
lokales Trainingsmodell
        ↓
PEFT + LoRA
        ↓
Adapter
        ↓
Basismodell + Adapter zusammenführen
        ↓
trainierte Variante in Ollama importieren
        ↓
Vorher / Nachher vergleichen
```

## Wichtig: Ollama und Training

Ollama wird hier für die **lokale Inferenz** benutzt.

PEFT kann das interne Ollama-Modell (GGUF) nicht direkt trainieren. Für das LoRA-Training wird deshalb dieselbe Modellarchitektur zusätzlich als **lokaler Transformers-/Safetensors-Modellordner** benötigt.

Die App lädt während des Trainings **nichts von Hugging Face oder einer anderen API**. Im Code steht ausdrücklich:

```python
local_files_only=True
```

Damit bleibt der Trainingslauf lokal.

## Projektstruktur

```text
bamf_lora_simple/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   └── text_samples.csv
└── models/
    └── qwen2.5-0.5b-instruct/   <- lokal bereitstellen
```

Nach dem Training entstehen zusätzlich:

```text
adapter/
merged_model/
Modelfile
training_info.json
```

## 1. PyCharm

Projektordner in PyCharm öffnen und einen Virtualenv-Interpreter anlegen.

Dann im PyCharm-Terminal:

```bash
pip install -r requirements.txt
```

## 2. Ollama vorbereiten

Ollama installieren und prüfen:

```bash
ollama --version
```

Basismodell laden:

```bash
ollama pull qwen2.5:0.5b
```

Prüfen:

```bash
ollama list
```

Falls Ollama nicht läuft:

```bash
ollama serve
```

## 3. Lokales Trainingsmodell

Für PEFT muss dieselbe Modellarchitektur zusätzlich lokal in diesem Ordner liegen:

```text
models/qwen2.5-0.5b-instruct/
```

Darin müssen die normalen Transformers-/Safetensors-Dateien liegen, z. B.:

```text
config.json
model.safetensors
tokenizer.json
tokenizer_config.json
...
```

Wichtig: Die App lädt diese Dateien **nur lokal**.

## 4. Parameter ändern

Alle wichtigen Einstellungen stehen getrennt in:

```text
config.py
```

Dort kannst du z. B. ändern:

```python
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 128
LORA_R = 8
LORA_ALPHA = 16
OLLAMA_BASE_MODEL = "qwen2.5:0.5b"
```

`app.py` muss dafür nicht geändert werden.

## 5. App starten

```bash
python app.py
```

Menü:

```text
1 - Basismodell testen
2 - LoRA trainieren
3 - Trainiertes Modell testen
4 - Vorher / Nachher vergleichen
0 - Ende
```

## 6. Was passiert bei Menüpunkt 2?

```text
text_samples.csv
      ↓
Text + richtiges Label
      ↓
lokales Qwen-Modell
      ↓
LoRA-Parameter hinzufügen
      ↓
nur diese kleinen Parameter trainieren
      ↓
adapter/
      ↓
Adapter mit Basismodell verschmelzen
      ↓
merged_model/
      ↓
ollama create
      ↓
bamf-qwen-lora
```

Danach kannst du das neue Modell auch direkt prüfen:

```bash
ollama list
```

oder:

```bash
ollama run bamf-qwen-lora
```

## 7. Was ist wo gespeichert?

### Ollama-Basismodell

Von Ollama selbst verwaltet:

```text
qwen2.5:0.5b
```

### Trainierter LoRA-Teil

```text
adapter/
```

### Vollständige zusammengeführte Variante

```text
merged_model/
```

### Trainierte Ollama-Variante

Von Ollama verwaltet unter dem Namen:

```text
bamf-qwen-lora
```

## 8. Was soll man aus dem Projekt lernen?

Nur diese Kernidee:

```text
Basismodell
+ gelabelte Trainingsdaten
+ LoRA
= angepasster Adapter

Basismodell
+ Adapter
= trainierte Variante
```

Ollama übernimmt danach die lokale Nutzung des Modells.
