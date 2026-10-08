# BAMF / ASM – einfache Ollama + LoRA Lern-App

Die App bleibt bewusst klein. **Menüpunkt 2 erledigt jetzt den kompletten Trainingsweg automatisch.**

```text
CSV
→ LoRA trainieren
→ Adapter speichern
→ mit Basismodell mergen
→ GGUF erzeugen
→ in Ollama importieren
→ fertig
```

Danach kann direkt Menüpunkt `3 - Trainiertes Modell testen` verwendet werden.

## Projektstruktur

```text
bamf_lora_simple_auto/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   └── text_samples.csv
├── models/
│   └── tinyllama-1.1b-chat-v0.6/
└── llama.cpp/
    └── convert_hf_to_gguf.py
```

Nach dem Training entstehen automatisch:

```text
adapter/
merged_model/
trained-model.gguf
Modelfile
training_info.json
```

## Einmalige Vorbereitung

### 1. Python-Pakete

```powershell
py -m pip install -r requirements.txt
```

### 2. Ollama-Basismodell

```powershell
ollama pull tinyllama:1.1b-chat-v0.6-fp16
```

### 3. Lokales TinyLlama-Trainingsmodell

Der Ordner muss existieren:

```text
models/tinyllama-1.1b-chat-v0.6/
```

Einmaliger Download:

```powershell
py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v0.6', local_dir='models/tinyllama-1.1b-chat-v0.6')"
```

### 4. llama.cpp einmalig bereitstellen

Im Projektordner:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
```

Dann die für den Konverter benötigten Pakete installieren:

```powershell
py -m pip install -r llama.cpp/requirements.txt
```

Danach muss für Training/Import nichts mehr manuell ausgeführt werden.

## Start

```powershell
py app.py
```

Menü:

```text
1 - Basismodell testen
2 - LoRA trainieren
3 - Trainiertes Modell testen
4 - Vorher / Nachher vergleichen
0 - Ende
```

## Was Menüpunkt 2 automatisch macht

```text
text_samples.csv
→ TinyLlama lokal laden
→ LoRA trainieren
→ adapter/
→ merge_and_unload()
→ merged_model/
→ convert_hf_to_gguf.py
→ trained-model.gguf
→ Modelfile mit FROM ./trained-model.gguf
→ ollama create trained-modell
```

Am Ende sollte `ollama list` automatisch ein Modell namens `trained-modell:latest` enthalten.

## Parameter

Alle Einstellungen bleiben getrennt in `config.py`, z. B.:

```python
EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_R = 8
LORA_ALPHA = 16
GGUF_OUTTYPE = "f16"
```
