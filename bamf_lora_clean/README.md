# BAMF / ASM – LoRA + Ollama Lernprojekt

Kleine Konsolen-App zum Lernen eines vollständigen lokalen Fine-Tuning-Ablaufs.
Die Daten und Labels sind **rein synthetisch** und keine fachliche Entscheidungslogik.

## Was Menüpunkt 2 automatisch macht

`LoRA trainieren -> Adapter speichern -> mergen -> GGUF erzeugen -> Ollama importieren`

Der frühere problematische direkte Safetensors-Import in Ollama wird **nicht** verwendet.
Das `Modelfile` zeigt auf `trained-model.gguf`.

## Einmalige Einrichtung unter Windows

Im Projektordner PowerShell öffnen:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
```

Danach das gewünschte Trainingsmodell herunterladen. Fertige Befehle stehen in:

`huggingface_download_commands.txt`

Für den mitgelieferten Standard TinyLlama:

```powershell
py -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TinyLlama/TinyLlama-1.1B-Chat-v0.6', local_dir='models/tinyllama-1.1b-chat-v0.6')"
ollama pull tinyllama:1.1b-chat-v0.6-fp16
```

## Start

```powershell
py app.py
```

Zuerst Menüpunkt `5 - Systemcheck` ausführen. Danach `2` für den kompletten automatischen Trainings-/Importablauf.

## Wichtige Ordner nach dem Training

- `adapter/` – nur die kleinen LoRA-Parameter
- `merged_model/` – Basismodell + LoRA zusammengeführt
- `trained-model.gguf` – Modellformat für Ollama
- `Modelfile` – wird von der App automatisch erzeugt
- `training_info.json` – kleine Trainingszusammenfassung

## TensorFlow

Dieses Projekt benutzt **PyTorch**, nicht TensorFlow. `app.py` setzt deshalb vor den Transformers-Imports `USE_TF=0` und `USE_FLAX=0`. TensorFlow ist nicht in `requirements.txt` enthalten.

## Modellwechsel

Siehe `huggingface_download_commands.txt`. Danach nur `LOCAL_TRAIN_MODEL` und `OLLAMA_BASE_MODEL` in `config.py` passend setzen.

## Hinweis zu großen Modellen

3B und besonders 7B können auf CPU sehr langsam sein und viel RAM benötigen. Die App verwendet absichtlich einen einfachen, verständlichen Full-Precision-Trainingspfad statt QLoRA/4-Bit-Sonderlogik.
