import whisper
import torch
from pathlib import Path

# ===== KONFIG =====
AUDIO_FILE = "voice.mp3"  # Pfad zu deiner Datei
MODEL_SIZE = "medium"  # tiny, base, small, medium, large
LANGUAGE = "de"  # "de" für Deutsch (optional)

# ===== CHECK =====
if not Path(AUDIO_FILE).exists():
    raise FileNotFoundError(f"Datei nicht gefunden: {AUDIO_FILE}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nutze Gerät: {device}")

# ===== MODEL LADEN =====
print("Lade Whisper-Modell...")
model = whisper.load_model(MODEL_SIZE, device=device)

# ===== TRANSKRIPTION =====
print("Starte Transkription...")
result = model.transcribe(
    AUDIO_FILE,
    language=LANGUAGE,   # kann auch weggelassen werden (auto-detect)
    fp16=(device == "cuda")
)

# ===== AUSGABE =====
print("\n===== TRANSKRIPT =====\n")
print(result["text"])

# ===== OPTIONAL: MIT ZEITSTEMPELN =====
print("\n===== SEGMENTE =====\n")
for segment in result["segments"]:
    start = segment["start"]
    end = segment["end"]
    text = segment["text"]
    print(f"[{start:.2f}s - {end:.2f}s] {text}")

# ===== OPTIONAL: IN DATEI SPEICHERN =====
output_file = "transkript.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"\nTranskript gespeichert in: {output_file}")