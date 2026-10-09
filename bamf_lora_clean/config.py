from pathlib import Path

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# MODELL
# -----------------------------------------------------------------------------
# Muss zum lokal heruntergeladenen Transformers/Safetensors-Modell passen.
LOCAL_TRAIN_MODEL = ROOT / "models" / "qwen2.5-3b-instruct"

# Vergleichsmodell, das bereits in Ollama vorhanden sein soll.
OLLAMA_BASE_MODEL = "qwen2.5:3b"

# Name des nach dem LoRA-Training neu erzeugten Ollama-Modells.
OLLAMA_TRAINED_MODEL = "trained-modell"
OLLAMA_URL = "http://localhost:11434/api/chat"

# -----------------------------------------------------------------------------
# DATEIEN / ORDNER
# -----------------------------------------------------------------------------
TRAIN_FILE = ROOT / "data" / "text_samples.csv"
ADAPTER_DIR = ROOT / "adapter"
MERGED_MODEL_DIR = ROOT / "merged_model"
TRAINING_OUTPUT_DIR = ROOT / "training_output"
TRAINING_INFO = ROOT / "training_info.json"

LLAMA_CPP_DIR = ROOT / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
GGUF_FILE = ROOT / "trained-model.gguf"
MODELFILE = ROOT / "Modelfile"
GGUF_OUTTYPE = "f16"

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 192

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# -----------------------------------------------------------------------------
# INFERENZ
# -----------------------------------------------------------------------------
TEMPERATURE = 0
MAX_NEW_TOKENS = 12

SYSTEM_PROMPT = """
Du klassifizierst einen synthetischen ASM-Lerntext in genau eine von zwei Klassen:

relevant
nicht_relevant

Definitionen:

relevant:
Der Text enthält konkrete Hinweise auf eigene aktive Kontakte, Unterstützung,
Zusammenarbeit oder organisatorische/logistische Beteiligung im Zusammenhang
mit einer bewaffneten oder sicherheitsrelevanten Gruppe.

Beispiele:
- regelmäßiger oder direkter Kontakt
- Treffen mit Mitgliedern
- Weitergabe von Nachrichten oder Informationen
- Transport von Personen oder Material
- Geldzahlungen oder finanzielle Unterstützung
- organisatorische oder logistische Hilfe
- konkrete Zusammenarbeit oder Beteiligung

nicht_relevant:
Der Text enthält keine konkrete eigene Beteiligung oder Unterstützung.

Dazu gehören zum Beispiel:
- allgemeine Aussagen über Krieg oder bewaffnete Gruppen
- reine Kenntnis über eine Organisation
- Hörensagen
- Opfer- oder Fluchterfahrungen
- Bedrohung durch eine Gruppe
- ausdrücklich verneinter Kontakt
- rein administrative oder persönliche Sachverhalte

Wichtig:
Bewerte nur den Inhalt des gegebenen Textes.
Erfinde keine zusätzlichen Informationen.
Wenn keine konkrete eigene Beteiligung oder Unterstützung erkennbar ist,
klassifiziere als nicht_relevant.

Antworte ausschließlich mit genau einem dieser beiden Werte:

relevant

oder

nicht_relevant

Keine Erklärung.
Keine zusätzlichen Wörter.
"""
