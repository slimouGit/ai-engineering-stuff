from pathlib import Path

ROOT = Path(__file__).parent

# ------------------------------------------------------------
# MODELL
# ------------------------------------------------------------

# Ollama-Basismodell für die lokale Inferenz.
OLLAMA_BASE_MODEL = "tinyllama:1.1b-chat-v0.6-fp16"

# Name der trainierten Variante in Ollama.
OLLAMA_TRAINED_MODEL = "trained-modell"

# Lokale Transformers-/Safetensors-Version derselben TinyLlama-Basis.
# PEFT trainiert dieses Modell lokal, ohne Download zur Laufzeit.
LOCAL_TRAIN_MODEL = ROOT / "models" / "tinyllama-1.1b-chat-v0.6"

OLLAMA_URL = "http://localhost:11434/api/chat"

# ------------------------------------------------------------
# DATEIEN
# ------------------------------------------------------------

TRAIN_FILE = ROOT / "data" / "text_samples.csv"
ADAPTER_DIR = ROOT / "adapter"
MERGED_MODEL_DIR = ROOT / "merged_model"
GGUF_FILE = ROOT / "trained-model.gguf"
MODELFILE = ROOT / "Modelfile"
TRAINING_INFO = ROOT / "training_info.json"

# llama.cpp wird nur für die automatische GGUF-Konvertierung gebraucht.
# Lege/clone llama.cpp einmalig in diesen Ordner.
LLAMA_CPP_DIR = ROOT / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
GGUF_OUTTYPE = "f16"

# ------------------------------------------------------------
# TRAININGSPARAMETER
# ------------------------------------------------------------

EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 128

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ------------------------------------------------------------
# INFERENZ
# ------------------------------------------------------------

TEMPERATURE = 0
MAX_NEW_TOKENS = 12

SYSTEM_PROMPT = """Du klassifizierst einen synthetischen ASM-Lerntext.
Erlaubte Klassen:
relevant
nicht_relevant
Antworte ausschließlich mit genau einer dieser beiden Klassen."""
