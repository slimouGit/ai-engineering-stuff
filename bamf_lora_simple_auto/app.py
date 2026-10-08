"""
Kleine Lern-App für:

1. Basismodell über Ollama testen
2. LoRA lokal mit PEFT trainieren
3. Modell automatisch nach GGUF konvertieren
4. GGUF automatisch in Ollama importieren
5. Vorher/Nachher vergleichen

Wichtig:
- Ollama wird für die Inferenz benutzt.
- PEFT trainiert LoRA lokal.
- Das Trainingsmodell liegt lokal als Safetensors-/Transformers-Modell.
- Nach dem Training laufen Merge, GGUF-Konvertierung und Ollama-Import automatisch.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# Wir arbeiten nur mit PyTorch, nicht mit TensorFlow.
os.environ["TRANSFORMERS_NO_TF"] = "1"

import requests
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import config


# ============================================================
# 1. DATEN LADEN
# ============================================================

def load_samples():
    """Liest text,label aus der CSV-Datei."""

    samples = []

    with open(config.TRAIN_FILE, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            label = row["label"].strip().lower()

            if label not in {"relevant", "nicht_relevant"}:
                raise ValueError(f"Ungültiges Label: {label}")

            samples.append({
                "text": row["text"].strip(),
                "label": label,
            })

    return samples


# ============================================================
# 2. OLLAMA AUFRUFEN
# ============================================================

def ask_ollama(model_name, text):
    """Schickt einen Text an ein lokal laufendes Ollama-Modell."""

    response = requests.post(
        config.OLLAMA_URL,
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_NEW_TOKENS,
            },
        },
        timeout=180,
    )

    response.raise_for_status()
    return response.json()["message"]["content"].strip()


# ============================================================
# 3. TRAININGSDATENSATZ
# ============================================================

class TextDataset(Dataset):
    """
    Wandelt Text + richtiges Label in Token-IDs um.

    Beispiel:
        Text: Ich möchte meinen Termin verschieben.
        Label: nicht_relevant
    """

    def __init__(self, samples, tokenizer):
        self.items = []

        for sample in samples:
            # Wir verwenden das Chat-Format des jeweiligen Modells.
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": sample["text"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            answer = sample["label"] + tokenizer.eos_token

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            input_ids = (prompt_ids + answer_ids)[:config.MAX_LENGTH]

            # -100 bedeutet: Prompt beim Loss ignorieren.
            # Gelernt wird nur die gewünschte Antwort / das Label.
            labels = ([-100] * len(prompt_ids) + answer_ids)[:config.MAX_LENGTH]
            attention_mask = [1] * len(input_ids)

            padding = config.MAX_LENGTH - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding
            attention_mask += [0] * padding
            labels += [-100] * padding

            self.items.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# ============================================================
# 4. LORA TRAINIEREN
# ============================================================

def train_lora():
    """
    Ein Menüpunkt erledigt den kompletten Ablauf:

    CSV
    -> LoRA-Training
    -> Adapter speichern
    -> Basismodell + Adapter mergen
    -> GGUF erzeugen
    -> in Ollama importieren
    """

    if not config.LOCAL_TRAIN_MODEL.exists():
        print("\nLokales Trainingsmodell fehlt:")
        print(config.LOCAL_TRAIN_MODEL)
        return

    samples = load_samples()

    print("\n--- LoRA-Training ---")
    print("Trainingsbeispiele:", len(samples))
    print("Epochen:", config.EPOCHS)

    # Alles wird ausschließlich aus dem lokalen Modellordner geladen.
    tokenizer = AutoTokenizer.from_pretrained(
        config.LOCAL_TRAIN_MODEL,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_TRAIN_MODEL,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # LoRA: Nur wenige Zusatzparameter werden trainiert.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainierbare Parameter:")
    model.print_trainable_parameters()

    dataset = TextDataset(samples, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(config.ROOT / "training_output"),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    start = time.time()
    result = trainer.train()
    runtime = time.time() - start

    # Alten Adapter ersetzen.
    if config.ADAPTER_DIR.exists():
        shutil.rmtree(config.ADAPTER_DIR)

    model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)

    print("\nLoRA-Adapter gespeichert:", config.ADAPTER_DIR)

    # LoRA-Adapter mit dem Basismodell verschmelzen.
    merged_model = model.merge_and_unload()

    if config.MERGED_MODEL_DIR.exists():
        shutil.rmtree(config.MERGED_MODEL_DIR)

    merged_model.save_pretrained(
        config.MERGED_MODEL_DIR,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(config.MERGED_MODEL_DIR)

    print("Merged Model gespeichert:", config.MERGED_MODEL_DIR)

    info = {
        "trained": True,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "training_examples": len(samples),
        "epochs": config.EPOCHS,
        "train_loss": float(result.training_loss),
        "runtime_seconds": round(runtime, 2),
    }

    config.TRAINING_INFO.write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Train Loss:", round(result.training_loss, 4))
    print("Dauer:", round(runtime, 1), "Sekunden")

    # Ab hier läuft alles automatisch weiter.
    convert_to_gguf()
    import_into_ollama()

    print("\nFertig. Das trainierte Modell kann jetzt direkt getestet werden.")


# ============================================================
# 5. MERGED MODEL -> GGUF
# ============================================================

def convert_to_gguf():
    """Konvertiert merged_model/ automatisch nach trained-model.gguf."""

    if not config.CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            "llama.cpp Konverter nicht gefunden:\n"
            f"{config.CONVERT_SCRIPT}\n\n"
            "Lege llama.cpp einmalig im Projektordner unter 'llama.cpp' ab."
        )

    if not config.MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Merged Model fehlt: {config.MERGED_MODEL_DIR}"
        )

    if config.GGUF_FILE.exists():
        config.GGUF_FILE.unlink()

    print("\n--- GGUF-Konvertierung ---")

    subprocess.run(
        [
            sys.executable,
            str(config.CONVERT_SCRIPT),
            str(config.MERGED_MODEL_DIR),
            "--outfile",
            str(config.GGUF_FILE),
            "--outtype",
            config.GGUF_OUTTYPE,
        ],
        cwd=config.ROOT,
        check=True,
    )

    if not config.GGUF_FILE.exists():
        raise RuntimeError("GGUF-Datei wurde nicht erzeugt.")

    print("GGUF erstellt:", config.GGUF_FILE)


# ============================================================
# 6. GGUF -> OLLAMA
# ============================================================

def import_into_ollama():
    """Importiert die fertige GGUF-Datei automatisch in Ollama."""

    if not config.GGUF_FILE.exists():
        raise FileNotFoundError(
            f"GGUF-Datei fehlt: {config.GGUF_FILE}"
        )

    # Wichtig: FROM zeigt jetzt auf GGUF, nicht mehr auf merged_model/.
    config.MODELFILE.write_text(
        f'''FROM ./{config.GGUF_FILE.name}
PARAMETER temperature {config.TEMPERATURE}
SYSTEM """{config.SYSTEM_PROMPT}"""
''',
        encoding="utf-8",
    )

    print("\n--- Ollama Import ---")
    print("Modell:", config.OLLAMA_TRAINED_MODEL)

    subprocess.run(
        [
            "ollama",
            "create",
            config.OLLAMA_TRAINED_MODEL,
            "-f",
            str(config.MODELFILE),
        ],
        cwd=config.ROOT,
        check=True,
    )

    print("Ollama-Modell erstellt:", config.OLLAMA_TRAINED_MODEL)


# ============================================================
# 7. MODELLE TESTEN
# ============================================================

def get_test_text():
    text = input("\nText [Enter = Beispiel]: ").strip()

    if not text:
        text = "Ich hatte regelmäßig Kontakt zu einer bewaffneten Gruppe."

    return text


def test_model(model_name):
    text = get_test_text()

    print("\nModell:", model_name)
    print("Text:", text)
    print("Antwort:", ask_ollama(model_name, text))


def compare_models():
    text = get_test_text()

    print("\n--- VOR TRAINING ---")
    print(ask_ollama(config.OLLAMA_BASE_MODEL, text))

    print("\n--- NACH LORA-TRAINING ---")
    print(ask_ollama(config.OLLAMA_TRAINED_MODEL, text))


# ============================================================
# 8. HAUPTMENÜ
# ============================================================

def main():
    while True:
        print("""
========================================
BAMF / ASM - OLLAMA + LORA
========================================
1 - Basismodell testen
2 - LoRA trainieren
3 - Trainiertes Modell testen
4 - Vorher / Nachher vergleichen
0 - Ende
""")

        choice = input("Auswahl: ").strip()

        try:
            if choice == "1":
                test_model(config.OLLAMA_BASE_MODEL)

            elif choice == "2":
                train_lora()

            elif choice == "3":
                test_model(config.OLLAMA_TRAINED_MODEL)

            elif choice == "4":
                compare_models()

            elif choice == "0":
                break

            else:
                print("Ungültige Auswahl.")

        except requests.RequestException as exc:
            print("\nOllama-Aufruf fehlgeschlagen:", exc)

        except subprocess.CalledProcessError as exc:
            print("\nExterner Befehl fehlgeschlagen:", exc)

        except Exception as exc:
            print("\nFehler:", exc)


if __name__ == "__main__":
    main()
