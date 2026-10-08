"""
Kleine Lern-App für:

1. Basismodell über Ollama testen
2. LoRA lokal mit PEFT trainieren
3. trainierte Variante automatisch in Ollama importieren
4. Vorher/Nachher vergleichen

Wichtig:
- Ollama wird für die Inferenz benutzt.
- PEFT trainiert LoRA lokal.
- Es gibt keine Hugging-Face-API-Aufrufe.
- Das Trainingsmodell muss bereits lokal als Safetensors-/Transformers-Modell liegen.
"""

import os

os.environ["TRANSFORMERS_NO_TF"] = "1"

import csv
import json
import shutil
import subprocess
import time
from datetime import datetime

import requests
import torch

from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

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

    Das Modell sieht z. B.:
        Text: Ich möchte meinen Termin verschieben.
        richtige Antwort: nicht_relevant
    """

    def __init__(self, samples, tokenizer):
        self.items = []

        for sample in samples:
            # Qwen-eigenes Chat-Format verwenden.
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

            # -100 bedeutet: Für den Prompt keinen Trainingsfehler berechnen.
            # Gelernt werden soll nur die gewünschte Antwort.
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
    Lädt das bereits lokal vorhandene Trainingsmodell,
    trainiert nur die kleinen LoRA-Parameter und importiert
    das Ergebnis anschließend automatisch in Ollama.
    """

    if not config.LOCAL_TRAIN_MODEL.exists():
        print("\nLokales Trainingsmodell fehlt:")
        print(config.LOCAL_TRAIN_MODEL)
        print("Siehe README: Das Modell muss dort lokal als Safetensors-Modell liegen.")
        return

    samples = load_samples()

    print("\n--- LoRA-Training ---")
    print("Trainingsbeispiele:", len(samples))
    print("Epochen:", config.EPOCHS)

    # local_files_only=True verhindert einen Download aus dem Internet.
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

    # Für einen einfachen Ollama-Import verschmelzen wir
    # Basismodell + LoRA zu einem vollständigen Modell.
    merged_model = model.merge_and_unload()

    if config.MERGED_MODEL_DIR.exists():
        shutil.rmtree(config.MERGED_MODEL_DIR)

    merged_model.save_pretrained(
        config.MERGED_MODEL_DIR,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(config.MERGED_MODEL_DIR)

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

    # Nach dem Training direkt die neue Variante in Ollama erstellen.
    import_into_ollama()


# ============================================================
# 5. TRAINIERTE VARIANTE IN OLLAMA IMPORTIEREN
# ============================================================

def import_into_ollama():
    """
    Importiert das fertig zusammengeführte Modell automatisch in Ollama.

    Nach erfolgreichem Training soll der Benutzer NICHT mehr
    manuell 'ollama create ...' ausführen müssen.
    """

    if not config.MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Kein zusammengeführtes Modell gefunden: "
            f"{config.MERGED_MODEL_DIR}"
        )

    # --------------------------------------------------------
    # Modelfile automatisch erzeugen
    # --------------------------------------------------------

    config.MODELFILE.write_text(
        f'''FROM ./merged_model

PARAMETER temperature {config.TEMPERATURE}

SYSTEM """{config.SYSTEM_PROMPT}"""
''',
        encoding="utf-8",
    )

    print("\n--- Ollama Import ---")
    print("Modell:", config.OLLAMA_TRAINED_MODEL)

    # --------------------------------------------------------
    # Modell automatisch in Ollama registrieren
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Kontrollieren, ob Ollama das Modell jetzt kennt
    # --------------------------------------------------------

    result = subprocess.run(
        [
            "ollama",
            "list",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    if config.OLLAMA_TRAINED_MODEL not in result.stdout:
        raise RuntimeError(
            "Training war erfolgreich, aber das Modell "
            "wurde nicht korrekt in Ollama registriert."
        )

    print("\nOllama-Modell erfolgreich erstellt:")
    print(config.OLLAMA_TRAINED_MODEL)


# ============================================================
# 6. MODELLE TESTEN
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
# 7. HAUPTMENÜ
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
            print("\nOllama nicht erreichbar:", exc)

        except subprocess.CalledProcessError as exc:
            print("\nOllama-Befehl fehlgeschlagen:", exc)

        except Exception as exc:
            print("\nFehler:", exc)


if __name__ == "__main__":
    main()
