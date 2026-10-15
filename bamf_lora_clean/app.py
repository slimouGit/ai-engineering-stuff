"""
BAMF/ASM Lernprojekt: Ollama + LoRA + automatischer GGUF-Import.

Menüpunkt 2 führt automatisch aus:
1. LoRA-Training
2. Adapter speichern
3. Basismodell + LoRA mergen
4. merged_model nach GGUF konvertieren
5. Modelfile erzeugen
6. GGUF als Ollama-Modell registrieren

Hinweis: Die Daten sind rein synthetisch und dienen nur dem Lernen.
"""

# Wichtig: Transformers soll in diesem Projekt NICHT TensorFlow laden.
# Dadurch vermeiden wir Konflikte mit einer eventuell global installierten
# TensorFlow/protobuf-Kombination. Das Projekt nutzt PyTorch.
import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

import config


VALID_LABELS = {"relevant", "nicht_relevant"}


def load_samples():
    """Liest text,label aus der CSV-Datei und prüft die Grundstruktur."""
    if not config.TRAIN_FILE.exists():
        raise FileNotFoundError(f"Trainingsdatei fehlt: {config.TRAIN_FILE}")

    samples = []
    with open(config.TRAIN_FILE, encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV muss die Spalten 'text' und 'label' enthalten.")

        for line_no, row in enumerate(reader, start=2):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()

            if not text:
                raise ValueError(f"Leerer Text in CSV-Zeile {line_no}")
            if label not in VALID_LABELS:
                raise ValueError(f"Ungültiges Label in CSV-Zeile {line_no}: {label}")

            samples.append({"text": text, "label": label})

    if not samples:
        raise ValueError("Die Trainingsdatei enthält keine Beispiele.")

    return samples


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


def build_prompt(tokenizer, text):
    """Verwendet das Chat-Template des Modells; hat aber einen Fallback."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback für Tokenizer ohne Chat-Template.
    return (
        f"System: {config.SYSTEM_PROMPT}\n"
        f"User: {text}\n"
        "Assistant:"
    )


class TextDataset(Dataset):
    """Bereitet Prompt + korrektes Label für Causal-LM-Training vor."""

    def __init__(self, samples, tokenizer):
        self.items = []

        eos = tokenizer.eos_token or ""
        for sample in samples:
            prompt = build_prompt(tokenizer, sample["text"])
            answer = sample["label"] + eos

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

            # Die Antwort soll niemals komplett durch MAX_LENGTH abgeschnitten werden.
            max_prompt_len = max(1, config.MAX_LENGTH - len(answer_ids))
            prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = (prompt_ids + answer_ids)[: config.MAX_LENGTH]
            labels = ([-100] * len(prompt_ids) + answer_ids)[: config.MAX_LENGTH]
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


def ensure_model_compatible(model):
    """Prüft, ob die in config eingestellten LoRA-Zielmodule existieren."""
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = [m for m in config.LORA_TARGET_MODULES if m not in module_names]
    if missing:
        raise RuntimeError(
            "LoRA-Zielmodule fehlen im Modell: " + ", ".join(missing) + "\n"
            "Passe LORA_TARGET_MODULES in config.py an."
        )


def train_lora():
    """Trainiert LoRA und stößt danach automatisch GGUF + Ollama-Import an."""
    if not config.LOCAL_TRAIN_MODEL.exists():
        raise FileNotFoundError(
            f"Lokales Trainingsmodell fehlt: {config.LOCAL_TRAIN_MODEL}\n"
            "Siehe huggingface_download_commands.txt."
        )

    samples = load_samples()

    print("\n--- 1/4 LoRA-Training ---")
    print("Modell:", config.LOCAL_TRAIN_MODEL)
    print("Trainingsbeispiele:", len(samples))
    print("Epochen:", config.EPOCHS)

    tokenizer = AutoTokenizer.from_pretrained(
        config.LOCAL_TRAIN_MODEL,
        local_files_only=True,
        trust_remote_code=False,
    )

    if tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer hat kein EOS-Token. Für dieses Lernprojekt bitte anderes Modell verwenden.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_gpu = torch.cuda.is_available()

    print("\n--- Hardware ---")
    if use_gpu:
        print("GPU wird verwendet:", torch.cuda.get_device_name(0))
    else:
        print("Keine CUDA-GPU gefunden -> CPU wird verwendet")

    model = AutoModelForCausalLM.from_pretrained(
        config.LOCAL_TRAIN_MODEL,
        local_files_only=True,
        dtype=torch.float16 if use_gpu else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    # Modell explizit auf die GPU legen.
    if use_gpu:
        model = model.to("cuda")

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    ensure_model_compatible(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = TextDataset(samples, tokenizer)

    if config.TRAINING_OUTPUT_DIR.exists():
        shutil.rmtree(config.TRAINING_OUTPUT_DIR)

    training_args = TrainingArguments(
        output_dir=str(config.TRAINING_OUTPUT_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=not use_gpu,
        dataloader_pin_memory=use_gpu,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    start = time.time()
    result = trainer.train()
    runtime = time.time() - start

    if config.ADAPTER_DIR.exists():
        shutil.rmtree(config.ADAPTER_DIR)
    model.save_pretrained(config.ADAPTER_DIR)
    tokenizer.save_pretrained(config.ADAPTER_DIR)

    print("\n--- 2/4 LoRA mergen ---")
    merged_model = model.merge_and_unload()

    if config.MERGED_MODEL_DIR.exists():
        shutil.rmtree(config.MERGED_MODEL_DIR)

    merged_model.save_pretrained(
        config.MERGED_MODEL_DIR,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(config.MERGED_MODEL_DIR)

    config.TRAINING_INFO.write_text(
        json.dumps(
            {
                "trained": True,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "training_examples": len(samples),
                "epochs": config.EPOCHS,
                "train_loss": float(result.training_loss),
                "runtime_seconds": round(runtime, 2),
                "base_model_path": str(config.LOCAL_TRAIN_MODEL),
                "lora_r": config.LORA_R,
                "lora_alpha": config.LORA_ALPHA,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Speicher freigeben, bevor der Konverter startet.
    del trainer
    del model
    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    convert_to_gguf()
    import_into_ollama()

    print("\nFERTIG.")
    print("Das trainierte Modell ist jetzt in Ollama verfügbar als:")
    print(config.OLLAMA_TRAINED_MODEL)


def ensure_llama_cpp():
    """Prüft llama.cpp. Falls es fehlt, wird es automatisch per git geklont."""
    if config.CONVERT_SCRIPT.exists():
        return

    if config.LLAMA_CPP_DIR.exists() and not config.CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"{config.LLAMA_CPP_DIR} existiert, aber convert_hf_to_gguf.py fehlt.\n"
            "Lösche den unvollständigen llama.cpp-Ordner und starte erneut."
        )

    if shutil.which("git") is None:
        raise RuntimeError(
            "llama.cpp fehlt und Git wurde nicht gefunden.\n"
            "Installiere Git oder klone llama.cpp manuell."
        )

    print("\nllama.cpp fehlt -> wird automatisch geklont ...")
    subprocess.run(
        ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(config.LLAMA_CPP_DIR)],
        cwd=config.ROOT,
        check=True,
    )

    if not config.CONVERT_SCRIPT.exists():
        raise FileNotFoundError("llama.cpp wurde geklont, aber Konvertierungsskript fehlt.")


def convert_to_gguf():
    """Konvertiert das gemergte Transformers-Modell automatisch nach GGUF."""
    print("\n--- 3/4 GGUF-Konvertierung ---")

    if not config.MERGED_MODEL_DIR.exists():
        raise FileNotFoundError(f"Merged Model fehlt: {config.MERGED_MODEL_DIR}")

    ensure_llama_cpp()

    if config.GGUF_FILE.exists():
        config.GGUF_FILE.unlink()

    command = [
        sys.executable,
        str(config.CONVERT_SCRIPT),
        str(config.MERGED_MODEL_DIR),
        "--outfile",
        str(config.GGUF_FILE),
        "--outtype",
        config.GGUF_OUTTYPE,
    ]

    print("Konvertiere nach:", config.GGUF_FILE)
    subprocess.run(command, cwd=config.LLAMA_CPP_DIR, check=True)

    if not config.GGUF_FILE.exists() or config.GGUF_FILE.stat().st_size == 0:
        raise RuntimeError("GGUF-Konvertierung lief ohne sichtbaren Fehler, aber GGUF-Datei fehlt.")

    print("GGUF erstellt:", config.GGUF_FILE)


def import_into_ollama():
    """Erzeugt ein Modelfile für GGUF und registriert es automatisch in Ollama."""
    print("\n--- 4/4 Ollama-Import ---")

    if not config.GGUF_FILE.exists():
        raise FileNotFoundError(f"GGUF-Datei fehlt: {config.GGUF_FILE}")
    if shutil.which("ollama") is None:
        raise RuntimeError("Der Befehl 'ollama' wurde nicht gefunden.")

    # Wichtig: FROM zeigt bewusst auf GGUF und NICHT auf merged_model/Safetensors.
    config.MODELFILE.write_text(
        f'''FROM ./{config.GGUF_FILE.name}\n\n'''
        f'''PARAMETER temperature {config.TEMPERATURE}\n\n'''
        f'''SYSTEM """{config.SYSTEM_PROMPT}"""\n''',
        encoding="utf-8",
    )

    subprocess.run(
        ["ollama", "create", config.OLLAMA_TRAINED_MODEL, "-f", str(config.MODELFILE)],
        cwd=config.ROOT,
        check=True,
    )

    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    if config.OLLAMA_TRAINED_MODEL.lower() not in result.stdout.lower():
        raise RuntimeError("Ollama create lief durch, aber das Modell erscheint nicht in 'ollama list'.")

    print("Ollama-Modell erfolgreich erstellt:", config.OLLAMA_TRAINED_MODEL)


def environment_check():
    """Kurzer Vorabcheck, ohne ein Training zu starten."""
    print("\n--- Systemcheck ---")

    checks = [
        ("Python", sys.version.split()[0]),
        ("Trainings-CSV", "OK" if config.TRAIN_FILE.exists() else "FEHLT"),
        ("Lokales HF-Modell", "OK" if config.LOCAL_TRAIN_MODEL.exists() else "FEHLT"),
        ("Git", shutil.which("git") or "FEHLT"),
        ("Ollama CLI", shutil.which("ollama") or "FEHLT"),
        ("llama.cpp Converter", "OK" if config.CONVERT_SCRIPT.exists() else "wird bei Bedarf geklont"),
    ]

    for name, value in checks:
        print(f"{name:22}: {value}")

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        print(f"{'Ollama Server':22}: OK")
    except Exception as exc:
        print(f"{'Ollama Server':22}: NICHT ERREICHBAR ({exc})")


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


def main():
    while True:
        print("""
========================================
BAMF / ASM - OLLAMA + LORA
========================================
1 - Basismodell testen
2 - LoRA trainieren + GGUF + Ollama (automatisch)
3 - Trainiertes Modell testen
4 - Vorher / Nachher vergleichen
5 - Systemcheck
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
            elif choice == "5":
                environment_check()
            elif choice == "0":
                break
            else:
                print("Ungültige Auswahl.")

        except requests.RequestException as exc:
            print("\nHTTP/Ollama-Fehler:", exc)
        except subprocess.CalledProcessError as exc:
            print("\nExterner Befehl fehlgeschlagen:", exc)
            print("Befehl:", " ".join(map(str, exc.cmd)) if exc.cmd else "unbekannt")
        except Exception as exc:
            print("\nFehler:", exc)


if __name__ == "__main__":
    main()
