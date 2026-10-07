"""
Lokale Textklassifikation mit TinyLlama + LoRA
==============================================

Die App macht nur das Nötigste:
1. Basismodell testen
2. LoRA auf `data/text_samples.csv` trainieren
3. Trainiertes Modell testen
4. Trainingsstatus anzeigen
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


ROOT = Path(__file__).parent
DATA_FILE = ROOT / "data" / "text_samples.csv"
ADAPTER_DIR = ROOT / "adapter"
TRAIN_INFO_FILE = ROOT / "training_info.json"

# HF_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
HF_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 160
TRAIN_EPOCHS = 3
DEFAULT_TEXT = "Ich habe während des Krieges viele bewaffnete Männer gesehen."
SYSTEM_PROMPT = "Antworte nur mit relevant oder nicht_relevant."

_BASE_CACHE = {"model": None, "tokenizer": None}
_TRAINED_CACHE = {"model": None, "tokenizer": None}


def load_samples():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {DATA_FILE}")

    rows = []
    with DATA_FILE.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if text and label in {"relevant", "nicht_relevant"}:
                rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError("Keine gueltigen Trainingsdaten in text_samples.csv gefunden.")

    return rows


def make_prompt(text):
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{text}</s>\n"
        f"<|assistant|>\n"
    )


def normalize_label(text):
    clean = " ".join(text.lower().split())
    if "nicht_relevant" in clean or "nicht relevant" in clean:
        return "nicht_relevant"
    if "relevant" in clean:
        return "relevant"
    return "unsicher"


def load_base_model():
    if _BASE_CACHE["model"] is not None:
        return _BASE_CACHE["model"], _BASE_CACHE["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_BASE_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.generation_config.max_length = None
    model.eval()

    _BASE_CACHE["model"] = model
    _BASE_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def load_trained_model():
    if _TRAINED_CACHE["model"] is not None:
        return _TRAINED_CACHE["model"], _TRAINED_CACHE["tokenizer"]

    adapter_file = ADAPTER_DIR / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError("Kein trainiertes Modell gefunden. Bitte zuerst LoRA trainieren.")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model, _ = load_base_model()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    model.generation_config.max_length = None

    _TRAINED_CACHE["model"] = model
    _TRAINED_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def classify_text(model, tokenizer, text):
    inputs = tokenizer(
        make_prompt(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return normalize_label(decoded)


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.items = []
        for row in rows:
            prompt_ids = tokenizer(make_prompt(row["text"]), add_special_tokens=False)["input_ids"][:MAX_LENGTH]
            answer_ids = tokenizer(row["label"] + "</s>", add_special_tokens=False)["input_ids"][:MAX_LENGTH]

            input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
            attention_mask = [1] * len(input_ids)

            padding = MAX_LENGTH - len(input_ids)
            if padding > 0:
                input_ids += [tokenizer.pad_token_id] * padding
                attention_mask += [0] * padding
                labels += [-100] * padding

            self.items.append(
                {
                    "input_ids": torch.tensor(input_ids),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(labels),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def train_lora():
    rows = load_samples()
    print(f"\nTrainingsbeispiele: {len(rows)}")
    print("\nEpochen: ", TRAIN_EPOCHS)

    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_BASE_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )

    dataset = TextDataset(rows, tokenizer)
    training_args = TrainingArguments(
        output_dir=str(ROOT / "training_output"),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        dataloader_pin_memory=False,
    )

    print("\nTraining startet ...")
    start = time.time()
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    result = trainer.train()
    runtime = time.time() - start

    ADAPTER_DIR.mkdir(exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    info = {
        "trained": True,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": HF_BASE_MODEL,
        "training_examples": len(dataset),
        "epochs": TRAIN_EPOCHS,
        "train_loss": float(result.training_loss),
        "runtime_seconds": round(runtime, 2),
    }
    TRAIN_INFO_FILE.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFertig.")
    print("Train Loss:", round(result.training_loss, 4))
    print("Dauer:", round(runtime, 1), "Sekunden")


def input_text():
    text = input("\nText eingeben [Enter = Beispiel]: ").strip()
    return text or DEFAULT_TEXT


def test_base_model():
    text = input_text()
    model, tokenizer = load_base_model()
    print("\nKlasse:", classify_text(model, tokenizer, text))


def test_trained_model():
    try:
        model, tokenizer = load_trained_model()
    except FileNotFoundError:
        print("\nBitte zuerst trainieren.")
        return

    text = input_text()
    print("\nKlasse:", classify_text(model, tokenizer, text))


def show_training_status():
    if not TRAIN_INFO_FILE.exists():
        print("\nNoch kein Training durchgeführt.")
        return

    info = json.loads(TRAIN_INFO_FILE.read_text(encoding="utf-8"))
    print("\n--- Trainingsstatus ---")
    print("Trainiert:", info["trained"])
    print("Zeitpunkt:", info["trained_at"])
    print("Trainingsbeispiele:", info["training_examples"])
    print("Epochen:", info["epochs"])
    print("Train Loss:", info["train_loss"])
    print("Dauer:", info["runtime_seconds"], "Sekunden")


def main():
    while True:
        print(
            """
====================================
Lokale Textklassifikation
====================================
1 - Basismodell testen
2 - LoRA trainieren
3 - Trainiertes Modell testen
4 - Trainingsstatus anzeigen
0 - Ende
"""
        )

        choice = input("Auswahl: ").strip()

        try:
            if choice == "1":
                test_base_model()
            elif choice == "2":
                train_lora()
            elif choice == "3":
                test_trained_model()
            elif choice == "4":
                show_training_status()
            elif choice == "0":
                break
            else:
                print("Ungültige Auswahl.")
        except Exception as exc:
            print("\nFehler:")
            print(exc)


if __name__ == "__main__":
    main()
