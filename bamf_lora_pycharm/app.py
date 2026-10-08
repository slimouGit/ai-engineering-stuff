"""
Lokale Textklassifikation mit TinyLlama + LoRA
==============================================

Diese Datei ist eine kleine Lern-App fuer folgendes Ziel:

1. Ein lokales Basismodell laden und testen
2. Einen LoRA-Adapter auf einem CSV-Datensatz trainieren
3. Das trainierte Modell spaeter wieder laden und testen
4. Den Trainingsstatus in einer JSON-Datei anzeigen

Die App ist bewusst klein gehalten, damit man den kompletten Ablauf
verstehen kann:

- Daten laden
- Prompt bauen
- Modell laden
- Vorhersage erzeugen
- Adapter speichern
- trainiertes Modell wiederverwenden
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

# Basismodell, das lokal geladen und mit LoRA angepasst wird.
# Wenn du spaeter ein anderes kleines Chat-Modell testen willst,
# musst du nur diesen Namen austauschen.
HF_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 160
TRAIN_EPOCHS = 3
DEFAULT_TEXT = "Ich habe während des Krieges viele bewaffnete Männer gesehen."
SYSTEM_PROMPT = "Antworte nur mit relevant oder nicht_relevant."

# Kleine In-Memory-Caches:
# - _BASE_CACHE speichert das Basismodell und den Tokenizer
# - _TRAINED_CACHE speichert das geladene Modell mit LoRA-Adapter
# Dadurch muessen grosse Modelle nicht bei jeder Auswahl neu geladen werden.
_BASE_CACHE = {"model": None, "tokenizer": None}
_TRAINED_CACHE = {"model": None, "tokenizer": None}


def load_samples():
    """Laedt `data/text_samples.csv` und filtert nur gueltige Trainingszeilen.

    Erwartet wird ein CSV-Format mit den Spalten:

    - `text`: der eigentliche Beispielsatz
    - `label`: `relevant` oder `nicht_relevant`

    Jede Zeile wird in ein Dictionary umgewandelt, damit sie spaeter direkt
    fuer das Training verarbeitet werden kann.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Trainingsdatei nicht gefunden: {DATA_FILE}")

    rows = []
    with DATA_FILE.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip().lower()

            # Nur Zeilen verwenden, die wirklich Text enthalten und ein
            # erwartetes Label besitzen. Alles andere wird ignoriert.
            if text and label in {"relevant", "nicht_relevant"}:
                rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError("Keine gueltigen Trainingsdaten in text_samples.csv gefunden.")

    return rows


def make_prompt(text):
    """Baut das Chat-Prompt im einfachen System/User/Assistant-Format.

    Das Basismodell bekommt damit eine klare Aufgabe:
    Es soll den Eingabetext als `relevant` oder `nicht_relevant` einordnen.
    """
    return (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{text}</s>\n"
        f"<|assistant|>\n"
    )


def normalize_label(text):
    """Reduziert freien Modelltext auf ein sauberes Klassenlabel.

    Das Modell antwortet nicht immer exakt nur mit einem Wort. Deshalb wird
    die Ausgabe nach `relevant` oder `nicht_relevant` durchsucht und auf
    eines dieser Labels normalisiert.
    """
    clean = " ".join(text.lower().split())
    if "nicht_relevant" in clean or "nicht relevant" in clean:
        return "nicht_relevant"
    if "relevant" in clean:
        return "relevant"
    return "unsicher"


def load_base_model():
    """Laedt das Basismodell und den dazugehoerigen Tokenizer.

    Der erste Aufruf kann laenger dauern, weil das Modell von Hugging Face
    geladen werden muss. Danach kommen Modell und Tokenizer aus dem Cache.
    """
    if _BASE_CACHE["model"] is not None:
        return _BASE_CACHE["model"], _BASE_CACHE["tokenizer"]

    # Der Tokenizer wandelt Text in Token-IDs um, damit das LLM damit arbeiten kann.
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
    if tokenizer.pad_token is None:
        # Falls kein eigenes Pad-Token definiert ist, wird das EOS-Token genutzt.
        tokenizer.pad_token = tokenizer.eos_token

    # Das eigentliche Sprachmodell wird lokal geladen.
    model = AutoModelForCausalLM.from_pretrained(
        HF_BASE_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # Fuer reine Klassifikation soll das Modell nur antworten, nicht weiter trainieren.
    model.config.use_cache = False
    model.generation_config.max_length = None
    model.eval()

    _BASE_CACHE["model"] = model
    _BASE_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def load_trained_model():
    """Laedt das Basismodell plus gespeicherten LoRA-Adapter.

    Falls der Adapter noch nicht existiert, wird ein Fehler geworfen,
    damit die App klar sagt: erst trainieren, dann testen.
    """
    if _TRAINED_CACHE["model"] is not None:
        return _TRAINED_CACHE["model"], _TRAINED_CACHE["tokenizer"]

    adapter_file = ADAPTER_DIR / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError("Kein trainiertes Modell gefunden. Bitte zuerst LoRA trainieren.")

    # Tokenizer kommt direkt aus dem Adapter-Ordner, damit genau die gleiche
    # Tokenisierung verwendet wird wie nach dem Training.
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Das Basismodell wird geladen und dann mit dem LoRA-Adapter kombiniert.
    base_model, _ = load_base_model()
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    model.generation_config.max_length = None

    _TRAINED_CACHE["model"] = model
    _TRAINED_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def classify_text(model, tokenizer, text):
    """Erzeugt fuer einen Text eine Modellvorhersage als Label.

    Ablauf:
    1. Text in Prompt-Form bringen
    2. Prompt tokenisieren
    3. Antwort vom Modell generieren
    4. Ausgabe auf ein Klassenlabel normalisieren
    """
    inputs = tokenizer(
        make_prompt(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        # Nur wenige neue Tokens erzeugen, weil die Antwort kurz sein soll.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return normalize_label(decoded)


class TextDataset(Dataset):
    """Ein kleines PyTorch-Dataset fuer das LoRA-Training.

    Aus jedem Trainingsbeispiel wird eine Sequenz gebaut, die aus Prompt und
    Zielantwort besteht. Der Prompt wird beim Loss ausgeblendet (`-100`),
    damit das Modell nur auf die gewuenschte Antwort optimiert wird.
    """

    def __init__(self, rows, tokenizer):
        self.items = []
        for row in rows:
            # Prompt und Zielantwort werden getrennt tokenisiert.
            prompt_ids = tokenizer(make_prompt(row["text"]), add_special_tokens=False)["input_ids"][:MAX_LENGTH]
            answer_ids = tokenizer(row["label"] + "</s>", add_special_tokens=False)["input_ids"][:MAX_LENGTH]

            # Eingabe besteht aus Prompt + Antwort.
            input_ids = (prompt_ids + answer_ids)[:MAX_LENGTH]

            # Nur die Antwort soll den Loss bekommen, nicht der Prompt.
            labels = ([-100] * len(prompt_ids) + answer_ids)[:MAX_LENGTH]
            attention_mask = [1] * len(input_ids)

            padding = MAX_LENGTH - len(input_ids)
            if padding > 0:
                # Auf feste Laenge auffuellen, damit alle Beispiele gleich lang sind.
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
    """Trainiert den LoRA-Adapter auf den CSV-Trainingsdaten.

    Nach dem Training werden zwei Artefakte gespeichert:
    - `adapter/`: die LoRA-Gewichte und der Tokenizer
    - `training_info.json`: Metadaten zum letzten Training
    """

    rows = load_samples()
    print(f"\nTrainingsbeispiele: {len(rows)}")
    print("\nEpochen: ", TRAIN_EPOCHS)

    # Tokenizer fuer das Basismodell laden.
    tokenizer = AutoTokenizer.from_pretrained(HF_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Vollstaendiges Basismodell laden.
    model = AutoModelForCausalLM.from_pretrained(
        HF_BASE_MODEL,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # LoRA erweitert nur einen kleinen Teil des Modells um trainierbare Gewichte.
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

    # Training auf CPU, bewusst klein gehalten.
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

    # Adapter-Ordner anlegen und trainierte Dateien speichern.
    ADAPTER_DIR.mkdir(exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Trainingsmetadaten fuer spaeteres Nachschlagen speichern.
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
    """Fragt Text per Konsole ab und liefert bei leerer Eingabe einen Standardtext."""
    text = input("\nText eingeben [Enter = Beispiel]: ").strip()
    return text or DEFAULT_TEXT


def test_base_model():
    """Testet nur das Basismodell ohne LoRA-Adapter."""
    text = input_text()
    model, tokenizer = load_base_model()
    print("\nKlasse:", classify_text(model, tokenizer, text))


def test_trained_model():
    """Testet das Modell mit geladenem LoRA-Adapter."""
    try:
        model, tokenizer = load_trained_model()
    except FileNotFoundError:
        print("\nBitte zuerst trainieren.")
        return

    text = input_text()
    print("\nKlasse:", classify_text(model, tokenizer, text))


def show_training_status():
    """Zeigt die Inhalte von `training_info.json` an."""
    if not TRAIN_INFO_FILE.exists():
        print("\nNoch kein Training durchgeführt.")
        return

    # Die Datei enthaelt die wichtigsten Kennzahlen des letzten Trainings.
    info = json.loads(TRAIN_INFO_FILE.read_text(encoding="utf-8"))
    print("\n--- Trainingsstatus ---")
    print("Trainiert:", info["trained"])
    print("Zeitpunkt:", info["trained_at"])
    print("Trainingsbeispiele:", info["training_examples"])
    print("Epochen:", info["epochs"])
    print("Train Loss:", info["train_loss"])
    print("Dauer:", info["runtime_seconds"], "Sekunden")


def main():
    """Startet die einfache Konsolenmenuefuehrung der App."""
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
