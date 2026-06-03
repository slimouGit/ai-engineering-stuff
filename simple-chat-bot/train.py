from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


BASE_MODEL = "distilgpt2"
OUTPUT_DIR = "./trained_model"
TRAINING_FILE = "training_data.txt"


def load_training_text(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    examples = [block.strip() for block in text.split("\n\n") if block.strip()]
    return examples


def main():
    examples = load_training_text(TRAINING_FILE)

    dataset = Dataset.from_dict({
        "text": examples
    })

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenized_dataset = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=2,
        save_strategy="epoch",
        logging_steps=1,
        learning_rate=5e-5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Modell gespeichert unter: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()