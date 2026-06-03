from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


MODEL_DIR = "./trained_model"


def generate_answer(question: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    tokenizer.pad_token = tokenizer.eos_token

    prompt = f"Frage: {question}\nAntwort:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def main():
    print("Mini-LLM Frage-Antwort-Konsole")
    print("Mit 'exit' beenden.\n")

    while True:
        question = input("Frage: ").strip()

        if question.lower() in ["exit", "quit", "ende"]:
            break

        answer = generate_answer(question)
        print("\n" + answer + "\n")


if __name__ == "__main__":
    main()