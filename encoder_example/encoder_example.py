from transformers import pipeline


def main():
    print("====================================")
    print("ENCODER - TEXTKLASSIFIKATION")
    print("====================================")

    # DistilBERT ist ein Encoder-Modell.
    # Dieses konkrete Modell wurde für englische Sentiment-Klassifikation trainiert.
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    while True:
        print("\nText eingeben.")
        print("Mit 'exit' beenden.")

        text = input("\nText: ").strip()

        if text.lower() == "exit":
            break

        if not text:
            print("Bitte Text eingeben.")
            continue

        # Text wird durch das Encoder-Modell klassifiziert.
        result = classifier(text)[0]

        label = result["label"]
        score = result["score"]

        print("\n--- Ergebnis ---")
        print("Label:", label)
        print(f"Score: {score:.4f}")

        print("Es ist nicht gut" if label == "NEGATIVE" else "Alles wird gut")




if __name__ == "__main__":
    main()