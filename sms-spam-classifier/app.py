"""
Einfache Konsolen-Anwendung (CLI),
mit der der Nutzer interaktiv SMS prüfen kann.
"""

from predict.predict import predict_sms


def main():
    """
    Hauptschleife der Anwendung.
    """

    print("SMS Spam Classifier")
    print("-------------------")
    print("Mit 'exit' beenden.\n")

    while True:
        # Nutzereingabe
        sms = input("SMS: ")

        # Abbruchbedingungen
        if sms.lower() in ["exit", "quit", "q"]:
            print("App beendet.")
            break

        if not sms.strip():
            print("Bitte gültigen Text eingeben.\n")
            continue

        # Vorhersage durchführen
        result = predict_sms(sms)

        # Ergebnis anzeigen
        print("\nVorhersage:", result["prediction"])
        print("Spam-Wahrscheinlichkeit:", result["spam_probability"])
        print()


if __name__ == "__main__":
    main()