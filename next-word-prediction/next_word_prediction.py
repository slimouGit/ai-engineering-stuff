# lstm_next_word_prediction.py

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ------------------------------------------------------------
# 1. Trainingsdaten
# ------------------------------------------------------------
# Das Modell lernt aus diesen Beispielsätzen.
# Ziel: Aus einer Wortfolge soll das jeweils nächste Wort vorhergesagt werden.
training_text = """
ich esse gerne pizza
ich esse gerne pasta
ich trinke gerne wasser
du trinkst gerne kaffee
du isst gerne pizza
wir essen heute pizza
wir trinken heute wasser
ich gehe heute einkaufen
du gehst heute arbeiten
"""


# ------------------------------------------------------------
# 2. Tokenisierung
# ------------------------------------------------------------
# Der Tokenizer wandelt Wörter in Zahlen um.
# Neuronale Netze können nicht direkt mit Wörtern rechnen,
# sondern brauchen numerische Eingaben.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([training_text])

# Beispiel:
# "ich" könnte die Zahl 1 bekommen,
# "gerne" die Zahl 2 usw.
word_index = tokenizer.word_index

# Um später aus vorhergesagten Zahlen wieder Wörter zu machen,
# wird das Dictionary umgedreht.
index_word = {v: k for k, v in word_index.items()}

# +1, weil Index 0 für Padding reserviert ist.
vocab_size = len(word_index) + 1

print("Dictionary:")
print(word_index)
print()


# ------------------------------------------------------------
# 3. Trainingssequenzen erzeugen
# ------------------------------------------------------------
# Aus jedem Satz werden mehrere Trainingsbeispiele erzeugt.
#
# Beispiel:
# Satz: "ich esse gerne pizza"
#
# Daraus entstehen:
# [ich, esse]              -> esse ist Zielwort
# [ich, esse, gerne]       -> gerne ist Zielwort
# [ich, esse, gerne, pizza] -> pizza ist Zielwort
#
# Später wird jeweils der letzte Token als Ziel y genutzt.
sequences = []

for line in training_text.strip().split("\n"):
    # Satz in Zahlenfolge umwandeln
    token_list = tokenizer.texts_to_sequences([line])[0]

    # Aus einem Satz mehrere Teilsequenzen bilden
    for i in range(1, len(token_list)):
        sequence = token_list[:i + 1]
        sequences.append(sequence)

# Die längste Sequenz bestimmt die spätere Eingabelänge.
max_sequence_len = max(len(seq) for seq in sequences)

# Alle Sequenzen müssen gleich lang sein.
# Kürzere Sequenzen werden vorne mit Nullen aufgefüllt.
sequences = pad_sequences(
    sequences,
    maxlen=max_sequence_len,
    padding="pre"
)

# X enthält alle Wörter außer dem letzten.
# y enthält jeweils das letzte Wort, das vorhergesagt werden soll.
X = sequences[:, :-1]
y = sequences[:, -1]

# y wird in One-Hot-Vektoren umgewandelt.
# Beispiel bei 5 Wörtern:
# Wort mit Index 3 -> [0, 0, 0, 1, 0]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)


# ------------------------------------------------------------
# 4. LSTM-Modell definieren
# ------------------------------------------------------------
model = Sequential([
    # Embedding-Schicht:
    # Wandelt Wort-IDs in lernbare Vektoren um.
    # Statt "ich" = 1 arbeitet das Modell mit einem dichten Vektor.
    Embedding(
        input_dim=vocab_size,
        output_dim=16,
        input_length=max_sequence_len - 1
    ),

    # LSTM-Schicht:
    # Verarbeitet die Wortfolge sequenziell.
    # Sie kann sich Kontext aus vorherigen Wörtern merken.
    LSTM(50),

    # Dense-Ausgabeschicht:
    # Gibt für jedes Wort im Dictionary einen Score aus.
    # Softmax macht daraus Wahrscheinlichkeiten.
    Dense(vocab_size, activation="softmax")
])

# Modell konfigurieren:
# - categorical_crossentropy: passend für Multi-Class Classification
# - Adam: Optimierungsverfahren
# - accuracy: misst, wie oft das wahrscheinlichste Wort korrekt ist
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

model.summary()


# ------------------------------------------------------------
# 5. Training
# ------------------------------------------------------------
# Das Modell sieht die Trainingsbeispiele mehrfach.
# In jeder Epoche werden die Gewichte angepasst.
model.fit(
    X,
    y,
    epochs=300,
    verbose=0
)

print("\nTraining abgeschlossen.\n")


# ------------------------------------------------------------
# 6. Vorhersagefunktion
# ------------------------------------------------------------
def predict_next_words(text, top_k=5):
    """
    Ermittelt die wahrscheinlichsten nächsten Wörter
    für einen eingegebenen Text.
    """

    # Eingabetext in Token-IDs umwandeln
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Falls kein bekanntes Wort enthalten ist, kann das Modell nichts Sinnvolles vorhersagen.
    if not token_list:
        print("Keine bekannten Wörter in der Eingabe.")
        return []

    # Eingabe auf gleiche Länge bringen wie beim Training
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    # Modell gibt Wahrscheinlichkeiten für alle Wörter im Dictionary zurück
    predictions = model.predict(token_list, verbose=0)[0]

    # Die top_k wahrscheinlichsten Wort-Indizes bestimmen
    top_indices = predictions.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        # Index 0 ist Padding und kein echtes Wort
        if index == 0:
            continue

        word = index_word.get(index, "<unbekannt>")
        probability = predictions[index]

        results.append((word, probability))

    return results


def generate_text(seed_text, next_words=5):
    """
    Erzeugt automatisch mehrere Wörter.
    Dabei wird immer das wahrscheinlichste nächste Wort angehängt.
    """

    result = seed_text

    for _ in range(next_words):
        predictions = predict_next_words(result, top_k=1)

        if not predictions:
            break

        next_word = predictions[0][0]
        result += " " + next_word

    return result


# ------------------------------------------------------------
# 7. Konsoleninteraktion
# ------------------------------------------------------------
print("Next Word Prediction mit LSTM")
print('Befehle: "exit" beenden, "auto <text>" generiert 5 Wörter\n')

while True:
    user_input = input("Text eingeben: ").strip()

    if user_input.lower() == "exit":
        break

    if user_input.lower().startswith("auto "):
        seed = user_input[5:]
        generated = generate_text(seed, next_words=5)
        print("\nGenerierter Text:")
        print(generated)
        print()
        continue

    predictions = predict_next_words(user_input, top_k=5)

    print("\nWahrscheinlichste nächste Wörter:")

    for word, probability in predictions:
        print(f"- {word}: {probability:.4f}")

    print()