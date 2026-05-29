import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Training data
# training_text = """
# ich esse gerne pizza
# ich esse gerne pasta
# ich trinke gerne wasser
# du trinkst gerne kaffee
# du isst gerne pizza
# wir essen heute pizza
# wir trinken heute wasser
# ich gehe heute einkaufen
# du gehst heute arbeiten
# """

training_text = """
Deep Learning hat die Entwicklung moderner Sprachmodelle grundlegend verändert. Insbesondere rekurrente neuronale Netze wie Long Short-Term Memory Modelle, kurz LSTM, eignen sich für die Verarbeitung von Textsequenzen und die Vorhersage des nächsten Wortes innerhalb eines Satzes. Ein LSTM kann Informationen aus vorherigen Wörtern speichern und relevante Zusammenhänge über längere Textabschnitte hinweg berücksichtigen. Dadurch entstehen natürlich wirkende Wortfolgen und flüssige Texte.
Die Wortvorhersage basiert darauf, dass das Modell aus großen Mengen von Trainingsdaten statistische Muster und sprachliche Strukturen lernt. Während des Trainings analysiert das Netzwerk Millionen von Wortkombinationen, Satzanfängen und grammatikalischen Beziehungen. Auf Basis dieser Informationen kann das Modell abschätzen, welches Wort mit hoher Wahrscheinlichkeit als nächstes folgt. Beispielsweise könnte auf die Wortfolge „Künstliche Intelligenz verändert“ das Wort „die“, „unsere“ oder „zunehmend“ folgen, abhängig vom zuvor gelernten Kontext.
LSTM-Modelle besitzen spezielle Speicherzellen und sogenannte Gates, die entscheiden, welche Informationen behalten, aktualisiert oder vergessen werden. Dadurch können auch längere Zusammenhänge innerhalb eines Textes erkannt werden. Dies ist besonders wichtig bei komplexen Sätzen, bei denen sich die Bedeutung erst durch mehrere vorherige Wörter ergibt. Klassische neuronale Netze stoßen hierbei schnell an ihre Grenzen, da frühere Informationen verloren gehen können.
Die Einsatzgebiete von Wortvorhersagen mit LSTM-Netzen sind vielfältig. Sie reichen von automatischer Textergänzung auf Smartphones über intelligente Chatbots bis hin zu Übersetzungssystemen und Suchmaschinen. Auch in der Softwareentwicklung werden Sprachmodelle genutzt, um Quellcode vorherzusagen oder Entwicklern passende Vorschläge während des Programmierens anzuzeigen.
Ein weiterer Vorteil von LSTM-Modellen liegt in ihrer Fähigkeit, unterschiedliche Sprachstile zu erlernen. Wird ein Modell beispielsweise mit wissenschaftlichen Texten trainiert, erzeugt es eher formelle und sachliche Sprache. Bei Trainingsdaten aus sozialen Netzwerken entstehen dagegen deutlich informellere Formulierungen. Die Qualität der Wortvorhersage hängt daher stark von der Menge und Vielfalt der verwendeten Trainingsdaten ab.
Obwohl moderne Transformer-Modelle heute in vielen Bereichen leistungsfähiger sind, stellen LSTM-Netze weiterhin einen wichtigen Einstieg in das Deep Learning für Sprachverarbeitung dar. Sie sind vergleichsweise verständlich aufgebaut und eignen sich hervorragend, um grundlegende Konzepte neuronaler Sprachmodelle und sequenzieller Datenverarbeitung zu erlernen.
"""

# 2. Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([training_text])
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
vocab_size = len(word_index) + 1

print("Dictionary:")
print(word_index)
print()

# 3. Create training sequences
sequences = []
for line in training_text.strip().split("\n"):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        sequence = token_list[:i + 1]
        sequences.append(sequence)

max_sequence_len = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding="pre")
X = sequences[:, :-1]
y = sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# print("Training sequences (word indices):")
# for seq in sequences:
#     print(seq)
# print()

# 4. Define LSTM model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_sequence_len - 1),
    LSTM(50),
    Dense(vocab_size, activation="softmax")
])
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)
# model.summary()

# 5. Training
print("Training läuft...")  # Add this before model.fit
model.fit(X, y, epochs=300, verbose=0)
print("\nTraining abgeschlossen.\n")  # This stays as is, shown after training

# 6. Prediction functions
def predict_next_words(text, top_k=5):
    token_list = tokenizer.texts_to_sequences([text])[0]
    print("token_list ", token_list)
    if not token_list:
        print("Keine bekannten Wörter in der Eingabe.")
        return []
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predictions = model.predict(token_list, verbose=0)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    results = []
    for index in top_indices:
        if index == 0:
            continue
        word = index_word.get(index, "<unbekannt>")
        probability = predictions[index]
        results.append((word, probability))
    return results

def generate_text(seed_text, next_words=5):
    result = seed_text
    for _ in range(next_words):
        predictions = predict_next_words(result, top_k=1)
        if not predictions:
            break
        next_word = predictions[0][0]
        result += " " + next_word
    return result

# 7. Console interaction
print("Next Word Prediction mit LSTM")

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
    print("predictions ", predictions)
    print("\nWahrscheinlichste nächste Wörter:")
    for word, probability in predictions:
        word_id = word_index.get(word, 0)
        print(f"- {word} ({word_id}): {probability:.4f}")
    print()