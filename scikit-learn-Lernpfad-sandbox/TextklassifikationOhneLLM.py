import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. BEISPIELDATEN ERSTELLEN
# ============================================================

# Jeder Text besitzt eine bekannte Klasse.
#
# Das ist unser Trainingsdatensatz.
# In einem echten Projekt würden diese Labels z. B.
# von Menschen annotiert werden.

daten = [
    ["Ich habe starke Rückenschmerzen.", "symptom"],
    ["Seit gestern habe ich Kopfschmerzen.", "symptom"],
    ["Mir ist häufig schwindelig.", "symptom"],
    ["Ich bekomme schlecht Luft.", "symptom"],
    ["Mein Bauch tut sehr weh.", "symptom"],
    ["Ich habe seit drei Tagen Fieber.", "symptom"],

    ["Ich nehme täglich Ibuprofen.", "medikament"],
    ["Ich nehme morgens eine Tablette.", "medikament"],
    ["Der Arzt hat mir Paracetamol verschrieben.", "medikament"],
    ["Ich nehme regelmäßig Blutdrucktabletten.", "medikament"],
    ["Abends nehme ich meine Medikamente.", "medikament"],
    ["Ich benutze ein Asthmaspray.", "medikament"],

    ["Ich habe keine Allergien.", "negation"],
    ["Ich nehme keine Medikamente.", "negation"],
    ["Nein, ich habe keine Vorerkrankungen.", "negation"],
    ["Ich rauche nicht.", "negation"],
    ["Ich habe keine Schmerzen.", "negation"],
    ["Nein, Fieber habe ich nicht.", "negation"],

    # --------------------------------------------------------
    # 100 zusätzliche Beispiele
    # --------------------------------------------------------

    ["Seit heute Morgen habe ich Halsschmerzen.", "symptom"],
    ["Meine Knie tun beim Gehen weh.", "symptom"],
    ["Ich habe einen starken Druck auf der Brust.", "symptom"],
    ["Seit zwei Tagen ist mir übel.", "symptom"],
    ["Ich muss ständig husten.", "symptom"],
    ["Meine Nase ist völlig verstopft.", "symptom"],
    ["Ich habe Schmerzen im rechten Ohr.", "symptom"],
    ["Seit gestern habe ich Durchfall.", "symptom"],
    ["Meine Augen brennen und tränen.", "symptom"],
    ["Ich fühle mich sehr schwach.", "symptom"],
    ["Beim Aufstehen wird mir schwindelig.", "symptom"],
    ["Ich habe stechende Schmerzen im Rücken.", "symptom"],
    ["Mein linker Arm fühlt sich taub an.", "symptom"],
    ["Ich habe seit heute Morgen Migräne.", "symptom"],
    ["Mein Hals ist stark geschwollen.", "symptom"],
    ["Ich habe Probleme beim Schlucken.", "symptom"],
    ["Seit einer Woche bin ich ständig müde.", "symptom"],
    ["Ich habe einen Hautausschlag am Arm.", "symptom"],
    ["Meine Hände zittern häufig.", "symptom"],
    ["Ich habe seit Tagen keinen Appetit.", "symptom"],
    ["Meine Gelenke sind morgens sehr steif.", "symptom"],
    ["Ich habe Schmerzen beim Wasserlassen.", "symptom"],
    ["Mein Herz schlägt manchmal sehr schnell.", "symptom"],
    ["Ich bekomme nachts starke Schweißausbrüche.", "symptom"],
    ["Ich habe seit gestern starke Zahnschmerzen.", "symptom"],
    ["Meine Füße sind stark angeschwollen.", "symptom"],
    ["Ich habe ein Brennen im Magen.", "symptom"],
    ["Seit drei Tagen habe ich Verstopfung.", "symptom"],
    ["Meine Stimme ist seit gestern heiser.", "symptom"],
    ["Ich sehe manchmal verschwommen.", "symptom"],
    ["Ich habe starke Schmerzen im Nacken.", "symptom"],
    ["Mein rechtes Bein fühlt sich schwer an.", "symptom"],
    ["Ich habe seit heute Morgen Ohrgeräusche.", "symptom"],
    ["Ich friere ständig, obwohl es warm ist.", "symptom"],

    ["Ich nehme morgens Metformin.", "medikament"],
    ["Abends nehme ich eine Tablette gegen Bluthochdruck.", "medikament"],
    ["Ich nehme bei Bedarf Paracetamol.", "medikament"],
    ["Mein Arzt hat mir Antibiotika verschrieben.", "medikament"],
    ["Ich benutze täglich ein Nasenspray.", "medikament"],
    ["Ich nehme regelmäßig Schilddrüsentabletten.", "medikament"],
    ["Ich nehme jeden Abend ein Schlafmittel.", "medikament"],
    ["Ich verwende eine Salbe gegen meinen Hautausschlag.", "medikament"],
    ["Ich nehme morgens und abends Pantoprazol.", "medikament"],
    ["Bei Kopfschmerzen nehme ich Aspirin.", "medikament"],
    ["Ich spritze mir täglich Insulin.", "medikament"],
    ["Ich benutze ein Kortisonspray.", "medikament"],
    ["Ich nehme einmal täglich Ramipril.", "medikament"],
    ["Ich bekomme regelmäßig eine Vitamin-B12-Spritze.", "medikament"],
    ["Ich nehme seit einer Woche ein Antibiotikum.", "medikament"],
    ["Ich nehme morgens eine Tablette gegen Allergien.", "medikament"],
    ["Ich benutze Augentropfen gegen trockene Augen.", "medikament"],
    ["Ich nehme täglich Magnesiumtabletten.", "medikament"],
    ["Ich nehme bei Bedarf ein Schmerzmittel.", "medikament"],
    ["Ich verwende eine Creme gegen Neurodermitis.", "medikament"],
    ["Ich nehme abends einen Cholesterinsenker.", "medikament"],
    ["Ich nehme morgens eine Eisentablette.", "medikament"],
    ["Ich benutze zweimal täglich mein Asthmaspray.", "medikament"],
    ["Ich nehme seit gestern Hustensaft.", "medikament"],
    ["Mein Arzt hat mir Diclofenac verschrieben.", "medikament"],
    ["Ich nehme täglich ein Medikament gegen Epilepsie.", "medikament"],
    ["Ich verwende regelmäßig ein Inhalationsspray.", "medikament"],
    ["Ich nehme morgens Omeprazol.", "medikament"],
    ["Ich nehme abends ein Beruhigungsmittel.", "medikament"],
    ["Ich nehme jeden Tag eine Tablette gegen Herzrhythmusstörungen.", "medikament"],
    ["Ich benutze eine antibiotische Salbe.", "medikament"],
    ["Ich nehme bei Bedarf Tropfen gegen Übelkeit.", "medikament"],
    ["Ich nehme regelmäßig ein Medikament gegen Migräne.", "medikament"],

    ["Ich habe keinen Husten.", "negation"],
    ["Nein, ich habe keine Atemnot.", "negation"],
    ["Ich nehme aktuell keine Tabletten.", "negation"],
    ["Ich habe keine bekannten Allergien.", "negation"],
    ["Fieber habe ich nicht.", "negation"],
    ["Ich habe keine Probleme beim Schlafen.", "negation"],
    ["Nein, mir ist nicht schwindelig.", "negation"],
    ["Ich habe keine Bauchschmerzen.", "negation"],
    ["Ich trinke keinen Alkohol.", "negation"],
    ["Ich nehme keine Schmerzmittel.", "negation"],
    ["Ich habe keine Herzprobleme.", "negation"],
    ["Nein, ich habe keinen Durchfall.", "negation"],
    ["Ich habe keine Übelkeit.", "negation"],
    ["Ich benutze keine Medikamente regelmäßig.", "negation"],
    ["Ich habe keine Beschwerden beim Wasserlassen.", "negation"],
    ["Ich habe keine Kopfschmerzen.", "negation"],
    ["Nein, ich bin nicht operiert worden.", "negation"],
    ["Ich habe keine chronischen Erkrankungen.", "negation"],
    ["Ich rauche überhaupt nicht.", "negation"],
    ["Ich habe keine Probleme mit dem Rücken.", "negation"],
    ["Ich habe keine Hautausschläge.", "negation"],
    ["Nein, ich habe keine Schlafstörungen.", "negation"],
    ["Ich nehme keine Blutdruckmedikamente.", "negation"],
    ["Ich habe keine Schmerzen in der Brust.", "negation"],
    ["Ich habe keine Probleme beim Atmen.", "negation"],
    ["Nein, ich habe keine Allergie gegen Penicillin.", "negation"],
    ["Ich habe keine Gelenkschmerzen.", "negation"],
    ["Ich nehme derzeit keine Antibiotika.", "negation"],
    ["Ich habe keine Beschwerden mit dem Magen.", "negation"],
    ["Ich habe keine Taubheitsgefühle.", "negation"],
    ["Nein, ich habe keine Sehstörungen.", "negation"],
    ["Ich habe keine Vorerkrankungen am Herzen.", "negation"],
    ["Ich nehme keine regelmäßigen Medikamente.", "negation"]
]

# Aus der Liste ein DataFrame erstellen.
df = pd.DataFrame(
    daten,
    columns=["text", "label"]
)

print("\n--- Datensatz ---")
print(df)


# ============================================================
# 2. X UND y DEFINIEREN
# ============================================================

# X = Texte, aus denen das Modell lernen soll.
X = df["text"]

# y = richtige Klasse zu jedem Text.
y = df["label"]

# ============================================================
# 4. TF-IDF + LOGISTIC REGRESSION
# ============================================================

# Ein ML-Modell kann mit normalen Sätzen nicht direkt rechnen.
#
# TF-IDF wandelt Wörter in Zahlen um.
#
# Vereinfacht:
#
# "Ich habe Kopfschmerzen"
#
# wird zu einem Zahlenvektor wie:
#
# [0.0, 0.4, 0.8, 0.0, ...]
#
# Wörter, die für bestimmte Texte besonders charakteristisch
# sind, bekommen dabei ein höheres Gewicht.

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer()
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# ============================================================
# 3. TRAIN UND TEST AUFTEILEN
# ============================================================

# 70 % Training
# 30 % Test
#
# stratify=y sorgt dafür, dass alle Klassen ungefähr
# gleichmäßig auf Train und Test verteilt bleiben.
for seed in [1, 2, 3, 4, 5]:
    X_train, X_test, y_train, y_test = train_test_split(
    X,
            y,
        test_size=0.3,
        random_state=seed,
        stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SEED:", seed)
    print(classification_report(y_test, y_pred))

    # Zeigt, welche Klassen richtig bzw. falsch vorhergesagt wurden
    matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(matrix)

    #                   symptom medikament negation
    #   Ist symptom         11      0        1
    #   Ist medikament      1       11       0
    #   Ist negation        0       0        12

# TF-IDF-Vektoren für die Testdaten holen
tfidf = model.named_steps["tfidf"]
X_test_tfidf = tfidf.transform(X_test)

# 1) Dichte Matrix (nur bei kleinen Datenmengen sinnvoll)
print("TF-IDF Shape:", X_test_tfidf.shape)
print(X_test_tfidf.toarray())

# 2) Mit Spaltennamen als DataFrame (lesbarer)
tfidf_df = pd.DataFrame(
    X_test_tfidf.toarray(),
    columns=tfidf.get_feature_names_out(),
    index=X_test.index
)
print(tfidf_df.head())





# ============================================================
# 5. MODELL TRAINIEREN
# ============================================================

# TF-IDF lernt dabei zunächst das Vokabular.
# Danach lernt LogisticRegression die Zusammenhänge
# zwischen Wörtern und Klassen.

# model.fit(X_train, y_train)


# ============================================================
# 6. TESTDATEN VORHERSAGEN
# ============================================================

#y_pred = model.predict(X_test)

vergleich = pd.DataFrame({
    "Text": X_test,
    "Tatsächlich": y_test,
    "Vorhersage": y_pred
})

print("\n--- Testdaten ---")
print(vergleich)


# ============================================================
# 7. MODELL AUSWERTEN
# ============================================================

print("\n--- Precision / Recall / F1 ---")
print(classification_report(y_test, y_pred))


# ============================================================
# 8. EIGENE TEXTE TESTEN
# ============================================================

texte = [
    "Seit heute Morgen habe ich starke Schmerzen.",
    "Ich nehme jeden Abend Aspirin.",
    "Ich habe keine bekannten Allergien."
]

vorhersagen = model.predict(texte)

print("\n--- Eigene Vorhersagen ---")

for text, prediction in zip(texte, vorhersagen):
    print(text)
    print("→", prediction)
    print()