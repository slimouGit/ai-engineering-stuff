# Threshold Metrics Demo

Eine kleine, nachvollziehbare Lern-App für:

- Precision
- Recall
- F1
- F2
- Threshold / Schwellenwert
- Modell speichern und wieder laden
- eigenen Threshold testen
- Metriken vergleichen
- eine gemeinsame Threshold-Grafik
- MLflow-Logging

## Beispiel-Use-Case

Die Daten bestehen aus kurzen Kundenservice-Nachrichten.

Das Modell entscheidet:

```text
dringend
oder
nicht_dringend
```

Beispiele:

```text
"Meine Karte wurde gestohlen."
=> dringend

"Wie ändere ich meine Adresse?"
=> nicht_dringend
```

Dadurch ist leicht nachvollziehbar, was die positive Klasse bedeutet.

## Installation

```powershell
py -m pip install -r requirements.txt
```

## MLflow starten

```powershell
py -m mlflow server --backend-store-uri sqlite:///mlflow.db
```

MLflow UI:

```text
http://localhost:5000
```

## App starten

```powershell
py app.py
```

## Menü

### 1 - Modell neu trainieren + alle Thresholds auswerten

Trainiert:

```text
Text
-> TF-IDF
-> Logistic Regression
-> Wahrscheinlichkeit "dringend"
```

Danach werden Thresholds von 0.05 bis 0.95 getestet.

Für jeden Threshold werden berechnet:

- Precision
- Recall
- F1
- F2

Die Ergebnisse landen in:

```text
threshold_metrics.csv
```

Die gemeinsame Grafik landet in:

```text
threshold_metrics.png
```

Das Modell wird gespeichert als:

```text
saved_model.joblib
```

### 2 - Eigenen Threshold testen und vergleichen

Du gibst zum Beispiel ein:

```text
0.30
```

Die App vergleicht deine Wahl mit:

- Standard-Threshold 0.50
- Threshold mit höchstem F1
- Threshold mit höchstem F2

Dadurch sieht man direkt, wie sich Precision, Recall, F1 und F2 verändern.

### 3 - Eigenen Kundentext testen

Du gibst beispielsweise ein:

```text
Ich sehe eine unbekannte Abbuchung auf meinem Konto.
```

und anschließend:

```text
0.30
```

Die App zeigt:

- Modellwahrscheinlichkeit für "dringend"
- verwendeten Threshold
- finale Klassifikation

### 4 - Werte anzeigen

Zeigt die komplette Tabelle aller getesteten Thresholds.

## Interpretation

Ein niedriger Threshold klassifiziert leichter als "dringend".

Typischer Effekt:

```text
Threshold sinkt
-> mehr positive Treffer
-> Recall steigt häufig
-> Precision sinkt häufig
```

Ein hoher Threshold ist strenger:

```text
Threshold steigt
-> weniger positive Treffer
-> Precision steigt häufig
-> Recall sinkt häufig
```

F1 bewertet Precision und Recall gleichgewichtet.

F2 gewichtet Recall stärker und ist deshalb interessant, wenn das Übersehen positiver Fälle besonders problematisch ist.
