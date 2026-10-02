# Warum diese Entscheidungen in diesem ML-Beispiel?

## 1. Warum verwende ich hier TF-IDF?

Weil die Eingaben Texte sind und der Klassifikator Zahlen braucht.

TF-IDF wandelt Sätze in numerische Vektoren um:
- Wörter werden gezählt
- aber wichtigere Wörter stärker gewichtet
- weniger wichtige Wörter bekommen weniger Gewicht

Das ist für Textklassifikation sehr geeignet, weil Wörter wie „Fieber“, „Schmerzen“, „Medikament" in einem Satz viel aussagekräftiger sind als sehr häufige Wörter wie „ich" oder „der".

Ohne TF-IDF könnte ein Modell mit Rohtexten nicht rechnen.

## 2. Warum Logistic Regression statt K-Means?

Weil hier eine **überwachte Klassifikation** vorliegt.

- `K-Means` ist ein **unüberwachtes Clustering-Verfahren**
- Es kennt keine Zielklassen wie `symptom`, `medikament`, `negation`
- Es gruppiert Daten nur nach Ähnlichkeit

Bei diesem Beispiel wissen wir aber schon, welche Texte zu welcher Klasse gehören. Deshalb ist ein überwachtes Modell wie `LogisticRegression` passend.

Kurz:
- K-Means: finde Muster ohne Labels
- Logistic Regression: lerne aus bekannten Labels

## 3. Warum brauche ich Ground Truth?

Ground Truth ist die **bekannte richtige Antwort**.

Ohne Ground Truth kann ich nicht wissen, ob das Modell richtig liegt.

Im Beispiel sind die Labels wie `relevant` oder `nicht_relevant` die Ground Truth.

Sie dient dazu:
- Vorhersagen zu vergleichen
- Fehler zu erkennen
- Maße wie Precision, Recall, F1 und Accuracy zu berechnen
- Fehleranalyse zu machen

Ohne Ground Truth ist das Modell nur ein Black Box ohne echte Bewertung.

## 4. Was sagt mir ein Recall von 0,70?

Recall bedeutet:

Von allen tatsächlich positiven Fällen wurden 70 % korrekt erkannt.

Formel:
- Recall = TP / (TP + FN)

Beispiel:
- 70 relevante Texte existieren
- das Modell erkennt 49 davon
- dann ist Recall = 49 / 70 = 0,70

Das heißt:
- das Modell verpasst 30 % der relevanten Fälle
- das ist wichtig, wenn Fehlermeldungen in der positiven Klasse teuer sind

Ein niedriger Recall bedeutet oft:
- zu viele False Negatives
- das Modell erkennt wichtige Fälle nicht

## 5. Was passiert, wenn ich den Threshold absenke?

Der Threshold entscheidet, ab wann ein Modell eine Klasse als positiv annimmt.

Wenn du den Threshold senkst, wird das Modell eher „positiv" sagen.

Das hat Folgen:
- mehr True Positives
- aber auch mehr False Positives
- Recall steigt meistens
- Precision fällt oft

Beispiel:
- bei 0,50 wird ein Text als relevant angenommen, sobald die Wahrscheinlichkeit >= 0,50 ist
- bei 0,30 würde schon bei 0,31 entschieden

Das heißt: weniger streng, mehr positive Vorhersagen, aber mehr Fehlalarme.

## 6. Warum kann ein F1 von 1,0 bei einem einzelnen Split irreführend sein?

Weil ein einzelner Split nur eine mögliche Aufteilung ist.

Wenn der Datensatz klein ist, kann ein zufällig günstiger Split entstehen:
- Train/Test-Aufteilung passt besonders gut
- die Daten sind leicht zu trennen
- F1 erscheint perfekt, obwohl das Modell in anderen Splits deutlich schlechter sein kann

Das ist oft ein Zufallseffekt, kein Zeichen für echte Robustheit.

Deshalb braucht man Cross-Validation oder mehrere Splits.

## 7. Warum Cross-Validation?

Cross-Validation untersucht die Leistung **über mehrere Teilungen** des Datensatzes.

Statt nur einen zufälligen Split zu nehmen, werden mehrere Durchläufe gemacht:
- der Datensatz wird in mehrere Teile aufgeteilt
- jede Teilmenge dient einmal als Testset
- der Mittelwert der Ergebnisse wird berechnet

Das liefert eine stabilere Einschätzung der Modellleistung.

Vorteile:
- weniger Zufall
- bessere Schätzung der Generalisierung
- robustere Bewertung

## 8. Was mache ich mit False Negatives?

False Negatives sind Fälle, die tatsächlich relevant sind, aber vom Modell als nicht relevant erkannt wurden.

Das ist besonders wichtig, wenn man „nicht relevante" Fälle fälschlich als relevant erkennt oder relevante Fälle nicht erkennt.

Mit FN gehst du so vor:
- sie einzeln lesen
- Muster erkennen
- ähnliche Texte im Training ergänzen
- Threshold prüfen
- Labels überprüfen
- ggf. mehr Daten sammeln

Beispiel:
- Ein Text enthält relevante Informationen, aber die Formulierung ist ungewöhnlich
- das Modell erkennt ihn nicht
- du kannst diese Formulierung gezielt in den Trainingsdaten ergänzen

## 9. Wann retrainiere ich ein Modell?

Ein Modell sollte neu trainiert werden, wenn sich die Daten oder die Aufgabe ändern.

Typische Gründe:
- neue Trainingsdaten kommen hinzu
- Labels wurden korrigiert
- Fehlermuster wurden erkannt
- die Datenverteilung hat sich verändert (Drift)
- das Modell arbeitet in Produktion schlecht
- neue Kategorien oder neue Formulierungen treten auf

Kurz:
- wenn sich die Umgebung verändert oder das Modell nicht mehr gut genug funktioniert

## Merksatz

Ein gutes Modell ist nicht nur schnell und genau, sondern auch verständlich, robust und testbar.

Daher sind:
- Ground Truth
- Recall/Precision/F1
- Threshold
- Cross-Validation
- Fehleranalyse

wichtige Werkzeuge, um ML-Ergebnisse wirklich zu bewerten.

