# Ground Truth und Error Analysis

Dieses Beispiel zeigt, wie man ein Textklassifikationsmodell nicht nur bewertet, sondern seine Fehler auch gezielt untersucht.

## Worum geht es?

Das Skript `ground_truth_error_analysis.py` trainiert ein einfaches Modell auf zwei Klassen:

- `relevant`
- `nicht_relevant`

Danach werden neue Testtexte vorhergesagt und mit der **Ground Truth** verglichen.

## Was ist Ground Truth?

**Ground Truth** bedeutet die bekannte richtige Antwort.

Beispiel:

- Text: `Ich kenne Personen aus dieser Gruppe.`
- Ground Truth: `relevant`

Mit der Ground Truth kannst du prüfen, ob das Modell richtig oder falsch liegt.

## Warum ist das nützlich?

Nicht nur die Gesamtgenauigkeit ist wichtig, sondern auch:

- Welche Fehler macht das Modell?
- Bei welchen Formulierungen passiert das?
- Gibt es systematische Muster?
- Fehlen passende Trainingsbeispiele?

Genau dafür ist **Error Analysis** da.

## Verwendete Begriffe

### True Positive (TP)

- Ground Truth: `relevant`
- Vorhersage: `relevant`

Das Modell liegt richtig.

### True Negative (TN)

- Ground Truth: `nicht_relevant`
- Vorhersage: `nicht_relevant`

Das Modell liegt richtig.

### False Positive (FP)

- Ground Truth: `nicht_relevant`
- Vorhersage: `relevant`

Das Modell meldet etwas als relevant, obwohl es nicht relevant ist.

### False Negative (FN)

- Ground Truth: `relevant`
- Vorhersage: `nicht_relevant`

Das Modell übersieht etwas Relevantes.

## Wie funktioniert das Skript?

Das Skript verwendet eine Pipeline mit zwei Schritten:

1. **`TfidfVectorizer`**
   - wandelt Texte in Zahlen um
   - wichtige Wörter bekommen höhere Gewichte

2. **`LogisticRegression`**
   - lernt aus diesen Zahlen, welche Klasse zu einem Text passt

Danach werden für die Testdaten ausgegeben:

- die Vorhersage
- die Wahrscheinlichkeit für `relevant`
- der Fehlertyp (`TP`, `TN`, `FP`, `FN`)

## Was zeigt die Ausgabe?

Das Skript gibt unter anderem aus:

- alle Testfälle mit Vorhersage
- `classification_report`
- `confusion_matrix`
- alle False Positives
- alle False Negatives
- alle Fehlerfälle insgesamt

So sieht man nicht nur, **wie gut** das Modell ist, sondern auch **wo** es scheitert.

## Was kannst du daraus lernen?

Dieses Beispiel hilft dir, die wichtigsten Grundlagen von Machine Learning besser zu verstehen:

- Was ist eine Ground Truth?
- Was ist eine Vorhersage?
- Was bedeuten FP und FN?
- Warum reicht eine einzige Metrik oft nicht aus?
- Wie analysiert man Fehler systematisch?

## Typische Fragen bei der Fehleranalyse

Wenn ein Fehler auftritt, kannst du fragen:

- Welche Wörter haben das Modell beeinflusst?
- Gibt es ähnliche Beispiele im Training?
- Ist das Label eindeutig?
- Ist der Threshold passend?
- Gibt es ein Muster bei den Fehlklassifikationen?

## So gehst du am besten vor

Am einfachsten ist es, wenn du Fehleranalyse in dieser Reihenfolge machst:

1. **Fehler automatisch finden**
   - Schaue dir `classification_report` und `confusion_matrix` an.
   - Filtere danach die falschen Vorhersagen.

2. **Fehlerfälle einzeln lesen**
   - Schau dir den genauen Text an.
   - Vergleiche `Ground Truth` und `Vorhersage`.
   - Prüfe die `confidence_relevant`.

3. **Ursachen vermuten**
   - Fehlen ähnliche Trainingsbeispiele?
   - Gibt es missverständliche Wörter oder Formulierungen?
   - Ist das Label vielleicht nicht eindeutig?

4. **Gezielt verbessern**
   - Mehr passende Trainingsdaten ergänzen.
   - Labels prüfen und korrigieren.
   - Features oder Modell anpassen.

## Gute Leitfragen für die Analyse

Wenn du einen Fehler siehst, helfen diese Fragen besonders:

- War der Text für einen Menschen selbst schon schwer eindeutig?
- Gibt es Wörter, die das Modell falsch gewichtet haben könnte?
- Ist das ein einmaliger Ausreißer oder ein wiederkehrendes Muster?
- Würde ein ähnlicher Satz im Training wahrscheinlich besser funktionieren?
- Sollte man den Threshold verändern, wenn das Modell zu oft zu sicher ist?

## Ausführen

Im Ordner mit dem Skript:

```powershell
python ground_truth_error_analysis.py
```

Falls du die Pakete noch nicht installiert hast:

```powershell
pip install pandas scikit-learn
```

## Lern-Tipp

Ändere testweise einzelne Testtexte und beobachte:

- ändert sich die Vorhersage?
- ändert sich die Confidence?
- entsteht ein False Positive oder False Negative?

So verstehst du das Thema viel schneller als nur durch Theorie.

## Merksatz

> Nicht nur zählen, wie viele Fehler ein Modell macht, sondern verstehen, **welche** Fehler es macht und **warum**.

