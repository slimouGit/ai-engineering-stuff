# Analyse-Ablauf: Ground Truth und Error Analysis

Diese separate README erklärt, wie du das Beispiel `ground_truth_error_analysis.py` für das Verständnis von Fehleranalyse nutzen kannst.

## Worum geht es?

Das Skript zeigt ein einfaches Textklassifikationsmodell für zwei Klassen:

- `relevant`
- `nicht_relevant`

Es vergleicht die Vorhersagen des Modells mit der **Ground Truth** und macht die Fehler sichtbar.

## Was ist die Idee hinter der Analyse?

Nicht nur die Gesamtleistung ist wichtig, sondern vor allem:

- Welche Fehler macht das Modell?
- Warum macht es diese Fehler?
- Welche Beispiele führen zu Verwechslungen?
- Was kann man an Daten, Labels oder Modell verbessern?

## Empfohlener Ablauf

Am besten gehst du bei der Fehleranalyse in dieser Reihenfolge vor:

### 1. Fehler automatisch finden

Zuerst lässt du das Modell auf die Testdaten laufen und schaust dir an:

- `classification_report`
- `confusion_matrix`
- Tabelle mit `Ground Truth` und `Vorhersage`

So erkennst du schnell, **wo** Fehler auftreten.

### 2. Fehlerfälle einzeln lesen

Danach schaust du dir die falschen Vorhersagen einzeln an.

Fragen dabei:

- Was war der genaue Text?
- Was war die richtige Klasse?
- Was hat das Modell vorhergesagt?
- Wie hoch war die Confidence?

### 3. Ursachen vermuten

Jetzt überlegst du, warum der Fehler passiert ist.

Mögliche Ursachen:

- ähnliche Wörter wie in anderen Klassen
- zu wenige passende Trainingsbeispiele
- unklare oder mehrdeutige Formulierungen
- Label ist nicht eindeutig
- Threshold ist zu niedrig oder zu hoch

### 4. Gezielt verbessern

Zum Schluss leitest du Verbesserungen ab:

- mehr passende Beispiele ergänzen
- Labels prüfen und korrigieren
- Threshold anpassen
- Modell oder Features verbessern

## Beispiel: False Positive

Ein konkreter Fehler aus dem Beispiel ist:

- Text: `Ich hatte nur Kontakt zur Behörde wegen meines Termins.`
- Ground Truth: `nicht_relevant`
- Vorhersage: `relevant`
- Confidence: `0.545`

### Was bedeutet das?

Das Modell hat den Text fälschlich als relevant eingestuft.
Die Confidence ist nur knapp über 0.5, also ist das Modell eher unsicher.

### Warum kann das passieren?

Wahrscheinlich hat das Modell das Wort **Kontakt** zu stark gewichtet.
Es erkennt ein ähnliches Wortmuster wie in relevanten Trainingsbeispielen, zum Beispiel:

- `Ich hatte Kontakt zu einer bewaffneten Gruppe.`

Das Modell versteht den Inhalt nicht wie ein Mensch, sondern reagiert auf Wortmuster.

### Wie könnte man das verbessern?

- mehr ähnliche `nicht_relevant`-Beispiele hinzufügen
- Formulierungen mit `Kontakt zur Behörde` ins Training aufnehmen
- prüfen, ob der Threshold zu locker ist
- weitere Beispiele mit harmlosen Verwaltungs- oder Termintexten ergänzen

## Was solltest du daraus lernen?

Dieses Beispiel zeigt dir sehr gut:

- wie Ground Truth funktioniert
- wie man Fehler nicht nur zählt, sondern versteht
- wie man aus einzelnen Fehlern Verbesserungen ableitet
- warum man bei Textmodellen immer die Beispiele lesen sollte

## Merksatz

> Fehleranalyse heißt nicht nur sehen, dass ein Modell falsch lag, sondern verstehen, **warum** es falsch lag und **wie** man es verbessern kann.

## Ausführen

Im Ordner mit dem Skript:

```powershell
python ground_truth_error_analysis.py
```

Falls die Pakete fehlen:

```powershell
pip install pandas scikit-learn
```

