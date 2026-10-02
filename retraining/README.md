# Retraining – ganz einfach erklärt

Diese Datei erklärt das Skript `retraining.py` für Anfänger.

## Worum geht es?

Die App zeigt, wie man ein Modell **nochmal besser trainiert**, wenn man neue gute Beispiele bekommt.

Das nennt man **Retraining**.

Ganz einfach gesagt:
- erst lernt das Modell Version 1 (`V1`)
- dann kommen neue, geprüfte Beispiele dazu
- daraus wird Version 2 (`V2`)
- danach vergleicht man beide Modelle

## Warum macht man das?

Ein Modell ist am Anfang oft noch nicht perfekt.
Es kennt nur wenige Beispiele.

Wenn später neue, fachlich geprüfte Beispiele dazukommen, kann das Modell dazulernen.

So wird es oft besser bei Dingen, die es vorher noch nicht gut konnte.

## Was macht das Skript genau?

### 1. Testdaten anlegen

Die Testdaten bleiben für beide Modelle gleich.

Das ist wichtig, damit der Vergleich fair ist.

Wenn man andere Testdaten nehmen würde, könnte man die Modelle nicht richtig vergleichen.

### 2. Modell V1 trainieren

`V1` lernt nur aus wenigen alten Beispielen.

Es kennt also noch nicht so viele Formulierungen.

### 3. Neue Ground-Truth-Daten hinzufügen

Dann werden neue, geprüfte Beispiele ergänzt.

Diese Beispiele wurden also bewusst überprüft und sind nicht einfach geraten.

Das sind neue Lernbeispiele für das Modell.

### 4. Modell V2 trainieren

`V2` bekommt nun:
- die alten Daten
- plus die neuen geprüften Daten

Dadurch kennt es mehr passende Formulierungen.

### 5. Beide Modelle vergleichen

Dann wird geschaut:
- wie gut war `V1`?
- wie gut war `V2`?
- hat sich der F1-Wert verbessert?

## Was ist die Idee dahinter?

Die App zeigt dir:

> Mehr gute und geprüfte Trainingsdaten können das Modell besser machen.

Wenn ein Modell bei neuen Texten Fehler macht, kann man diese Fehler analysieren und danach neue Beispiele ergänzen.

So wird aus Fehleranalyse eine Verbesserung.

## Warum bleibt das Testset gleich?

Weil man fair vergleichen will.

Wenn `V1` und `V2` auf verschiedenen Testdaten getestet würden, wüsste man nicht, ob das Modell wirklich besser geworden ist.

Darum gilt:
- gleiches Testset für beide Modelle
- dann ist der Vergleich ehrlich und fair

## Was lernt man daraus?

- Was Retraining ist
- Warum neue gute Daten wichtig sind
- Warum Fehleranalyse nützlich ist
- Warum man Modelle fair vergleichen muss
- Warum ein Modell nach dem Nachtrainieren oft besser wird

## Merksatz

> Erst Fehler verstehen, dann gute neue Beispiele hinzufügen, dann Modell neu trainieren.

## Ausführen

Im Ordner mit der Datei:

```powershell
python retraining.py
```

Falls `pandas`, `scikit-learn` oder andere Pakete fehlen:

```powershell
pip install pandas scikit-learn
```

## Kurz gesagt

Diese App zeigt:
- Modell V1 = alter Stand
- neue geprüfte Beispiele = mehr Lernstoff
- Modell V2 = neu trainierte Version
- gleiches Testset = fairer Vergleich

Damit sieht man gut, ob Retraining wirklich geholfen hat.

