# Class Imbalance – ganz einfach erklärt

Diese kleine Lern-App zeigt, was **Class Imbalance** bedeutet und warum das für Machine Learning wichtig ist.

## Worum geht es in der App?

Die Datei `class_imbalance.py` baut absichtlich einen Datensatz, in dem eine Klasse viel häufiger vorkommt als die andere.

- `nicht_relevant` kommt sehr oft vor
- `relevant` kommt nur selten vor

Genau das nennt man **Class Imbalance** oder auf Deutsch: **Klassen-Ungleichgewicht**.

## Was macht die App?

Die App vergleicht zwei Modelle:

1. **normales Logistic Regression Modell**
2. **Logistic Regression mit `class_weight='balanced'`**

Dann schaut sie, wie gut beide Modelle die seltene Klasse `relevant` erkennen.

## Warum ist das wichtig?

Wenn eine Klasse viel häufiger vorkommt, kann das Modell faul werden.

Es kann dann einfach oft die häufige Klasse vorhersagen und trotzdem scheinbar „gut" aussehen.

Beispiel:
- Wenn fast alles `nicht_relevant` ist, kann das Modell einfach fast immer `nicht_relevant` sagen.
- Dann wirkt die Genauigkeit vielleicht okay, aber die seltene Klasse wird schlecht erkannt.

## Warum ist `stratify=y` wichtig?

Beim Aufteilen in Trainings- und Testdaten benutzt die App:

```python
stratify=y
```

Das bedeutet:
- Train und Test sollen ein ähnliches Klassenverhältnis behalten
- also ungefähr genauso viele `relevant`- und `nicht_relevant`-Fälle enthalten

Ohne das könnte es passieren, dass im Test fast nur eine Klasse vorkommt.

## Was macht `class_weight='balanced'`?

Das sagt dem Modell:

> „Die seltene Klasse ist wichtiger und soll stärker beachtet werden."

Das heißt:
- Fehler bei `relevant` zählen mehr
- das Modell soll die seltene Klasse ernster nehmen
- oft wird dadurch der Recall für die seltene Klasse besser

## Was zeigen die Ausgaben?

### `classification_report`

Der Report zeigt pro Klasse:

- **Precision**: Wie oft war die Vorhersage richtig?
- **Recall**: Wie viele echte Fälle hat das Modell gefunden?
- **F1-Score**: Kombination aus Precision und Recall
- **Support**: Wie viele Beispiele es von der Klasse gibt

### `confusion_matrix`

Die Confusion Matrix zeigt:

- wie viele Fälle richtig erkannt wurden
- wie viele Fälle verwechselt wurden

Das hilft dir zu sehen, wo das Modell Probleme hat.

## Was lernt man aus der App?

- Was Klassen-Ungleichgewicht ist
- Warum ein Modell bei seltenen Klassen schummeln kann
- Warum `stratify=y` wichtig ist
- Warum `class_weight='balanced'` helfen kann
- Wie man Modelle bei ungleichen Klassen besser bewertet

## Einfache Merkhilfe

> Wenn eine Klasse viel seltener ist, reicht „Genauigkeit" allein oft nicht aus.
> Man muss auch schauen, ob die seltene Klasse wirklich erkannt wird.

## Ausführen

Im Ordner mit der Datei:

```powershell
python class_imbalance.py
```

Falls du `pandas` oder `scikit-learn` noch nicht installiert hast:

```powershell
pip install pandas scikit-learn
```

## Kurz gesagt

Diese App zeigt dir:

- ein Modell ohne Sonderbehandlung für Klassen-Ungleichgewicht
- ein Modell mit `class_weight='balanced'`
- und den Unterschied zwischen beiden

Damit verstehst du besser, warum bei ungleichen Daten nicht nur Accuracy wichtig ist.

