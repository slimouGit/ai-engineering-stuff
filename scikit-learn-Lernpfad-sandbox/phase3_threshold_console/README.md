# Phase 3 – Confidence und Thresholds

Diese Version ist eine **reine Konsolenanwendung**.

Kein Streamlit, kein Frontend.

## Installation

```bash
pip install pandas scikit-learn
```

## Start in PyCharm

1. Projektordner öffnen.
2. `phase3_threshold_console.py` öffnen.
3. Rechtsklick → **Run 'phase3_threshold_console'**.

Oder im Terminal:

```bash
python phase3_threshold_console.py
```

## Was passiert?

```text
Text
↓
TF-IDF
↓
Logistic Regression
↓
predict_proba()
↓
Confidence
↓
Threshold
↓
relevant / nicht_relevant
```

## Beispiel

Modell:

```text
Confidence relevant = 0.72
```

Threshold:

```text
0.50
```

Dann:

```text
0.72 >= 0.50
→ relevant
```

Wird der Threshold auf `0.80` erhöht:

```text
0.72 < 0.80
→ nicht_relevant
```

Das Modell wurde nicht verändert.

Nur die Entscheidungsregel wurde verändert.

## Lernziele

- `predict_proba()`
- Confidence
- Threshold
- Precision
- Recall
- F1
- False Positive
- False Negative
