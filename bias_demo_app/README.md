
# Bias Demo App

Diese kleine Streamlit-App veranschaulicht Bias mit einer einfachen Datengrundlage.

## Szenarien

1. Kein Bias
2. Modell-Bias
3. Sampling-Bias
4. Label-Bias

## Installation

```powershell
py -m pip install -r requirements.txt
```

## Start

```powershell
py -m streamlit run app.py
```

Die App zeigt:
- zwei Gruppen A und B
- Qualifikationswerte
- Modell-Scores
- einen veränderbaren Threshold
- positive Entscheidungen pro Gruppe
- sichtbare Unterschiede zwischen den Gruppen

Ziel ist nicht ein realistisches Produktivmodell, sondern ein leicht verständliches Lernbeispiel.
