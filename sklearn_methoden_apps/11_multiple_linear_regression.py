"""
MULTIPLE LINEARE REGRESSION

Mehrere Eingabewerte werden genutzt,
um einen Zahlenwert vorherzusagen.

Beispiel:
Bearbeitungsdauer eines Tickets in Minuten.

Features:
- komplexitaet
- versuche
- textlaenge

Ziel:
- dauer_min
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ============================================================
# 1. DATEN
# ============================================================

df = pd.DataFrame({
    "komplexitaet": [
        1, 1, 2, 2, 3, 3, 4, 4,
        5, 5, 2, 3, 4, 1, 5, 3
    ],

    "versuche": [
        1, 2, 1, 2, 2, 3, 3, 4,
        4, 5, 3, 4, 2, 1, 5, 2
    ],

    "textlaenge": [
        120, 180, 220, 300, 350, 420, 500, 600,
        700, 850, 310, 460, 520, 100, 900, 380
    ],

    "dauer_min": [
        20, 30, 40, 55, 75, 95, 120, 145,
        170, 210, 60, 100, 115, 18, 220, 80
    ]
})


# ============================================================
# 2. X UND y
# ============================================================

# X = mehrere Features
X = df[
    [
        "komplexitaet",
        "versuche",
        "textlaenge"
    ]
]

# y = Zielwert
y = df["dauer_min"]


# ============================================================
# 3. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ============================================================
# 4. MODELL
# ============================================================

model = LinearRegression()


# ============================================================
# 5. TRAINIEREN
# ============================================================

# INPUT:
# X_train = komplexitaet, versuche, textlaenge
# y_train = bekannte Bearbeitungsdauer

model.fit(
    X_train,
    y_train
)


# ============================================================
# 6. VORHERSAGE
# ============================================================

# INPUT:
# X_test
#
# OUTPUT:
# vorhergesagte Dauer in Minuten

y_pred = model.predict(
    X_test
)


# ============================================================
# 7. MODELL BEWERTEN
# ============================================================

# MAE:
# Durchschnittliche Abweichung in Minuten.
#
# Beispiel:
# MAE = 8
# -> Vorhersagen liegen im Schnitt etwa 8 Minuten daneben.

mae = mean_absolute_error(
    y_test,
    y_pred
)

print(
    "MAE:",
    round(mae, 2)
)


# R²:
# Wie gut erklären die Features die Unterschiede in der Dauer?
#
# grob:
# 1.0 = sehr gut
# 0.0 = kaum besser als Mittelwert

r2 = r2_score(
    y_test,
    y_pred
)

print(
    "R²:",
    round(r2, 2)
)


# ============================================================
# 8. IST / VORHERSAGE VERGLEICHEN
# ============================================================

for actual, pred in zip(
    y_test,
    y_pred
):
    print(
        "Ist:",
        actual,
        "| Vorhersage:",
        round(pred, 1)
    )


# ============================================================
# 9. NEUEN FALL VORHERSAGEN
# ============================================================

neuer_fall = pd.DataFrame({
    "komplexitaet": [3],
    "versuche": [2],
    "textlaenge": [400]
})

vorhersage = model.predict(
    neuer_fall
)

print(
    "\nVorhergesagte Bearbeitungsdauer:",
    round(vorhersage[0], 1),
    "Minuten"
)


# ============================================================
# 10. EINFLUSS DER FEATURES ANSEHEN
# ============================================================

# coef_ zeigt, wie stark jedes Feature
# die Vorhersage beeinflusst.

for feature, coefficient in zip(
    X.columns,
    model.coef_
):

    print(
        feature,
        "→ Koeffizient:",
        round(coefficient, 3)
    )