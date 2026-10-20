import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. BEISPIELDATEN ERZEUGEN
# ============================================================

# Wir erzeugen 100 Personen:
# 50 aus Gruppe A
# 50 aus Gruppe B

np.random.seed(42)

gruppe_a = np.random.normal(
    loc=0.65,
    scale=0.10,
    size=50
)

gruppe_b = np.random.normal(
    loc=0.65,
    scale=0.10,
    size=50
)

# Werte auf Bereich 0 bis 1 begrenzen
gruppe_a = np.clip(gruppe_a, 0, 1)
gruppe_b = np.clip(gruppe_b, 0, 1)


# ============================================================
# 2. DATAFRAME BAUEN
# ============================================================

df_a = pd.DataFrame({
    "Gruppe": "A",
    "Qualifikation": gruppe_a
})

df_b = pd.DataFrame({
    "Gruppe": "B",
    "Qualifikation": gruppe_b
})

df = pd.concat(
    [df_a, df_b],
    ignore_index=True
)


# ============================================================
# 3. BIAS EINBAUEN
# ============================================================

# Gruppe A:
# Modell-Score entspricht der Qualifikation

df.loc[
    df["Gruppe"] == "A",
    "Modell_Score"
] = df["Qualifikation"]


# Gruppe B:
# bekommt systematisch 0.15 Punkte abgezogen

df.loc[
    df["Gruppe"] == "B",
    "Modell_Score"
] = (
    df["Qualifikation"] - 0.15
)


# Werte wieder auf 0 bis 1 begrenzen

df["Modell_Score"] = df["Modell_Score"].clip(
    0,
    1
)


# ============================================================
# 4. THRESHOLD
# ============================================================

threshold = 0.60


# Entscheidung:
# Score >= Threshold -> positiv
# Score < Threshold  -> negativ

df["Entscheidung"] = (
    df["Modell_Score"] >= threshold
).astype(int)


# ============================================================
# 5. AUSWERTUNG
# ============================================================

gruppe_a_positiv = df[
    df["Gruppe"] == "A"
]["Entscheidung"].mean()

gruppe_b_positiv = df[
    df["Gruppe"] == "B"
]["Entscheidung"].mean()


print()
print("====================================")
print("BIAS-DEMO")
print("====================================")

print(
    f"Positive Entscheidungen Gruppe A: "
    f"{gruppe_a_positiv * 100:.1f} %"
)

print(
    f"Positive Entscheidungen Gruppe B: "
    f"{gruppe_b_positiv * 100:.1f} %"
)

print(
    f"Differenz: "
    f"{(gruppe_a_positiv - gruppe_b_positiv) * 100:.1f} Prozentpunkte"
)


# ============================================================
# 6. EIN PAAR BEISPIELDATEN AUSGEBEN
# ============================================================

print()
print("Beispieldaten:")
print()

print("\nGruppe A:")
print(
    df[df["Gruppe"] == "A"]
    .head(5)
    .to_string(index=False)
)

print("\nGruppe B:")
print(
    df[df["Gruppe"] == "B"]
    .head(5)
    .to_string(index=False)
)


# ============================================================
# 7. GRAFIK
# ============================================================

plt.figure(
    figsize=(9, 6)
)


# Gruppe A

a = df[
    df["Gruppe"] == "A"
]

plt.scatter(
    a["Qualifikation"],
    a["Modell_Score"],
    label="Gruppe A"
)


# Gruppe B

b = df[
    df["Gruppe"] == "B"
]

plt.scatter(
    b["Qualifikation"],
    b["Modell_Score"],
    label="Gruppe B"
)


# Threshold-Linie

plt.axhline(
    y=threshold,
    linestyle="--",
    label=f"Threshold {threshold}"
)


plt.xlabel(
    "Tatsächliche Qualifikation"
)

plt.ylabel(
    "Modell-Score"
)

plt.title(
    "Beispiel für systematischen Bias"
)

plt.xlim(
    0,
    1
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.grid(
    True
)

plt.show()