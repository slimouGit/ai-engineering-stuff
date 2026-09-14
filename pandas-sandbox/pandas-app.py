import pandas as pd

df = pd.read_csv("support_cases.csv")
print("--- df.columns ---")
print(df.columns)
print("--- df.head(3) ---")
print(df.head(3))
print("--- df.info() ---")
print(df.info())
print("--- df.describe() ---")
print(df.describe())

print(df[["channel","status"]])

print("--- Alle Spalten ---")
for column in df.columns:
    print(column)

print("--- Fehlende Werte ---")
print(df.isna().sum())

print("--- Dublikate finden ---")
print(df.duplicated().sum())
print(df.duplicated())
df = df.drop_duplicates()
print(df.duplicated().sum())

print("--- Welche Ticket-Kategorie kommt am häufigsten vor? ---")
print(df["priority"].value_counts())

print("--- Nur technische Tickets ---")
print(df[df["category"] == "Technik"])

print("--- Nur kritische Tickets ---")
print(df[df["priority"] == "critical"])

print("--- Mehrere Bedingungen ---")
gruppiert = df[
    (df["category"] == "Technik") &
    (df["priority"] == "high")
]
print(gruppiert)

print("--- Zeigt Spaltennamen, Datentypen, Anzahl gefüllter Werte und Speicherverbrauch. ---")
print(df.info())

print("--- Zeigt statistische Kennzahlen für numerische Spalten. ---")
print(df.describe())

print("--- Gibt Anzahl der Zeilen und Spalten zurück. ---")
print(df.shape)
