import pandas as pd

df = pd.read_csv("support_cases.csv")
#1. Daten laden und verstehen
print("--------- df.head() ---------")
print(df.head())
print("--------- df.info() ---------")
print(df.info())
print("------------- df[sla_breached] -------------")
print(df["sla_breached"])
print("--------- df[sla_breached].value_counts() ---------")
print(df["sla_breached"].value_counts())

#2. Erstes sehr einfaches Modell (Wir verwenden zunächst nur numerische Werte)
features = [
    "customer_age",
    "attempts",
    "resolution_minutes",
    "sla_minutes",
    "text_length",
    "cost_eur"
]
X = df[features]
y = df["sla_breached"]

#3. Fehlende Werte behandeln (In customer_age fehlen teilweise Werte. Auch satisfaction_score ist bei offenen Tickets teilweise leer.)
print("X.median() ", X.median())
print("X.isna().sum() ", X.isna().sum())
print("X.isna() ", X.isna())
print(X["customer_age"])
X = X.fillna(X.median())

#4. Train und Test aufteilen
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#5. Erstes Modell: Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

#6. Vorhersagen machen
y_pred = model.predict(X_test)
print("y_pred[:10]")
print(y_pred[:10])

#7. Modell evaluieren
from sklearn.metrics import classification_report
print("classification_report(y_test, y_pred)")

print(classification_report(y_test, y_pred))
#Class False (support 95): precision = 0.99 → of predicted False, 99% were correct. recall = 1.00 → all 95 actual False were correctly identified. f1 = 0.99 balances those two.
#Class True (support 6): precision = 1.00 → every sample predicted as True was actually True (no false positives). recall = 0.83 → 83% of the 6 true True samples were found, i.e. 5/6 detected and 1 missed (false negative). f1 = 0.91.
#accuracy = 0.99 → (95 + 5) / 101 = 100/101 ≈ 0.99 correct overall.
#macro avg = unweighted mean of per-class metrics (treats both classes equally).
#weighted avg = mean of per-class metrics weighted by support (reflects imbalance; dominated by the larger class).


#print(df[["customer_age", "attempts", "resolution_minutes", "sla_minutes", "text_length", "cost_eur", "sla_breached"]])


# alternative: force full text output
print(df[["customer_age", "attempts", "resolution_minutes", "sla_minutes", "text_length", "cost_eur", "sla_breached"]].to_string(index=False))

#8. Vorhersage treffen
print("--------- Vorhersage treffen ---------")
new_ticket = pd.DataFrame([{
    "customer_age": 54,
    "attempts": 3,
    "resolution_minutes": 300,
    "sla_minutes": 240,
    "text_length": 500,
    "cost_eur": 120
}])

prediction = model.predict(new_ticket)

print(prediction)
