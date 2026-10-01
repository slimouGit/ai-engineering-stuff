"""
RANDOM FOREST
Viele Decision Trees stimmen gemeinsam ab.
Meist robuster als ein einzelner Baum.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.DataFrame({
    "monate_kunde":[2,5,12,24,36,3,8,18,40,6,30,1,15,22],
    "support_tickets":[8,6,4,2,1,9,7,3,1,8,2,10,4,3],
    "zufriedenheit":[2,2,3,4,5,1,2,4,5,2,4,1,3,4],
    "churn":["ja","ja","nein","nein","nein","ja","ja",
             "nein","nein","ja","nein","ja","nein","nein"]
})

X = df[["monate_kunde","support_tickets","zufriedenheit"]]
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# n_estimators = Anzahl der Bäume.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))
