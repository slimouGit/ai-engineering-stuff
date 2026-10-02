"""
K-MEANS
Clustering ohne bekannte Zielvariable y.
Das Modell gruppiert ähnliche Daten automatisch.
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.DataFrame({
    "alter":[21,23,25,35,38,40,55,58,62,24,39,60],
    "ausgaben":[20,25,30,70,80,75,120,130,140,28,78,135]
})

X = df[["alter","ausgaben"]]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10))
])

# fit_predict() trainiert und liefert direkt die Cluster-Nummer.
df["cluster"] = model.fit_predict(X)

print(df)
print("\nCluster 0/1/2 sind nur technische Gruppennamen.")


# single prediction after model was fitted
sample = [[20, 23]]  # [alter, ausgaben]
pred = model.predict([[20, 23]])
print("Predicted cluster:", int(pred[0]))

# or using a DataFrame with the same column names
import pandas as pd
sample_df = pd.DataFrame([[20, 23]], columns=["alter", "ausgaben"])
print("Predicted cluster (from DataFrame):", int(model.predict(sample_df)[0]))
