"""
PCA
Dimensionsreduktion: mehrere Merkmale werden auf weniger Komponenten reduziert.
PCA ist kein Klassifikationsmodell.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.DataFrame({
    "feature_1":[10,12,14,20,22,24,30,32],
    "feature_2":[20,22,25,35,38,40,50,52],
    "feature_3":[5,6,7,10,11,12,15,16],
    "feature_4":[100,110,120,160,170,180,220,230]
})

# Vor PCA skalieren wir die Features.
X_scaled = StandardScaler().fit_transform(df)

# 4 Features -> 2 neue Hauptkomponenten.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

result = pd.DataFrame(X_pca, columns=["PC1","PC2"])

print("--- Reduzierte Daten ---")
print(result)

print("\nErklärte Varianz je Komponente:")
print(pca.explained_variance_ratio_)

print("Zusammen:", round(pca.explained_variance_ratio_.sum(), 3))
