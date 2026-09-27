import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("car.csv")
print(df.head())
features = [
    "buying",
    "maintenance",
    "doors",
    "persons",
    "lug_boot",
    "safety"
]
X = df[features]
y = df["class"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

X_encoded = encoder.fit_transform(X)



