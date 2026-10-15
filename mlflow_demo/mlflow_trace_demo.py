import mlflow

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# MLflow konfigurieren
# --------------------------------------------------

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow Trace Demo")


# --------------------------------------------------
# Trainingsfunktion als Trace
# --------------------------------------------------

@mlflow.trace
def train_model(c_value, X_train, X_test, y_train, y_test):

    model = LogisticRegression(
        C=c_value,
        max_iter=500
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    return accuracy


# --------------------------------------------------
# Daten laden
# --------------------------------------------------

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.3,
    random_state=42
)


# --------------------------------------------------
# Zwei Modellvarianten testen
# --------------------------------------------------

for c_value in [0.1, 1.0]:

    with mlflow.start_run(
        run_name=f"C_{c_value}"
    ):

        accuracy = train_model(
            c_value,
            X_train,
            X_test,
            y_train,
            y_test
        )

        # Parameter speichern
        mlflow.log_param(
            "C",
            c_value
        )

        # Metrik speichern
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        print(
            f"C={c_value} -> Accuracy={accuracy:.3f}"
        )