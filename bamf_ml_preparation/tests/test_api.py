from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    response = client.post(
        "/predict",
        json={"text": "Ich möchte meinen Termin auf nächste Woche verschieben."},
    )
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
