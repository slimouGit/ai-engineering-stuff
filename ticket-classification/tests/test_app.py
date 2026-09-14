from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_standardized_success_payload():
    response = client.post(
        "/predict",
        json={"text": "Ich habe mein Passwort vergessen"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["category"] in ["Technik", "Zugang", "Abrechnung", "Allgemein"]
    assert "message" in payload


def test_predict_rejects_empty_text():
    response = client.post(
        "/predict",
        json={"text": "   "},
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["success"] is False
    assert payload["category"] is None
    assert "message" in payload
