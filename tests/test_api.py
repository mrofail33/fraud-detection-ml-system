from fraud_detection.api import create_app
from fastapi.testclient import TestClient


def test_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_requires_trained_model(tmp_path):
    client = TestClient(create_app(tmp_path / "missing.joblib"))

    response = client.post("/predict", json={"amount": 100, "time": 10, "features": {}})

    assert response.status_code == 503
    assert "Model artifact not found" in response.json()["detail"]
