from fastapi.testclient import TestClient
from app_api import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200


def test_prediction():

    response = client.post(
        "/predict",
        json={
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    )

    assert response.status_code == 200

    assert "prediction" in response.json()


def test_invalid_features():

    response = client.post(
        "/predict",
        json={
            "features": [1, 2]
        }
    )

    assert response.status_code == 400