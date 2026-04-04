import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

positive_test_data = [
    ("What a great MLOps lecture", "positive"),
    ("Amazing experience", "positive"),
]


@pytest.mark.parametrize("text, expected", positive_test_data)
def test_predict_positive(text, expected):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == expected


negative_test_data = [
    ("This is terrible", "negative"),
    ("Worst thing ever", "negative"),
]


@pytest.mark.parametrize("text, expected", negative_test_data)
def test_predict_negative(text, expected):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == expected


neutral_test_data = [
    ("It's okay, nothing special", "neutral"),
    ("Not bad, not great", "neutral"),
]


@pytest.mark.parametrize("text, expected", neutral_test_data)
def test_predict_neutral(text, expected):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == expected
