from fastapi.testclient import TestClient

from main import app
from ml import EXAMPLE_CENSUS_DATA_POS, EXAMPLE_CENSUS_DATA_NEG

client = TestClient(app)


def test_predict_neg():
    response = client.post("/")
    assert response.status_code == 200
    assert response.json() == {
        {"message": "welcome"}}


def test_predict_pos():
    response = client.post(
        "/predict/",
        json=EXAMPLE_CENSUS_DATA_POS,
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": "[1]"}


def test_predict_neg():
    response = client.post(
        "/predict/",
        json=EXAMPLE_CENSUS_DATA_NEG,
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": "[0]"}
