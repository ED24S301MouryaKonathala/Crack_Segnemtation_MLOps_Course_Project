import requests

def test_ping():
    response = requests.get("http://localhost:8000/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    with open("tests/sample.jpg", "rb") as file:
        files = {"file": ("sample.jpg", file, "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files)
        assert response.status_code == 200
        assert "anomaly_score" in response.json()
        assert "prediction" in response.json()

def test_metrics():
    response = requests.get("http://localhost:8000/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


#Note: You'll need a small dummy image called sample.jpg inside tests/ folder for the predict API test.