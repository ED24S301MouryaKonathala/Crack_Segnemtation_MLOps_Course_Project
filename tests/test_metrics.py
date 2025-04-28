import requests

def test_prometheus_scraping():
    response = requests.get("http://localhost:9090/api/v1/targets")
    assert response.status_code == 200
    assert "status" in response.json()
