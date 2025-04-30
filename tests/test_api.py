import requests
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_predict():
    """Test prediction endpoint"""
    test_image = "frontend/app/static/demo_image.jpg"
    
    if not os.path.exists(test_image):
        logger.error(f"Test image not found at {test_image}")
        return
        
    with open(test_image, "rb") as f:
        files = {"file": ("test.jpg", f.read(), "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files)
        
    print(f"Predict Response status: {response.status_code}")
    print(f"Predict Response body: {response.json() if response.ok else response.text}")

def test_health():
    """Test health check endpoint"""
    response = requests.get("http://localhost:8000/ping")
    print(f"Health Response status: {response.status_code}")
    print(f"Health Response body: {response.json() if response.ok else response.text}")

if __name__ == "__main__":
    test_health()
    test_predict()
