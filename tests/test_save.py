import requests
import os

def test_save():
    # Test image path
    test_image = "frontend/app/static/demo_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"Test image not found at {test_image}")
        return
        
    # Send to backend
    with open(test_image, "rb") as f:
        files = {"file": ("test.jpg", f.read(), "image/jpeg")}
        response = requests.post("http://localhost:8000/save-for-retrain", files=files)
        
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.json() if response.ok else response.text}")
    
if __name__ == "__main__":
    test_save()
