from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import requests

app = FastAPI()

BACKEND_URL = "http://backend:8000"

@app.get("/", response_class=HTMLResponse)
async def upload_form():
    content = """
    <html>
        <body>
            <h2>Upload an Image for Anomaly Detection</h2>
            <form action="/predict" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input type="submit">
            </form>
        </body>
    </html>
    """
    return content

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    response = requests.post(f"{BACKEND_URL}/predict", files=files)
    result = response.json()
    return f"""
    <html>
        <body>
            <h2>Prediction Result</h2>
            <p><strong>Prediction:</strong> {result.get('prediction')}</p>
            <p><strong>Anomaly Score:</strong> {result.get('anomaly_score')}</p>
            <a href="/">Back to Upload</a>
        </body>
    </html>
    """
