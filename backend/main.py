from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import torch
from torchvision import transforms
from PIL import Image
import io
import os

app = FastAPI()

model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "models/efficientad_model.pth"
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location="cpu")
        model.eval()
    else:
        print("⚠️ Warning: Model file not found. Prediction endpoint will not work.")

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            content={"error": "Model not loaded. Please train the model first."},
            status_code=503
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            score = torch.sigmoid(output).item()

        result = "anomaly" if score > 0.5 else "normal"
        return JSONResponse(content={"anomaly_score": score, "prediction": result})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
