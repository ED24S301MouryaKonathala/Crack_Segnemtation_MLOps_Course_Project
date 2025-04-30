from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.utils import load_model, predict_mask
from prometheus_fastapi_instrumentator import Instrumentator
from PIL import Image
import io
import logging
from pathlib import Path
import os
from datetime import datetime

app = FastAPI(
    title="Crack Segmentation API",
    description="API for Crack Segmentation using Attention U-Net",
    version="1.0"
)

# Simple CORS middleware - allow all since we're in Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Set debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Simple path configuration
SAVE_DIR = Path("/app/model_retrain/model_retrain_data/images")

# Global model variable
model = None

def get_model():
    global model
    if model is None:
        try:
            model = load_model()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail="Model loading failed")
    return model

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.get("/ping")
async def ping():
    logger.info("Health check requested.")
    return {"message": "API is alive!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Get model
        model = get_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
            
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')  # Ensure RGB
        
        # Make prediction
        mask = predict_mask(model, image)
        
        return {
            "status": "success",
            "prediction": mask.tolist(),
            "shape": list(mask.shape)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-for-retrain")
async def save_for_retrain(file: UploadFile = File(...)):
    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        file_path = SAVE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return {"status": "success", "path": str(file_path)}
    except Exception as e:
        logger.error(f"Save failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    try:
        logger.info("Metrics endpoint called.")
        return instrumentator.metrics()
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
