import os
from pathlib import Path
from src.utils import load_params

BASE_DIR = Path(__file__).resolve().parent
params = load_params(BASE_DIR / "params.yaml")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "CrackSegmentation-AttentionUNet"

MODEL_PARAMS = params.get("train", {})
API_HOST = "0.0.0.0"
API_PORT = 8000

PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000
