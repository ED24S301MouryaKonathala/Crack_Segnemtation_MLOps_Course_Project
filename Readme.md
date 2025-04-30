# Crack Detection System - MLOps Project

## 1. Project Overview
An end-to-end MLOps system for crack detection using a PyTorch Attention U-Net model, built with:
- DVC for data and model versioning
- MLflow for experiment tracking
- FastAPI for backend serving
- Streamlit for interactive frontend with user feedback and retraining capability
- Feedback-driven retraining pipeline: users can submit unsatisfactory images for model improvement
- Prometheus and Grafana for monitoring
- Full Docker Compose orchestration

Runs fully **locally** — no external cloud services required.

## 2. Architecture Diagram
```
User → Streamlit UI (8500) → FastAPI Backend (8000) → Crack Detection & Feedback
                                    ↓
                      /metrics → Prometheus (9090) → Grafana (3000)
                                    ↓
                Feedback images → Retraining Pipeline (DVC, MLflow, PyTorch)
```

## 3. Setup Instructions

### Step 1: Clone Repository
```bash
git clone <your-repository-url>
cd efficientad-mlops
```

### Step 2: Install DVC
```bash
pip install dvc
```

### Step 3: Initialize and Configure DVC
```bash
dvc init
dvc remote add -d local_storage .dvc_storage
```

### Step 4: Run Docker Compose
```bash
docker-compose up --build
```

### Step 5: Access Services

| Service | URL | Description |
|:---|:---|:---|
| Frontend UI | http://localhost:8500 | Streamlit interface for image upload and feedback |
| Backend API | http://localhost:8000 | FastAPI service |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Monitoring dashboards |

## 4. DVC Commands for Versioning

### Add Data
```bash
dvc add data/
dvc push
```

### Reproduce Pipeline
```bash
dvc repro
```

### Visualize Pipeline
```bash
dvc dag
```

## 5. API Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/ping` | GET | Health check |
| `/predict` | POST | Upload image and receive anomaly detection result |
| `/metrics` | GET | Prometheus-compatible metrics endpoint |
| `/save-for-retrain` | POST | Save user feedback images for retraining |

## 6. Folder Structure
```plaintext
crack-detection/
├── backend/                # FastAPI backend service
│   ├── app/
│   │   ├── main.py         # API endpoints
│   │   └── model.py        # Model inference
│   ├── requirements.txt    # Backend dependencies
│   └── Dockerfile
├── frontend/               # Streamlit frontend
│   ├── app/
│   │   ├── streamlit_app.py # UI interface (with feedback & retraining)
│   │   └── static/          # Static assets
│   ├── requirements.txt    # Frontend dependencies
│   └── Dockerfile
├── model_retrain/          # Retraining pipeline scripts and data
│   ├── model_retrain.py    # Retraining logic
│   ├── run_retrain_pipeline.py # Pipeline entrypoint
│   ├── retrain_params.yaml # Retraining hyperparameters
│   └── model_retrain_data/ # User feedback images and masks
├── src/                   # Core ML code
│   ├── train.py            # Training script
│   ├── data_loader.py      # Data loading utilities
│   └── utils.py            # Helper functions
├── tests/                 # Test files
│   ├── test_api.py         # Test FastAPI endpoints
│   ├── test_metrics.py     # Test metrics calculation
│   ├── test_train.py       # Test training pipeline
│   ├── test_save.py        # Test image saving for retraining
│   └── sample.jpg          # Sample image for tests
├── monitoring/            # Monitoring configs
│   ├── prometheus.yml      # Prometheus scrape configuration
│   └── grafana/
│       └── dashboards/     # Grafana dashboard JSONs
│       
├── models/                # Saved models (DVC tracked)
├── docker-compose.yml     # Service orchestration
└── README.md
```

## 7. Model Training & Retraining

The segmentation model is a PyTorch implementation of Attention U-Net, matching the architecture in the provided notebook. The model is trained using `src/train.py` and saved to `models/attention_unet.pth`.

- 10% of the training set is used for validation.
- The test set is used for final evaluation.
- All metrics (train/val/test loss and accuracy) are logged to MLflow.
- The trained model is saved locally, logged to MLflow, and tracked by DVC.

### Feedback-Driven Retraining

- Users can submit images for retraining via the Streamlit UI if the prediction is unsatisfactory.
- Submitted images are stored in `model_retrain/model_retrain_data/images/` (and masks in `masks/`).
- When enough new feedback images are collected, the retraining pipeline (`model_retrain/run_retrain_pipeline.py`) can be triggered.
- Retraining uses DVC to version new data and models, and MLflow to track experiments.

## 8. Monitoring

- Prometheus scrapes backend metrics from `/metrics` every 5 seconds.
- Grafana visualizes:
  - API response times
  - API request counts
  - Backend container CPU and memory usage
  - Host system CPU and memory usage (via Windows Exporter)

## 9. Testing

Run individual test scripts:
```bash
python tests/test_api.py      # Test API endpoints
python tests/test_metrics.py  # Test metrics calculation
python tests/test_train.py    # Test training pipeline
python tests/test_save.py     # Test image saving for retraining
```

Tests cover:
- FastAPI endpoints `/ping`, `/predict`, `/save-for-retrain`
- Metrics calculation (Dice coefficient, IoU)
- Data loader and training pipeline
- Image saving and feedback for retraining workflow

## 10. Model Retraining Pipeline

The system includes a feedback-driven retraining pipeline:

1. Users provide feedback on predictions via the Streamlit UI.
2. If results are unsatisfactory, images are automatically saved to `model_retrain/model_retrain_data/images/`.
3. When enough new images are collected, run:
   ```bash
   python model_retrain/run_retrain_pipeline.py
   ```
   This will:
   - Check if enough new data is available
   - Track new data with DVC
   - Retrain the model using the feedback data
   - Log the new model to MLflow and DVC

Retraining data structure:
```plaintext
model_retrain/
└── model_retrain_data/
    ├── images/             # User feedback images
    ├── masks/              # Corresponding masks (if available)
    └── metadata.json       # Feedback metadata (optional)
```

## 11. Documentation Deliverables

- Architecture Diagram
- High-Level Design (HLD)
- Low-Level Design (LLD)
- Test Plan and Reports
- User Manual

## 12. Important Notes

- No cloud services are used.
- All datasets and models are tracked using DVC with a local remote at `.dvc_storage/`.
- All services run locally inside Docker containers orchestrated by Docker Compose.
