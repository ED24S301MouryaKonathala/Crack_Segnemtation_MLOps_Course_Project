# EfficientAD Visual Anomaly Detection — End-to-End MLOps Project

## 1. Project Overview
An end-to-end MLOps system for visual anomaly detection using the EfficientAD model, built with:
- DVC for data versioning
- MLflow for experiment tracking
- FastAPI for backend serving
- Simple Web UI frontend
- Prometheus and Grafana for monitoring
- Full orchestration using Docker Compose

Runs fully **locally** — no external cloud services.

## 2. Architecture Diagram
```
User → Frontend (Docker) → FastAPI Backend (Docker) → /predict (EfficientAD Detection)
                                         ↓
                      /metrics → Prometheus (Docker) → Grafana (Docker)
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

| Service | URL |
|:---|:---|
| Frontend (Upload Images) | http://localhost:8500 |
| Backend (FastAPI APIs) | http://localhost:8000 |
| Prometheus (Metrics Scraping) | http://localhost:9090 |
| Grafana (Dashboards) | http://localhost:3001 |
| MLflow UI (Experiment Tracking) | http://localhost:5000 |

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

## 6. Folder Structure
```plaintext
efficientad-mlops/
├── backend/
│   ├── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   └── main.py
│   ├── Dockerfile
│   └── requirements.txt
├── src/
│   ├── train.py
│   ├── data_loader.py
│   └── utils.py
├── monitoring/
│   └── prometheus.yml
├── tests/
│   ├── test_api.py
│   ├── test_train.py
│   ├── test_metrics.py
│   └── sample.jpg
├── data/ (DVC-tracked)
├── models/ (DVC-tracked)
├── docker-compose.yml
├── dvc.yaml
├── params.yaml
├── README.md
├── requirements.txt
└── .gitignore
```

## 7. Monitoring

- Prometheus scrapes backend metrics from `/metrics` every 5 seconds.
- Grafana visualizes:
  - API response times
  - API request counts
  - Backend container CPU and memory usage
  - Host system CPU and memory usage (via Windows Exporter)

## 8. Testing

Run unit tests with:
```bash
pytest tests/
```
Tests cover:
- FastAPI health check `/ping`
- Prediction endpoint `/predict`
- Prometheus metrics exposure `/metrics`
- Data loader and model training validation
- Prometheus scraping validation

## 9. Documentation Deliverables

- Architecture Diagram
- High-Level Design (HLD)
- Low-Level Design (LLD)
- Test Plan and Reports
- User Manual

## 10. Important Notes

- No cloud services are used.
- All datasets and models are tracked using DVC with a local remote at `.dvc_storage/`.
- All services run locally inside Docker containers orchestrated by Docker Compose.
