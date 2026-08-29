# Project Intelligence System

A modular Deep Learning platform with FastAPI backend and web frontend — designed for end-to-end AI model development, experimentation, and deployment.

---

## Overview

This project provides a complete ML engineering framework covering:
- **Data pipeline** — raw ingestion to processed features
- **Modeling** — Deep Learning with PyTorch / TensorFlow
- **Experimentation** — Jupyter notebooks structured by phase
- **Deployment** — FastAPI backend + Docker containerization

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | PyTorch ≥ 2.0, TensorFlow ≥ 2.12 |
| NLP | HuggingFace Transformers |
| Computer Vision | OpenCV, Pillow |
| Backend API | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy, scikit-learn |
| Containerization | Docker + Docker Compose |
| Web Server | Nginx |
| Testing | Pytest + pytest-asyncio |

---

## Project Structure

```
Project-Intelligence-System/
├── api/                        # FastAPI Backend
│   ├── main.py                 # App entry point, CORS, routing
│   ├── routers/                # API route handlers
│   ├── schemas/                # Pydantic request/response models
│   ├── services/               # Business logic & ML inference
│   ├── middleware/             # Custom middleware
│   └── core/                  # Config, security, dependencies
│
├── src/                        # Core Python package
│   ├── data/                   # Data loading & preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model definitions (PyTorch/TF)
│   ├── utils/                  # Shared utilities
│   └── visualization/          # Plotting & reporting tools
│
├── notebooks/                  # Jupyter Notebooks (phased)
│   ├── 01_exploratory/         # EDA & data understanding
│   ├── 02_preprocessing/       # Data cleaning & feature engineering
│   ├── 03_modeling/            # Model training & experiments
│   └── 04_evaluation/          # Model evaluation & comparison
│
├── models/                     # Model artifacts
│   ├── saved/                  # Trained model weights
│   ├── checkpoints/            # Training checkpoints
│   └── exports/                # ONNX / TorchScript exports
│
├── data/                       # Data directories (not committed)
│   ├── raw/                    # Original, immutable data
│   ├── interim/                # Intermediate transformations
│   ├── processed/              # Final processed datasets
│   └── external/               # Third-party data
│
├── deploy/
│   └── docker/
│       ├── Dockerfile
│       └── docker-compose.yml
│
├── frontend/                   # Web UI
│   ├── templates/              # Jinja2 HTML templates
│   └── static/                 # CSS, JS, images
│
├── tests/                      # Test suites
│   ├── unit/
│   ├── integration/
│   └── api/
│
├── scripts/
│   └── download_dataset.py     # Dataset download helper
├── config/                     # YAML configs
├── docs/                       # Documentation
├── reports/                    # Generated metrics & figures
├── .env.example                # Environment variables template
└── requirements.txt
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/alkayyiss-ds/Project-Intelligence-System.git
cd Project-Intelligence-System

python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and config
```

### 3. Download Dataset

```bash
python scripts/download_dataset.py
```

### 4. Run API Server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | API status check |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI documentation |
| `POST` | `/predict` | Run model inference *(planned)* |
| `GET` | `/models` | List available models *(planned)* |

---

## Development Workflow

```
1. Data Collection    →  data/raw/
2. EDA                →  notebooks/01_exploratory/
3. Preprocessing      →  notebooks/02_preprocessing/ + src/data/
4. Feature Eng        →  src/features/
5. Modeling           →  notebooks/03_modeling/ + src/models/
6. Evaluation         →  notebooks/04_evaluation/
7. API Deployment     →  api/ + deploy/docker/
```

---

## Docker Deployment

```bash
cd deploy/docker
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Author

**alkayyiss-ds** — [github.com/alkayyiss-ds](https://github.com/alkayyiss-ds)
