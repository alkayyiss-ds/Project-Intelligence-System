# 🧠 Project Intelligence System

Sistem kecerdasan buatan berbasis Deep Learning dengan deployment web menggunakan FastAPI.

## 📁 Struktur Proyek

`
Project-Intelligence-System/
├── data/               # Dataset (raw, processed, interim, external)
├── notebooks/          # Jupyter Notebooks (EDA, preprocessing, modeling, evaluasi)
├── src/                # Source code utama (data, features, models, utils)
├── models/             # Model artifacts (checkpoints, saved models, exports)
├── api/                # Backend FastAPI (routers, schemas, services)
├── frontend/           # Web frontend (templates, static files)
├── tests/              # Unit & integration tests
├── deploy/             # Konfigurasi deployment (Docker, Nginx)
├── config/             # File konfigurasi
├── docs/               # Dokumentasi proyek
├── logs/               # Log files
└── reports/            # Laporan & visualisasi hasil
`

## 🚀 Quick Start

### 1. Clone Repository
`ash
git clone https://github.com/alkayyiss-ds/Project-Intelligence-System.git
cd Project-Intelligence-System
`

### 2. Setup Environment
`ash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
`

### 3. Jalankan API
`ash
uvicorn api.main:app --reload
`

### 4. Buka Web
Akses http://localhost:8000 di browser.

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch / TensorFlow
- **Backend**: FastAPI / Flask
- **Frontend**: HTML, CSS, JavaScript
- **Containerization**: Docker
- **Web Server**: Nginx

## 📊 Workflow

1. **Data Collection** → data/raw/
2. **EDA** → 
otebooks/01_exploratory/
3. **Preprocessing** → 
otebooks/02_preprocessing/ + src/data/
4. **Modeling** → 
otebooks/03_modeling/ + src/models/
5. **Evaluation** → 
otebooks/04_evaluation/
6. **Deployment** → pi/ + rontend/ + deploy/

## 👤 Author

**alkayyiss-ds**  
GitHub: [alkayyiss-ds](https://github.com/alkayyiss-ds)
