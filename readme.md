# AI-Powered Microgrid Fault Detection & Pipeline

A machine learning and real-time processing pipeline designed for microgrid fault detection, classification, and anomaly monitoring. This repository integrates unsupervised anomaly detection (**Isolation Forest**) with supervised multi-class fault classification (**XGBoost**) to monitor power grid telemetry.

---

## 📌 Project Overview

Modern smart grids and microgrids require continuous monitoring to identify operational anomalies and classify electrical faults (e.g., short circuits, line-to-ground faults, voltage sags) in real time. 

This repository provides:
- **Data Simulation & Stream Processing**: Simulated microgrid telemetry nodes (`mock_grid_node.py`).
- **Anomaly Detection**: Unsupervised `IsolationForest` model to flag abnormal voltage, current, and frequency spikes.
- **Fault Classification**: Supervised `XGBoost` model to categorize specific microgrid fault types.
- **Pipeline Orchestration**: Real-time processing script (`pipeline_processor.py`) that integrates feature preprocessing, model inference, and log outputs.

---

## 📁 Repository Structure

```text
Grad_PRJ_AI/
├── grid_pipeline/
│   ├── mock_grid_node.py            # Simulated grid sensor telemetry generator
│   ├── pipeline_processor.py        # Real-time data pipeline & inference engine
│   ├── train_IsolationForest.py     # Training script for anomaly detection
│   ├── train_XGBoost.py             # Training script for fault classification
│   ├── iso_forest_model.pkl         # Serialized Isolation Forest model
│   ├── xgboost_fault_classifier.json# Pre-trained XGBoost classifier model
│   ├── microgrid_training_data.csv  # Raw microgrid sensor telemetry dataset
│   ├── training_data_final.csv      # Preprocessed feature dataset for training
│   └── terminal_output.log          # Pipeline execution logs
├── .gitignore                       # Ignored files (venv, cached models, logs)
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- WSL / Linux or macOS / Windows terminal

### 1. Clone the Repository
```bash
git clone git@github.com:moameneslam/Grad_PRJ_AI.git
cd Grad_PRJ_AI
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Models (Optional)
If you wish to retrain the anomaly detector or classifier using updated dataset files:

* Train the Isolation Forest model:
  ```bash
  python grid_pipeline/train_IsolationForest.py
  ```

* Train the XGBoost classifier:
  ```bash
  python grid_pipeline/train_XGBoost.py
  ```

### 2. Run the Real-Time Pipeline Engine
Execute the main pipeline processor to ingest telemetry streams and perform live anomaly detection and classification:

```bash
python grid_pipeline/pipeline_processor.py
```

---

## 🛠️ Requirements & Tech Stack

- **Python 3.x**
- **Scikit-Learn**: Isolation Forest model implementation
- **XGBoost**: Gradient boosted decision trees for multi-class classification
- **Pandas & NumPy**: Data cleaning, feature engineering, and matrix operations
- **Joblib**: Model serialization and loading

---

## 📄 License
This project is part of a Graduation Project research effort. Free to use for academic and non-commercial purposes.