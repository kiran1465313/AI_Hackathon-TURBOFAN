# Turbofan Engine RUL Predictor (CMAPSS FD001) — LSTM + Random Forest Ensemble

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-green)](https://www.gradio.app/)

Predict **Remaining Useful Life (RUL)** of aircraft turbofan engines using NASA CMAPSS FD001 sensor time-series data.  
This project trains:
- an **LSTM** time-series model (deep learning)
- a **Random Forest** regressor (classical ML)
and combines both using a **weighted ensemble** to improve accuracy and stability.

It also provides a **Gradio web dashboard** where users upload a `test_FD001.txt`-style file, select **Engine #X**, and see:
- Ensemble RUL + LSTM-only RUL
- GREEN / YELLOW / RED health status
- RUL trend curve, sensor trend, gauge, and fleet-level charts.

---

## Features
- Upload CMAPSS-style test file and get engine-wise predictions
- Engine details panel (Engine #X):
  - **Ensemble RUL**
  - **LSTM-only RUL**
  - Status badge (Green/Yellow/Red)
  - Rolling **RUL trend** (Ensemble vs LSTM)
  - Sensor trend plot + gauge
- Fleet overview:
  - Top-K critical engines
  - RUL distribution histogram
  - Engine-wise scatter plot

---

## Tech Stack
- Python, NumPy, Pandas
- TensorFlow / Keras (LSTM)
- Scikit-learn (RandomForestRegressor, StandardScaler)
- Plotly (interactive charts + gauge)
- Gradio (web UI)

---

## Dataset (CMAPSS / FD001)
Each dataset file has **26 columns**:
1. unit number (`engine_id`)
2. time in cycles (`cycle`)
3. operational setting 1..3
4. sensor measurements (multiple sensors)  
FD001 contains 100 train trajectories and 100 test trajectories. [web:148]

Expected input file: space-separated `test_FD001.txt` format.

---

## Repository Structure (recommended)
```text
.
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── ai_hackathon.py
├── gradio_app.py
├── lstm_rul_fd001.h5
├── scaler_fd001.pkl
├── feature_cols_fd001.pkl
├── rf_rul_fd001.pkl
└── ensemble_cfg.pkl
