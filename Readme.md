
text
# Turbofan Engine RUL Predictor (CMAPSS FD001)  
## LSTM + Random Forest Ensemble with Interactive Web Dashboard

> **Tagline:** From raw engine sensors to clear “Green / Yellow / Red” health decisions — combining Deep Learning and Classic ML in a single, intuitive dashboard.

---

## 🛠 Problem & Motivation

Modern aircraft engines stream thousands of sensor readings across their lifetime. Deciding **when** to send an engine for maintenance is critical:

- Too **early** → wasted maintenance cost & downtime  
- Too **late** → unexpected failures & safety risk  

Our project turns NASA’s **CMAPSS FD001** turbofan dataset into a complete **predictive-maintenance system** that estimates **Remaining Useful Life (RUL)** for each engine and presents the results through an **interactive, visual dashboard** that anyone (including non-ML judges) can understand.

---

## 🎯 What Our System Does

Given a test file with sensor readings for multiple engines, the system:

1. **Predicts RUL for every engine** (how many cycles left before failure).  
2. **Combines two models** — an LSTM and a Random Forest — into a smart **ensemble** that is more accurate and robust than either model alone.  
3. Provides a **web dashboard** where you can:
   - Upload a CMAPSS-style test file
   - Select **Engine #X**
   - See:
     - **Ensemble RUL** (final decision)
     - **LSTM-only RUL** (for transparency)
     - **Health status:** GREEN / YELLOW / RED
     - RUL trend over time
     - Sensor trends
     - A gauge-style “health meter”
   - View **fleet-level analytics**: Top-K critical engines, RUL distributions, and scatter plots.

---

## 🤖 Model Architecture & Approach

### 1. Data & Preprocessing

- Dataset: **NASA CMAPSS FD001** (turbofan engine degradation).  
- Each record: `engine_id`, `cycle`, 3 operational settings, multiple sensor channels.  
- Steps:
  - Remove near-constant sensors (low information).
  - Standardize features using a **StandardScaler** fitted on the training set and stored for reuse at inference.
  - Build **30-cycle sliding windows**: for every engine and cycle, we look at the previous 30 cycles of sensor data to form one training example.

This converts raw logs into structured sequences suitable for time-series deep learning and into flattened vectors suitable for tree-based models.

---

### 2. Model 1 — LSTM (Deep Learning Time-Series)

- Input: sequence of shape `(SEQ_LEN=30, num_features)`.
- Architecture (example):
  - LSTM layers → Dropout → Dense layers → Single RUL output.
- Strengths:
  - Learns **temporal degradation patterns** directly from sensor sequences.
  - Captures both short-term fluctuations and long-term trends in engine health.

The LSTM is trained on the processed windows and saved for later inference.

---

### 3. Model 2 — Random Forest Regressor (Tabular ML)

- Input: same 30-cycle window, but **flattened** into a vector (`30 × num_features`).
- Model: **RandomForestRegressor**
  - Multiple trees with controlled depth and leaf size for robust predictions.
- Strengths:
  - Very strong baseline for **tabular data**.
  - Handles noisy and partially redundant features well.
  - Uses averaging across many trees to **reduce overfitting** and stabilize predictions.

Training is optimized for CPU (e.g., Colab Free Tier) using:
- Limited `max_depth` and optional `max_samples`.
- Chunk-wise training + verbose logs so progress is clearly visible.

---

### 4. Weighted Ensemble — Best of Both Worlds

Different model families see the data differently. To exploit this, we build a **weighted ensemble**:

```text
RUL_ensemble = w * RUL_lstm + (1 - w) * RUL_rf
Where:

RUL_lstm = LSTM prediction

RUL_rf = Random Forest prediction

w = weight learned on a validation set

We perform a small grid search over w and pick the value that minimizes validation RMSE.
This simple, principled ensembling:

Improves RUL accuracy on held-out data.

Reduces sensitivity to the weaknesses of any single model.

Provides more robust predictions across different engines.

📊 Web Dashboard (Gradio + Plotly)
The system includes a Gradio web app with Plotly visualizations.

Engine Details View (Engine #X)
After uploading a test file and selecting an engine ID, the dashboard shows:

Ensemble RUL (final decision – what we recommend using).

LSTM-only RUL (so judges can compare and see the effect of ensembling).

Status badge:

GREEN → RUL high → Airworthy / OK

YELLOW → RUL moderate → Schedule maintenance

RED → RUL low → Critical / Ground soon

RUL trend curve over cycles:

Two lines: Ensemble RUL vs LSTM RUL over time (rolling windows).

Background color bands show risk zones (e.g., red/yellow/green bands).

Sensor trend plot:

Choose any sensor (s1–s21) to see its trajectory vs cycle.

Gauge chart:

Circular gauge summarizing health based on Ensemble RUL.

This makes the model’s decision transparent and visually intuitive.

Fleet Overview
For the whole uploaded test file:

Top-K Critical Engines:

Bar chart showing engines with the lowest Ensemble RUL.

RUL Distribution:

Histogram of Ensemble RUL to understand overall fleet health.

Scatter Plot:

Engine ID vs Ensemble RUL, showing which engines are safe vs risky.

🧱 Repository Structure (Suggested)
text
.
├── ai_hackathon.py                 # Training pipeline (LSTM + preprocessing)
├── gradio_app.py                   # Web dashboard (LSTM + RF + Ensemble)
├── train_FD001.txt                 # Training data (raw CMAPSS format)
├── test_FD001.txt                  # Test data (raw CMAPSS format)
├── RUL_FD001.txt                   # Ground-truth RUL for test engines
├── lstm_rul_fd001.h5               # Saved LSTM model
├── scaler_fd001.pkl                # Saved StandardScaler
├── feature_cols_fd001.pkl          # Saved feature list
├── rf_rul_fd001.pkl                # Saved Random Forest model
└── ensemble_cfg.pkl                # Ensemble config (w_lstm, seq_len)
🚀 How to Run
1. Install Dependencies
bash
pip install -U numpy pandas scikit-learn tensorflow joblib gradio plotly
2. Train Models (if not already done)
Run ai_hackathon.py (or the training notebook) to:

Compute RUL labels for training.

Train the LSTM.

Save lstm_rul_fd001.h5, scaler_fd001.pkl, feature_cols_fd001.pkl.

Run the Random Forest + ensemble training script to:

Train rf_rul_fd001.pkl.

Save ensemble_cfg.pkl (containing w_lstm and seq_len).

3. Launch the Dashboard (Local)
bash
python gradio_app.py
Open the printed URL (e.g., http://127.0.0.1:7860).

4. Launch in Google Colab (for demo / judges)
In Colab, after defining demo:

python
demo.launch(share=True)
This generates a public link that you can share with judges during a presentation.

🔍 Health Status Thresholds
Status is currently determined from Ensemble RUL using:

python
GREEN_TH = 50   # cycles
YELLOW_TH = 20  # cycles
RUL >= GREEN_TH → GREEN

YELLOW_TH <= RUL < GREEN_TH → YELLOW

RUL < YELLOW_TH → RED

These values are configurable to match different safety policies.

✅ Why You Should Care
Full pipeline: from raw NASA CMAPSS data to ready-to-use maintenance decisions.

Model diversity: uses both deep learning (LSTM) and classic ML (Random Forest) and justifies combining them with a data-driven ensemble.

Explainability: shows not only a number but also visual trends and clear color-coded statuses.

Practical UI: the Gradio dashboard turns a complex ML system into something that looks and feels like a real tool a maintenance engineer could use.

Engineering effort: includes training, optimization for CPU environments, model saving/loading, and a robust inference pipeline.

🔮 Future Work
Extend to other CMAPSS subsets (FD002/FD003/FD004).

Add explainability (SHAP / feature importance) for the Random Forest.

Auto-generate PDF health reports per engine.

Incorporate trend slope into status

