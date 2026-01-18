# Turbofan Engine RUL Predictor (CMAPSS FD001) — LSTM + Random Forest Ensemble

A predictive-maintenance web application that estimates the **Remaining Useful Life (RUL)** of turbofan aircraft engines using time-series sensor data from the NASA CMAPSS FD001 dataset. The goal is to predict how many operating cycles remain before an engine reaches failure conditions, helping teams schedule maintenance at the right time—neither too early (wasting cost/downtime) nor too late (increasing failure risk).

---

## What this project does

This system converts raw engine sensor logs into:
- A **numeric RUL prediction** (remaining cycles)
- A **traffic-light health status**: **GREEN / YELLOW / RED**
- Easy-to-understand **graphs** (RUL trend, sensor trends) and **fleet overview charts**

Users upload a CMAPSS-style test file, select **Engine #X**, and instantly see:
- **Ensemble RUL** (final best prediction)
- **LSTM-only RUL** (for transparency/comparison)
- Health status and recommended action
- Rolling RUL trend over time (steady degradation)
- Sensor trend plots for deeper insight
- Fleet-level ranking of most critical engines

---

## Why this matters (real-world value)

In real maintenance workflows, incorrect timing is costly:
- **Too early** → unnecessary maintenance cost + downtime  
- **Too late** → unexpected failures + safety risk  

This project helps maintenance teams take data-driven decisions by continuously estimating remaining life from sensor behavior across cycles.

---

## How the system works (simple flow)

1. **User uploads** a CMAPSS FD001-style `test_FD001.txt` file containing multiple engines.
2. The pipeline **preprocesses** the data using the same transformations learned during training.
3. Two models predict RUL:
   - **LSTM** (deep learning, time-series)
   - **Random Forest** (robust classical ML)
4. Their predictions are combined into a **weighted ensemble** for better accuracy.
5. The dashboard visualizes:
   - Engine-level details for Engine #X
   - Fleet-level critical engine list and distribution plots

---

## Data processing (what happens inside)

### Feature preparation
- Input file contains: `engine_id`, `cycle`, operational settings, and sensor readings.
- Near-constant columns can be removed (optional) to reduce noise and speed training.

### Scaling / Standardization
- Sensor values vary widely in magnitude, so features are **standardized** using a scaler fit on training data.
- The same saved scaler is reused at inference to prevent mismatch between training and prediction.

### Time-windowing (Sequence building)
- Engine degradation is a time pattern, not a single snapshot.
- We use `SEQ_LEN = 30` cycles:
  - Each prediction uses the **last 30 cycles** of sensor history for that engine.

---

## Models

### Model 1 — LSTM (Deep Learning time-series)
The LSTM learns temporal degradation patterns from 30-cycle sequences.  
It is useful because it can capture:
- steady degradation trends,
- short and long dependencies across time,
- patterns not visible in a single cycle.

### Model 2 — Random Forest Regressor (Classical ML)
The Random Forest is trained on the same 30-cycle window, but the window is **flattened** into one vector so it becomes a fixed-size input suitable for tree models.  
It is useful because it:
- handles noisy sensor data well,
- often performs strongly on tabular problems,
- is stable due to averaging many trees.

---

## Why the ensemble is better

Different model types have different strengths:
- LSTM captures sequence dynamics.
- Random Forest is robust on tabular signals.

We combine them into a **weighted ensemble**:

```text
RUL_ensemble = w * RUL_lstm + (1 - w) * RUL_rf
