# Turbofan Engine RUL Predictor (CMAPSS FD001) — LSTM + Random Forest Ensemble

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20Dashboard-green)](https://www.gradio.app/)

A predictive-maintenance system that estimates **Remaining Useful Life (RUL)** of turbofan engines from sensor time-series (NASA CMAPSS FD001).  
It trains two models (LSTM + Random Forest) and combines them into a **weighted ensemble** for improved accuracy and robustness, then serves results via an interactive **Gradio + Plotly dashboard**.

---

## Why this project matters
In real maintenance workflows, engines should not be serviced too early (wasting cost) or too late (risking failure).  
This project converts raw sensor logs into:
- a clear **RUL number**
- a **traffic-light health status** (GREEN / YELLOW / RED)
- intuitive **trend charts** that a non-technical user can understand.

---

## What the dashboard shows
### Engine Details (Engine #X)
When a user selects an engine ID:
- **Ensemble RUL** (final best prediction)
- **LSTM-only RUL** (for transparency/comparison)
- Status badge: **GREEN / YELLOW / RED**
- Rolling **RUL Trend** over cycles (Ensemble vs LSTM lines)
- Sensor trend plot (choose any sensor)
- Gauge meter (quick “health” visual)

### Fleet Overview (Whole uploaded file)
- Top-K most critical engines (lowest RUL)
- RUL distribution histogram
- Engine-wise scatter plot

---

## Why we use 2 models (and why ensemble is better)
### LSTM (Deep Learning)
- Best for learning **patterns in sequences** (time-series behavior across cycles).
- Captures temporal trends that single-row models miss.

### Random Forest (Classical ML)
- Strong baseline for tabular data and often very robust on noisy sensors.
- Random forests average many decision trees, which helps **reduce overfitting and improve predictive accuracy**. [web:185]

### Weighted Ensemble (Best of both)
We combine both predictions:

```text
RUL_ensemble = w * RUL_lstm + (1 - w) * RUL_rf
