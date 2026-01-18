# Predictive Maintenance System: RUL Estimation
> **Goal:** Estimate Remaining Useful Life (RUL) of turbofan engines to prevent unsafe failures and optimize maintenance schedules.

## 🏗️ System Architecture

```mermaid
graph TD
    User([User / Maintenance Team]) -->|Upload Test File| Input[Raw Sensor Data]
    
    subgraph "Data Pipeline"
        Input --> Clean[Feature Selection]
        Clean --> Scale[Standardization (Saved Scaler)]
        Scale --> Window[Sequence Generation (30 Cycles)]
    end
    
    subgraph "Inference Engine"
        Window --> LSTM[Model 1: LSTM (Time-Series)]
        Window --> RF[Model 2: Random Forest (Tabular)]
        LSTM --> Ensemble((Weighted Ensemble))
        RF --> Ensemble
    end
    
    Ensemble -->|RUL Prediction| Dash[Web Dashboard]
    
    subgraph "Dashboard Outputs"
        Dash --> Fleet[Fleet Insights]
        Dash --> Engine[Engine Deep Dive]
    end
    
    style Ensemble fill:#f9f,stroke:#333,stroke-width:2px
    style Dash fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 🚀 Key Features

### 1. Hybrid AI Modeling
We combine the strengths of Deep Learning and Classical ML to ensure robust predictions.

> [!TIP]
> **Why an Ensemble?**
> *   **LSTM:** Captures complex time-dependent degradation patterns.
> *   **Random Forest:** Robust to noise and effective on tabular data.
> *   **Result:** The weighted ensemble minimizes error even if one model struggles with specific engine data.

<details>
<summary><strong>🔍 Click to view Model Details</strong></summary>

### LSTM (Deep Learning)
*   **Input:** Sliding window of 30 cycles.
*   **Strength:** Learns long/short-term dependencies in sensor noise.
*   **Deployment:** Loaded in "prediction-only" mode for fast inference.

### Random Forest (Classical ML)
*   **Input:** Flattened vector of the 30-cycle window.
*   **Strength:** Stable predictions via averaging multiple decision trees.
*   **Training:** Optimized with chunked training to handle large datasets on standard hardware.

</details>

### 2. Intelligent Data Processing
Raw sensor data is automatically transformed to match the model's training conditions.

<details>
<summary><strong>⚙️ Processing Pipeline Steps</strong></summary>

1.  **Feature Selection:** Removes constant/redundant sensors that add no predictive value.
2.  **Standardization:** Applies the *exact same* scaling parameters used during training to ensure consistency.
3.  **Windowing:** Converts single-row readings into 30-cycle historical sequences to capture utilization trends.

</details>

---

## 📊 Dashboard Capabilities

The Gradio-based interface is designed for clarity and rapid decision-making.

| Feature | Description | Utility |
| :--- | :--- | :--- |
| **Fleet Insights** | Distributes RUL across all engines. | Identifies which engines are most critical immediately. |
| **Engine Deep Dive** | Specific analysis for a selected Engine ID. | Shows detailed health status and sensor trends. |
| **Health Colors** | 🟢 Healthy / 🟡 Warning / 🔴 Critical | Instant visual cues for non-technical users. |
| **Transparent AI** | Shows Ensemble RUL *and* LSTM-only RUL. | Builds trust by showing model consensus. |

---

## ⚡ Quick Start Guidance

> [!IMPORTANT]
> Ensure all trained model files (`.h5`, `.pkl`) and the scaler are present in the directory before running.

1.  **Install Dependencies:** Run the setup script to install deep learning and dashboard libraries.
2.  **Load Models:** The system will automatically load the saved LSTM and Random Forest models.
3.  **Launch Dashboard:**
    ```bash
    python dashboard_app.py
    ```
4.  **Upload Data:** Select your CMAPSS test file and view the insights!

---
*Based on NASA CMAPSS FD001 Dataset*
