# Aircraft Engine Remaining Useful Life (RUL) Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade predictive-maintenance dashboard for forecasting the **Remaining Useful Life (RUL)** of commercial aircraft turbofan engines using multi-channel sensor telemetry and ensemble machine learning.

---

## 1. Project Description

**Project Title:** Aircraft Engine Remaining Useful Life  
**Domain:** Manufacturing & Aerospace Predictive Maintenance  
**Problem Type:** Regression  
**Description:** Estimate the remaining useful life of an aircraft engine using sensor telemetry data and machine-learning regression algorithms.

Continuous mechanical and thermal operation of turbofan engines induces wear across compressors, turbines, and bearings. This project implements a machine-learning ensemble pipeline trained on engine degradation telemetry to predict RUL cycles and categorize engine health into actionable operational tiers.

---

## 2. Problem Statement

Aircraft engines operate under extreme temperatures, pressures, and rotational speeds. Over prolonged flight cycles, components experience progressive aerodynamic erosion, blade fouling, and thermal degradation. Traditional maintenance practices rely either on:
- **Preventive Maintenance:** Fixed flight hour intervals that risk over-maintaining healthy components.
- **Corrective Maintenance:** Repairs conducted after component failure, causing unscheduled groundings and safety hazards.

Predictive Maintenance (PdM) addresses these limitations by continuously monitoring sensor data to forecast Remaining Useful Life, optimizing overhaul scheduling and enhancing fleet reliability.

---

## 3. Objective

To build and deploy a machine-learning regression application that accurately estimates the Remaining Useful Life (in operational cycles) of an aircraft turbofan engine from its current sensor readings, providing real-time diagnostics and trend analysis for maintenance decision-makers.

---

## 4. Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset - FD001**

- **Origin:** NASA Ames Prognostics Center of Excellence (PCoE)
- **Engine Type:** High-bypass commercial turbofan engine
- **Operating Conditions:** Sea-level flight conditions
- **Fault Mode:** High-Pressure Compressor (HPC) degradation
- **Fleet Scale:** 100 training engines (run to complete failure) and 100 test engines (truncated prior to failure)
- **Target Variable:** Remaining Useful Life (RUL) in flight cycles, capped at 125 cycles following a piecewise linear degradation model.

---

## 5. Machine Learning Models

The predictive system utilizes an ensemble framework:
1. **Random Forest Regressor:** Ensemble of 300 bagged decision trees to minimize estimation variance.
2. **Extra Trees Regressor:** 300 extremely randomized trees offering superior robustness to sensor measurement noise.
3. **Gradient Boosting Regressor:** 300 sequential boosting stages optimizing non-linear residuals.
4. **Voting Regressor (Final Model):** Soft voting ensemble combining the predictions of the three base models to deliver generalized predictions.

---

## 6. Evaluation Metrics

Evaluated on cross-engine validation and the official 100-engine NASA C-MAPSS FD001 test set:

| Model | MAE (Cycles) | RMSE (Cycles) | R² Score |
|---|---:|---:|---:|
| **Random Forest Regressor** | 12.4816 | 17.1007 | 0.8320 |
| **Extra Trees Regressor** | 12.4261 | 16.8752 | 0.8364 |
| **Gradient Boosting Regressor** | 12.4583 | 16.9987 | 0.8340 |
| **Voting Ensemble (Validation)** | 12.3520 | 16.8422 | 0.8370 |
| **Voting Ensemble (Official Test Set)** | **12.6120** | **17.7962** | **0.8166** |

- **MAE (Mean Absolute Error):** Average prediction discrepancy in operational flight cycles.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction anomalies.
- **R² Score:** Proportion of variance in engine degradation explained by the model.

---

## 7. Selected Features

Out of 21 raw sensor channels, 6 constant or near-zero variance sensors (`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) were removed during exploratory data analysis. The 15 retained features are:

| Feature | Sensor Description | Physical Unit |
|---|---|---|
| `sensor_2` | LPC Outlet Temperature (T24) | °R |
| `sensor_3` | HPC Outlet Temperature (T30) | °R |
| `sensor_4` | LPT Outlet Temperature (T50) | °R |
| `sensor_6` | Bypass Duct Pressure (P15) | psia |
| `sensor_7` | HPC Outlet Pressure (P30) | psia |
| `sensor_8` | Physical Fan Speed (Nf) | rpm |
| `sensor_9` | Physical Core Speed (Nc) | rpm |
| `sensor_11` | Static Pressure at HPC Outlet (Ps30) | psia |
| `sensor_12` | Ratio of Fuel Flow to Ps30 (phi) | pps/psi |
| `sensor_13` | Corrected Fan Speed (Nf_dmd) | rpm |
| `sensor_14` | Corrected Core Speed (Nc_dmd) | rpm |
| `sensor_15` | Bypass Ratio (BPR) | ratio |
| `sensor_17` | Bleed Enthalpy (htBleed) | BTU/lbm |
| `sensor_20` | HPT Cool Air Flow (W31) | lbm/s |
| `sensor_21` | LPT Cool Air Flow (W32) | lbm/s |

---

## 8. Technology Stack

- **Application Framework:** Streamlit
- **Machine Learning:** Scikit-learn, XGBoost, Joblib
- **Data Manipulation:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express & Plotly Graph Objects
- **Styling:** Custom CSS (Aviation Cockpit Theme)

---

## 9. Project Structure

```text
aircraft-rul-project/
│
├── .streamlit/
│   └── config.toml                     # Dark navy aviation UI theme configuration
├── Aircraft_Engine_RUL_Ensemble.ipynb  # Training notebook (Google Colab source)
├── app.py                              # Main Streamlit web application
├── features.json                       # 15 active model input feature names
├── model_info.json                     # Model metadata, parameters & benchmark metrics
├── requirements.txt                    # Project dependencies
├── rul_ensemble_model.pkl              # Serialized trained Voting Ensemble model
├── sample_sensor_data.csv              # Sample CSV template for testing batch prediction
└── README.md                           # Comprehensive documentation
```

---

## 10. Installation

Clone or download the project folder, open a terminal in the root directory, and install the required dependencies:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 11. Running Locally

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 12. Model Files

- `rul_ensemble_model.pkl`: Serialized model file generated from `Aircraft_Engine_RUL_Ensemble.ipynb`.
- `features.json`: Specifies the exact feature ordering required for model inference.
- `model_info.json`: Stores official test metrics, dataset identifiers, and model parameters.

> **Note:** If `rul_ensemble_model.pkl` is absent, the application gracefully alerts the user with `● MODEL NOT LOADED` status without crashing.

---

## 13. GitHub Setup

1. Initialize git and commit your files:
```bash
git init
git add .
git commit -m "Initial commit: Aircraft Engine RUL Streamlit Application"
```
2. For large model files (`rul_ensemble_model.pkl` > 100 MB), configure **Git Large File Storage (Git LFS)**:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git commit -m "Configure Git LFS for model binary"
```
3. Push to GitHub:
```bash
git remote add origin https://github.com/<your-username>/aircraft-rul-project.git
git branch -M main
git push -u origin main
```

---

## 14. Streamlit Deployment

To deploy on **Streamlit Community Cloud**:
1. Log in to [share.streamlit.io](https://share.streamlit.io/) with your GitHub account.
2. Click **New app**.
3. Select your repository: `<your-username>/aircraft-rul-project`.
4. Set Main file path: `app.py`.
5. Click **Deploy!**.

The application requires no local database, GPU, or model retraining during runtime.

---

## 15. Operational Interpretation Categories

Predicted RUL values are mapped into operational guidance categories:

- **Healthy (> 80 cycles):** Nominal turbofan operation. Normal flight schedule recommended.
- **Moderate (40 - 80 cycles):** Routine telemetry monitoring recommended.
- **Warning (20 - 40 cycles):** Priority maintenance inspection recommended.
- **Critical (≤ 20 cycles):** Urgent maintenance required; components near failure boundary.

---

## 16. Limitations

1. **Simulation Bounds:** The model is trained on simulated thermodynamic cycle data from the NASA C-MAPSS simulator; real-world commercial engines operate under variable atmospheric conditions and diverse flight envelopes.
2. **Advisory Tool:** The application provides statistical estimation for maintenance decision support and does not replace official FAA/EASA certified engine airworthiness inspections.
3. **Sensor Completeness:** Accurate prediction requires all 15 active telemetry channels to be present and calibrated.
