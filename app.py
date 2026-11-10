"""
================================================================================
AIRCRAFT ENGINE HEALTH MONITOR - REMAINING USEFUL LIFE (RUL) PREDICTION
Streamlit Web Application
--------------------------------------------------------------------------------
Domain:       Manufacturing / Aerospace Predictive Maintenance
Problem:      Regression
Dataset:      NASA C-MAPSS Jet Engine Simulated Data (FD001)
Target:       Remaining Useful Life (RUL) in operational cycles
Ensemble:     Voting Regressor (Random Forest, Extra Trees, Gradient Boosting)
================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Suppress sklearn unpickling version warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# SCIKIT-LEARN DESERIALIZATION COMPATIBILITY ALIAS
# ------------------------------------------------------------------------------
# When unpickling models trained across different scikit-learn versions,
# Cython internal modules like '_loss' can resolve to 'sklearn._loss._loss'.
try:
    import sklearn._loss
    import sklearn._loss._loss
    sys.modules['_loss'] = sklearn._loss._loss
except Exception:
    pass

import joblib

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Aircraft Engine Health Monitor | RUL Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# CUSTOM CSS STYLING (Dark Navy / Aviation Cockpit Aesthetic)
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0B0F19;
        color: #E2E8F0;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main Header Container */
    .header-box {
        background: linear-gradient(135deg, #111827 0%, #162032 50%, #0F172A 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        color: #F8FAFC;
        margin: 0;
        padding: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #38BDF8;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* Status Badges */
    .status-badge-ready {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .status-badge-missing {
        display: inline-flex;
        align-items: center;
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Metric / KPI Cards */
    .kpi-card {
        background: #151D2A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #00D2FF;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #00D2FF;
    }

    /* Section Cards */
    .glass-card {
        background: #131B29;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .glass-card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 14px;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
    }

    /* Workflow Pipeline Step */
    .pipeline-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 10px 0;
    }
    .pipeline-step {
        background: #172033;
        border: 1px solid #2A374E;
        border-radius: 8px;
        padding: 14px 20px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .pipeline-num {
        background: #00D2FF;
        color: #0B0F19;
        font-weight: 700;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    .pipeline-text {
        font-size: 0.95rem;
        color: #E2E8F0;
        font-weight: 500;
    }
    .pipeline-arrow {
        text-align: center;
        color: #38BDF8;
        font-size: 1.2rem;
        margin: -4px 0;
    }

    /* Prediction Result Cards */
    .result-hero {
        background: linear-gradient(135deg, #111D33 0%, #162645 100%);
        border: 2px solid #00D2FF;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 25px rgba(0, 210, 255, 0.15);
    }
    .result-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .result-number {
        font-size: 3.4rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .result-unit {
        font-size: 1.2rem;
        color: #38BDF8;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Health Badges */
    .health-pill {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 24px;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 10px;
    }
    .health-healthy {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
    }
    .health-moderate {
        background-color: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border: 1px solid #38BDF8;
    }
    .health-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
    }
    .health-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #EF4444;
        border: 1px solid #EF4444;
    }

    /* Primary Predict Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0284C7 0%, #00D2FF 100%);
        color: #0B0F19;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        transition: all 0.3s ease;
        letter-spacing: 0.8px;
        box-shadow: 0 4px 14px rgba(0, 210, 255, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #0369A1 0%, #38BDF8 100%);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.5);
        transform: translateY(-1px);
        color: #0B0F19;
    }

    /* Sidebar info box */
    .sidebar-info {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 14px;
        margin-top: 20px;
    }
    .sidebar-info-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #1F2937;
        font-size: 0.82rem;
    }
    .sidebar-info-row:last-child {
        border-bottom: none;
    }
    .sidebar-info-label {
        color: #94A3B8;
        font-weight: 500;
    }
    .sidebar-info-val {
        color: #F1F5F9;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# SENSOR DESCRIPTIONS & TYPICAL BOUNDS (From NASA C-MAPSS FD001)
# ------------------------------------------------------------------------------
SENSOR_META = {
    "sensor_2": {
        "name": "LPC Outlet Temperature (T24)",
        "unit": "°R",
        "desc": "Total temperature at Low Pressure Compressor outlet",
        "default": 642.58,
        "min": 640.0,
        "max": 645.0,
        "step": 0.05
    },
    "sensor_3": {
        "name": "HPC Outlet Temperature (T30)",
        "unit": "°R",
        "desc": "Total temperature at High Pressure Compressor outlet",
        "default": 1581.22,
        "min": 1560.0,
        "max": 1615.0,
        "step": 0.1
    },
    "sensor_4": {
        "name": "LPT Outlet Temperature (T50)",
        "unit": "°R",
        "desc": "Total temperature at Low Pressure Turbine outlet",
        "default": 1398.91,
        "min": 1380.0,
        "max": 1445.0,
        "step": 0.1
    },
    "sensor_6": {
        "name": "Bypass Duct Pressure (P15)",
        "unit": "psia",
        "desc": "Total pressure in bypass-duct",
        "default": 21.60,
        "min": 21.4,
        "max": 21.8,
        "step": 0.01
    },
    "sensor_7": {
        "name": "HPC Outlet Pressure (P30)",
        "unit": "psia",
        "desc": "Total pressure at High Pressure Compressor outlet",
        "default": 554.53,
        "min": 548.0,
        "max": 557.0,
        "step": 0.05
    },
    "sensor_8": {
        "name": "Physical Fan Speed (Nf)",
        "unit": "rpm",
        "desc": "Physical rotational speed of turbofan rotor",
        "default": 2388.06,
        "min": 2387.5,
        "max": 2388.6,
        "step": 0.01
    },
    "sensor_9": {
        "name": "Physical Core Speed (Nc)",
        "unit": "rpm",
        "desc": "Physical rotational speed of core shaft",
        "default": 9054.72,
        "min": 9015.0,
        "max": 9105.0,
        "step": 0.1
    },
    "sensor_11": {
        "name": "HPC Outlet Static Pressure (Ps30)",
        "unit": "psia",
        "desc": "Static pressure at High Pressure Compressor outlet",
        "default": 46.93,
        "min": 46.5,
        "max": 48.8,
        "step": 0.02
    },
    "sensor_12": {
        "name": "Fuel Flow Ratio to Ps30 (phi)",
        "unit": "pps/psi",
        "desc": "Ratio of fuel mass flow rate to Ps30 pressure",
        "default": 522.33,
        "min": 517.0,
        "max": 524.5,
        "step": 0.05
    },
    "sensor_13": {
        "name": "Corrected Fan Speed (Nf_dmd)",
        "unit": "rpm",
        "desc": "Temperature and pressure corrected fan speed",
        "default": 2388.06,
        "min": 2387.5,
        "max": 2388.6,
        "step": 0.01
    },
    "sensor_14": {
        "name": "Corrected Core Speed (Nc_dmd)",
        "unit": "rpm",
        "desc": "Corrected rotational speed of engine core",
        "default": 8130.11,
        "min": 8095.0,
        "max": 8175.0,
        "step": 0.1
    },
    "sensor_15": {
        "name": "Bypass Ratio (BPR)",
        "unit": "ratio",
        "desc": "Ratio between bypass air and engine core air",
        "default": 8.4024,
        "min": 8.30,
        "max": 8.60,
        "step": 0.005
    },
    "sensor_17": {
        "name": "Bleed Enthalpy (htBleed)",
        "unit": "BTU/lbm",
        "desc": "Enthalpy of air bled from compressor",
        "default": 393.0,
        "min": 385.0,
        "max": 402.0,
        "step": 1.0
    },
    "sensor_20": {
        "name": "HPT Cool Air Flow (W31)",
        "unit": "lbm/s",
        "desc": "High Pressure Turbine cooling airflow rate",
        "default": 38.81,
        "min": 38.0,
        "max": 39.6,
        "step": 0.02
    },
    "sensor_21": {
        "name": "LPT Cool Air Flow (W32)",
        "unit": "lbm/s",
        "desc": "Low Pressure Turbine cooling airflow rate",
        "default": 23.3552,
        "min": 22.7,
        "max": 23.9,
        "step": 0.01
    }
}

# ------------------------------------------------------------------------------
# LOAD CONFIGURATION & METADATA
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model_info.json")
MODEL_FILE_PATH = os.path.join(BASE_DIR, "rul_ensemble_model.pkl")

@st.cache_data
def load_feature_config():
    """Load exact feature list and order from features.json."""
    if os.path.exists(FEATURES_PATH):
        try:
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Fallback to standard 15 features if file read encounters issue
    return [
        "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7",
        "sensor_8", "sensor_9", "sensor_11", "sensor_12", "sensor_13",
        "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ]

@st.cache_data
def load_model_metadata():
    """Load model information from model_info.json."""
    if os.path.exists(MODEL_INFO_PATH):
        try:
            with open(MODEL_INFO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

REQUIRED_FEATURES = load_feature_config()
MODEL_INFO = load_model_metadata()

# ------------------------------------------------------------------------------
# CACHED MODEL LOADING
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained ML ensemble model...")
def load_trained_model():
    """
    Load the trained Voting Ensemble model (rul_ensemble_model.pkl).
    Returns (model, status_message, is_loaded).
    Does not crash if model is missing or incompatible.
    """
    if not os.path.exists(MODEL_FILE_PATH):
        return None, "Trained model not found.\n\nPlease add rul_ensemble_model.pkl to the project folder.", False
    
    try:
        model = joblib.load(MODEL_FILE_PATH)
        return model, "Model loaded successfully", True
    except Exception as e:
        err_msg = f"Error deserializing model: {str(e)}\nPlease verify model file integrity."
        return None, err_msg, False

MODEL, MODEL_MSG, MODEL_LOADED = load_trained_model()

# ------------------------------------------------------------------------------
# UTILITY FUNCTIONS: HEALTH INTERPRETATION & PREDICTION
# ------------------------------------------------------------------------------
def interpret_health_status(rul_value):
    """
    Categorize predicted RUL into standardized health tiers.
    Configurable interpretation categories based on C-MAPSS degradation profile:
      - Healthy:  RUL > 80 cycles
      - Moderate: 40 < RUL <= 80 cycles
      - Warning:  20 < RUL <= 40 cycles
      - Critical: RUL <= 20 cycles
    """
    if rul_value > 80:
        return "Healthy", "health-healthy", "Engine demonstrates nominal operating behavior with minimal thermal or mechanical degradation. Normal flight operation recommended."
    elif rul_value > 40:
        return "Moderate", "health-moderate", "Engine demonstrates early-to-mid stage wear signatures. Continued telemetry monitoring is recommended during scheduled inspections."
    elif rul_value > 20:
        return "Warning", "health-warning", "Significant degradation trends detected across key pressure and temperature sensors. Priority maintenance inspection recommended."
    else:
        return "Critical", "health-critical", "Turbofan components are near failure threshold. Immediate maintenance scheduling and off-wing inspection required."

def predict_single_record(feature_dict):
    """
    Run prediction on a single engine sensor measurement dictionary.
    Returns (ensemble_pred, base_model_preds).
    """
    if not MODEL_LOADED or MODEL is None:
        return None, None
    
    # Construct DataFrame in the EXACT feature order expected by the model
    input_df = pd.DataFrame([feature_dict])[REQUIRED_FEATURES]
    
    # Generate prediction from the actual trained Voting Ensemble
    ensemble_pred = float(MODEL.predict(input_df)[0])
    
    # Extract individual base model predictions if available
    base_preds = {}
    try:
        if hasattr(MODEL, "named_estimators_") and hasattr(MODEL, "estimators_"):
            for name, estimator in zip(MODEL.named_estimators_.keys(), MODEL.estimators_):
                clean_name = name.replace("_", " ").title()
                val = float(estimator.predict(input_df)[0])
                base_preds[clean_name] = round(val, 2)
    except Exception:
        pass
        
    return ensemble_pred, base_preds

def predict_batch_df(df):
    """
    Run predictions on an uploaded DataFrame.
    Validates columns, orders them, and applies the trained model.
    """
    if not MODEL_LOADED or MODEL is None:
        raise ValueError("Model is not loaded. Cannot run batch predictions.")
    
    # Select exact required features
    input_df = df[REQUIRED_FEATURES].copy()
    
    # Validate numeric types
    for col in REQUIRED_FEATURES:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    if input_df.isna().any().any():
        raise ValueError("The uploaded CSV contains non-numeric or missing sensor values.")
        
    preds = MODEL.predict(input_df)
    return preds

# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ✈️ SYSTEM NAVIGATION")
    page = st.radio(
        "Select Module:",
        [
            "Dashboard",
            "RUL Prediction",
            "Sensor Analysis",
            "Model Comparison",
            "Model Information",
            "About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    
    # Model Status Indicator in Sidebar
    if MODEL_LOADED:
        st.markdown(
            '<div class="status-badge-ready">● AI MODEL READY</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="status-badge-missing">● MODEL NOT LOADED</div>',
            unsafe_allow_html=True
        )
    
    # Sidebar Project Metadata Card
    st.markdown(
        """
        <div class="sidebar-info">
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Domain</span>
                <span class="sidebar-info-val">Manufacturing</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Problem</span>
                <span class="sidebar-info-val">Regression</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Dataset</span>
                <span class="sidebar-info-val">NASA C-MAPSS</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-label">Target</span>
                <span class="sidebar-info-val">Remaining Useful Life</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Predictive Maintenance System • Turbofan C-MAPSS FD001")

# ------------------------------------------------------------------------------
# APPLICATION MAIN HEADER
# ------------------------------------------------------------------------------
status_html = (
    '<span class="status-badge-ready">● AI MODEL READY</span>'
    if MODEL_LOADED
    else '<span class="status-badge-missing">● MODEL NOT LOADED</span>'
)

st.markdown(
    f"""
    <div class="header-box">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <h1 class="header-title">✈️ AIRCRAFT ENGINE HEALTH MONITOR</h1>
                <div class="header-subtitle">Remaining Useful Life Prediction using Machine Learning</div>
            </div>
            <div>
                {status_html}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# If model is missing, display the designated warning message without crashing
if not MODEL_LOADED:
    st.warning(
        """
        **Trained model not found.**
        
        Please add `rul_ensemble_model.pkl` to the project folder.
        """
    )


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "Dashboard":
    # 1. Four Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">DOMAIN</div>
                <div class="kpi-value">Manufacturing</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">PROBLEM</div>
                <div class="kpi-value">Regression</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">DATASET</div>
                <div class="kpi-value">NASA C-MAPSS</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-label">TARGET</div>
                <div class="kpi-value">RUL (Cycles)</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Project Objective Section
    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-title">🎯 Project Objective</div>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #CBD5E1; margin: 0;">
                Estimate the remaining useful life of an aircraft engine using sensor data and machine-learning regression models.
            </p>
            <p style="font-size: 0.92rem; color: #94A3B8; margin-top: 10px; margin-bottom: 0;">
                Continuous operation of turbofan engines results in thermal and mechanical wear of internal aerodynamic components.
                By ingesting multi-channel sensor telemetry (temperatures, pressures, fan speeds, bleed air, and bypass ratios),
                this system forecasts the operational cycles remaining before unscheduled maintenance or component overhaul is mandated.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. Workflow Pipeline Visualization
    col_pipe, col_metrics = st.columns([1.1, 0.9])
    
    with col_pipe:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">🔄 End-to-End ML Pipeline Architecture</div>
                <div class="pipeline-container">
                    <div class="pipeline-step">
                        <div class="pipeline-num">1</div>
                        <div class="pipeline-text"><b>Aircraft Engine Sensor Data</b><br><span style="color:#94A3B8; font-size:0.85rem;">21 raw telemetry channels + operational flight cycle tracking</span></div>
                    </div>
                    <div class="pipeline-arrow">▼</div>
                    <div class="pipeline-step">
                        <div class="pipeline-num">2</div>
                        <div class="pipeline-text"><b>Data Preprocessing & RUL Calculation</b><br><span style="color:#94A3B8; font-size:0.85rem;">Run-to-failure cycle calculation capped at 125 cycles</span></div>
                    </div>
                    <div class="pipeline-arrow">▼</div>
                    <div class="pipeline-step">
                        <div class="pipeline-num">3</div>
                        <div class="pipeline-text"><b>Feature Selection</b><br><span style="color:#94A3B8; font-size:0.85rem;">Variance filtering removed 6 uninformative sensors; 15 retained</span></div>
                    </div>
                    <div class="pipeline-arrow">▼</div>
                    <div class="pipeline-step">
                        <div class="pipeline-num">4</div>
                        <div class="pipeline-text"><b>Machine Learning Models & Voting Ensemble</b><br><span style="color:#94A3B8; font-size:0.85rem;">Random Forest + Extra Trees + Gradient Boosting Regressors</span></div>
                    </div>
                    <div class="pipeline-arrow">▼</div>
                    <div class="pipeline-step">
                        <div class="pipeline-num">5</div>
                        <div class="pipeline-text"><b>Model Evaluation & Engine Health Insight</b><br><span style="color:#94A3B8; font-size:0.85rem;">MAE: 12.61 cycles | R²: 0.8166 | Health status tier assignment</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_metrics:
        # Display Official Model Test Metrics
        test_mae = MODEL_INFO["test_metrics"]["MAE"] if MODEL_INFO else 12.6120
        test_rmse = MODEL_INFO["test_metrics"]["RMSE"] if MODEL_INFO else 17.7962
        test_r2 = MODEL_INFO["test_metrics"]["R2"] if MODEL_INFO else 0.8166
        
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="glass-card-title">📊 Evaluated Benchmark Performance</div>
                <p style="font-size: 0.9rem; color: #94A3B8;">
                    Results evaluated on the official NASA C-MAPSS FD001 test fleet (100 independent test engines):
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 18px 0;">
                    <div style="background:#172033; padding:12px; border-radius:8px; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-weight:600;">TEST MAE</div>
                        <div style="font-size:1.35rem; color:#38BDF8; font-weight:700;">{test_mae}</div>
                        <div style="font-size:0.72rem; color:#64748B;">cycles</div>
                    </div>
                    <div style="background:#172033; padding:12px; border-radius:8px; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-weight:600;">TEST RMSE</div>
                        <div style="font-size:1.35rem; color:#38BDF8; font-weight:700;">{test_rmse}</div>
                        <div style="font-size:0.72rem; color:#64748B;">cycles</div>
                    </div>
                    <div style="background:#172033; padding:12px; border-radius:8px; text-align:center;">
                        <div style="font-size:0.75rem; color:#94A3B8; font-weight:600;">TEST R² SCORE</div>
                        <div style="font-size:1.35rem; color:#34D399; font-weight:700;">{test_r2}</div>
                        <div style="font-size:0.72rem; color:#64748B;">variance explained</div>
                    </div>
                </div>
                <div style="background:#0F172A; border-left: 3px solid #00D2FF; padding: 12px 16px; border-radius: 4px; font-size: 0.85rem; color: #CBD5E1;">
                    <b>Ensemble Architecture:</b> Voting Regressor averaging predictions across 3 diverse tree-based regression algorithms to minimize prediction variance and guard against single-model overfitting.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Navigation Quick Actions
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">⚡ Quick Actions</div>
                <p style="font-size: 0.88rem; color: #94A3B8;">
                    Explore real-time predictive degradation estimation or inspect degradation trends across cycles:
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🚀 Run RUL Prediction", use_container_width=True):
                st.session_state["nav_to_pred"] = True
                st.rerun()
        with c_btn2:
            if st.button("📈 Explore Sensor Analysis", use_container_width=True):
                st.session_state["nav_to_sensor"] = True
                st.rerun()


# ==============================================================================
# PAGE 2: RUL PREDICTION (MAIN FUNCTIONALITY)
# ==============================================================================
elif page == "RUL Prediction":
    st.markdown("## 🛠️ Predict Remaining Useful Life")
    st.markdown(
        "Enter the current aircraft engine sensor measurements to estimate its remaining useful life."
    )
    
    # Check if model is loaded before proceeding with prediction UI
    if not MODEL_LOADED:
        st.error(
            """
            **Trained model not found.**
            
            Please add `rul_ensemble_model.pkl` to the project folder to enable inference.
            """
        )
    
    # Provide two operational tabs: Single Engine Assessment & CSV Batch Upload
    pred_tab1, pred_tab2 = st.tabs([
        "🔍 Single Engine Assessment (Interactive Inputs)",
        "📁 Batch Assessment (CSV Upload)"
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: SINGLE ENGINE INPUTS
    # --------------------------------------------------------------------------
    with pred_tab1:
        st.markdown("#### Engine Sensor Telemetry")
        st.caption(
            f"The model uses {len(REQUIRED_FEATURES)} selected sensor measurements in the exact configuration defined in features.json."
        )
        
        # Preset Quick Loaders for Demonstration
        st.markdown("##### 💡 Load Representative Engine Presets:")
        preset_cols = st.columns(4)
        
        # Define presets based on actual dataset observations
        presets = {
            "healthy": {
                'sensor_2': 641.82, 'sensor_3': 1589.70, 'sensor_4': 1400.60, 'sensor_6': 21.61,
                'sensor_7': 554.36, 'sensor_8': 2388.06, 'sensor_9': 9046.19, 'sensor_11': 47.47,
                'sensor_12': 521.66, 'sensor_13': 2388.02, 'sensor_14': 8138.62, 'sensor_15': 8.4195,
                'sensor_17': 392.0, 'sensor_20': 39.06, 'sensor_21': 23.4190
            },
            "midlife": {
                'sensor_2': 642.58, 'sensor_3': 1581.22, 'sensor_4': 1398.91, 'sensor_6': 21.60,
                'sensor_7': 554.53, 'sensor_8': 2388.06, 'sensor_9': 9054.72, 'sensor_11': 46.93,
                'sensor_12': 522.33, 'sensor_13': 2388.06, 'sensor_14': 8130.11, 'sensor_15': 8.4024,
                'sensor_17': 393.0, 'sensor_20': 38.81, 'sensor_21': 23.3552
            },
            "warning": {
                'sensor_2': 642.88, 'sensor_3': 1589.75, 'sensor_4': 1418.89, 'sensor_6': 21.61,
                'sensor_7': 553.80, 'sensor_8': 2388.14, 'sensor_9': 9064.50, 'sensor_11': 47.60,
                'sensor_12': 521.80, 'sensor_13': 2388.14, 'sensor_14': 8131.46, 'sensor_15': 8.4119,
                'sensor_17': 395.0, 'sensor_20': 38.93, 'sensor_21': 23.2740
            },
            "critical": {
                'sensor_2': 643.50, 'sensor_3': 1602.10, 'sensor_4': 1428.40, 'sensor_6': 21.61,
                'sensor_7': 551.40, 'sensor_8': 2388.18, 'sensor_9': 9072.10, 'sensor_11': 48.15,
                'sensor_12': 519.80, 'sensor_13': 2388.18, 'sensor_14': 8120.20, 'sensor_15': 8.5120,
                'sensor_17': 397.0, 'sensor_20': 38.45, 'sensor_21': 23.1100
            }
        }
        
        # Initialize session state for sensor inputs if not set
        for f in REQUIRED_FEATURES:
            default_val = SENSOR_META.get(f, {}).get("default", 100.0)
            if f not in st.session_state:
                st.session_state[f] = default_val

        with preset_cols[0]:
            if st.button("🟢 Nominal (Early Life)"):
                for k, v in presets["healthy"].items():
                    st.session_state[k] = v
                st.rerun()
        with preset_cols[1]:
            if st.button("🔵 Moderate Wear (Mid Life)"):
                for k, v in presets["midlife"].items():
                    st.session_state[k] = v
                st.rerun()
        with preset_cols[2]:
            if st.button("🟠 Advisory Warning"):
                for k, v in presets["warning"].items():
                    st.session_state[k] = v
                st.rerun()
        with preset_cols[3]:
            if st.button("🔴 Critical Degradation"):
                for k, v in presets["critical"].items():
                    st.session_state[k] = v
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3-Column Grid for Sensor Inputs
        cols = st.columns(3)
        input_values = {}
        
        for idx, feature_key in enumerate(REQUIRED_FEATURES):
            col_target = cols[idx % 3]
            meta = SENSOR_META.get(feature_key, {
                "name": feature_key,
                "unit": "",
                "desc": "Sensor measurement",
                "default": 100.0,
                "min": 0.0,
                "max": 10000.0,
                "step": 0.1
            })
            
            with col_target:
                label_text = f"{feature_key.upper()} ({meta['unit']})" if meta['unit'] else feature_key.upper()
                help_text = f"{meta['name']} — {meta['desc']}"
                
                # Retrieve current value from session state
                curr_val = float(st.session_state.get(feature_key, meta["default"]))
                
                val = st.number_input(
                    label=label_text,
                    value=curr_val,
                    format="%.4f" if meta["step"] < 0.1 else "%.2f",
                    step=float(meta["step"]),
                    help=help_text,
                    key=f"input_{feature_key}"
                )
                input_values[feature_key] = float(val)
                st.session_state[feature_key] = float(val)
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Prominent Predict Button
        predict_clicked = st.button("🚀 PREDICT RUL", use_container_width=True)
        
        if predict_clicked:
            if not MODEL_LOADED:
                st.error("Cannot predict: Model file rul_ensemble_model.pkl is missing.")
            else:
                with st.spinner("Executing inference across Random Forest, Extra Trees, and Gradient Boosting models..."):
                    pred_rul, base_preds = predict_single_record(input_values)
                    
                    if pred_rul is not None:
                        # Store in session state for cross-page persistence
                        st.session_state["last_prediction"] = pred_rul
                        st.session_state["last_inputs"] = input_values
                        st.session_state["last_base_preds"] = base_preds
                        
                        health_label, health_class, explanation = interpret_health_status(pred_rul)
                        
                        # Display Large Professional Result Card
                        st.markdown(
                            f"""
                            <div class="result-hero">
                                <div class="result-title">ESTIMATED REMAINING USEFUL LIFE</div>
                                <div class="result-number">{pred_rul:.1f} <span class="result-unit">CYCLES</span></div>
                                <div style="margin-top: 14px;">
                                    <span class="health-pill {health_class}">ENGINE HEALTH STATUS: {health_label.upper()}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        # Visual Health Progress Bar / Gauge
                        col_gauge, col_info = st.columns([1, 1])
                        
                        with col_gauge:
                            # Plotly Gauge Chart for visual health
                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=round(pred_rul, 1),
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Engine Health Meter (RUL Cycles)", 'font': {'color': "#E2E8F0", 'size': 14}},
                                number={'suffix': " cyc", 'font': {'color': "#00D2FF", 'size': 26}},
                                gauge={
                                    'axis': {'range': [0, 125], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                                    'bar': {'color': "#00D2FF", 'thickness': 0.25},
                                    'bgcolor': "#162032",
                                    'borderwidth': 1,
                                    'bordercolor': "#334155",
                                    'steps': [
                                        {'range': [0, 20], 'color': 'rgba(239, 68, 68, 0.3)'},
                                        {'range': [20, 40], 'color': 'rgba(245, 158, 11, 0.3)'},
                                        {'range': [40, 80], 'color': 'rgba(56, 189, 248, 0.3)'},
                                        {'range': [80, 125], 'color': 'rgba(16, 185, 129, 0.3)'}
                                    ],
                                    'threshold': {
                                        'line': {'color': "#EF4444", 'width': 3},
                                        'thickness': 0.75,
                                        'value': 20
                                    }
                                }
                            ))
                            fig_gauge.update_layout(
                                height=240,
                                margin=dict(l=20, r=20, t=40, b=20),
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={'color': "#CBD5E1"}
                            )
                            st.plotly_chart(fig_gauge, use_container_width=True)
                            
                        with col_info:
                            st.markdown(
                                f"""
                                <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                                    <div class="glass-card-title">🔍 Diagnostic Assessment</div>
                                    <p style="font-size: 0.95rem; color: #F1F5F9; line-height: 1.5;">
                                        <b>Current status: {health_label}</b>
                                    </p>
                                    <p style="font-size: 0.9rem; color: #94A3B8; line-height: 1.5;">
                                        {explanation}
                                    </p>
                                    <div style="background:#0F172A; border-left: 3px solid #64748B; padding: 10px 14px; border-radius: 4px; font-size: 0.78rem; color: #94A3B8; margin-top: 10px;">
                                        ⚠️ <b>Operational Advisory:</b> This model provides predictive degradation estimations for maintenance scheduling analysis and does not guarantee aircraft safety or failure prevention.
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                        # Base Model Decomposition Breakdown
                        if base_preds:
                            st.markdown("##### 🤝 Voting Ensemble Constituent Estimations:")
                            bm_cols = st.columns(len(base_preds))
                            for idx, (m_name, m_val) in enumerate(base_preds.items()):
                                with bm_cols[idx]:
                                    st.markdown(
                                        f"""
                                        <div style="background:#131D2D; border:1px solid #1E293B; border-radius:8px; padding:12px; text-align:center;">
                                            <div style="font-size:0.75rem; color:#94A3B8; font-weight:600;">{m_name}</div>
                                            <div style="font-size:1.3rem; color:#38BDF8; font-weight:700;">{m_val:.1f}</div>
                                            <div style="font-size:0.7rem; color:#64748B;">cycles</div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

    # --------------------------------------------------------------------------
    # TAB 2: CSV BATCH UPLOAD
    # --------------------------------------------------------------------------
    with pred_tab2:
        st.markdown("#### Batch Telemetry Assessment via CSV")
        st.write(
            "Upload a comma-separated values (CSV) file containing multi-channel engine sensor records. "
            "The system validates column integrity, ignores extra non-sensor columns, and generates RUL predictions for each row."
        )
        
        # Download Sample CSV Button
        sample_csv_path = os.path.join(BASE_DIR, "sample_sensor_data.csv")
        if os.path.exists(sample_csv_path):
            with open(sample_csv_path, "rb") as f_sample:
                sample_bytes = f_sample.read()
            st.download_button(
                label="📥 Download Sample Sensor CSV Template",
                data=sample_bytes,
                file_name="sample_sensor_data.csv",
                mime="text/csv",
                help="Click to download a ready-to-test CSV with valid C-MAPSS sensor readings"
            )
        
        uploaded_file = st.file_uploader(
            "Select CSV file for batch processing:",
            type=["csv"],
            help="Upload a CSV file containing the 15 required sensor columns"
        )
        
        if uploaded_file is not None:
            try:
                # 1. Read CSV using Pandas
                batch_df = pd.read_csv(uploaded_file)
                
                # Check for empty CSV
                if batch_df.empty:
                    st.error("The uploaded CSV file is empty. Please upload a file with data.")
                else:
                    st.write(f"Successfully loaded CSV with **{len(batch_df)} records** and **{len(batch_df.columns)} columns**.")
                    
                    # 2. Validate required columns
                    missing_cols = [c for c in REQUIRED_FEATURES if c not in batch_df.columns]
                    
                    if missing_cols:
                        st.error(
                            f"""
                            **Missing required sensor columns:**  
                            `{', '.join(missing_cols)}`  
                            
                            Please verify that your CSV contains all 15 required features.
                            """
                        )
                    else:
                        st.success("All 15 required sensor columns are present. Ready for inference.")
                        
                        # Preview uploaded data
                        with st.expander("👀 View Uploaded Data Preview", expanded=False):
                            st.dataframe(batch_df.head(10), use_container_width=True)
                        
                        # Generate Batch Predictions Button
                        if st.button("⚡ GENERATE BATCH PREDICTIONS", use_container_width=True):
                            if not MODEL_LOADED:
                                st.error("Trained model not found. Cannot run batch predictions.")
                            else:
                                with st.spinner("Processing batch predictions with trained ensemble model..."):
                                    predictions = predict_batch_df(batch_df)
                                    
                                    # Create results DataFrame
                                    results_df = batch_df.copy()
                                    results_df["Predicted_RUL"] = np.round(predictions, 2)
                                    results_df["Health_Status"] = results_df["Predicted_RUL"].apply(
                                        lambda x: interpret_health_status(x)[0]
                                    )
                                    
                                    # Store in session state for the Sensor Analysis page
                                    st.session_state["batch_results"] = results_df
                                    
                                    # Summary KPIs across batch
                                    avg_rul = results_df["Predicted_RUL"].mean()
                                    min_rul = results_df["Predicted_RUL"].min()
                                    healthy_count = (results_df["Health_Status"] == "Healthy").sum()
                                    critical_count = (results_df["Health_Status"] == "Critical").sum()
                                    
                                    st.markdown("##### 📈 Batch Overview:")
                                    b1, b2, b3, b4 = st.columns(4)
                                    b1.metric("Average Predicted RUL", f"{avg_rul:.1f} cyc")
                                    b2.metric("Minimum Predicted RUL", f"{min_rul:.1f} cyc")
                                    b3.metric("Healthy Engines", f"{healthy_count} / {len(results_df)}")
                                    b4.metric("Critical Engines", f"{critical_count} / {len(results_df)}")
                                    
                                    # Display Results Table
                                    st.markdown("##### 📋 Engine Predictions Table:")
                                    
                                    # Format display columns
                                    display_cols = []
                                    if "unit" in results_df.columns:
                                        display_cols.append("unit")
                                    if "cycle" in results_df.columns:
                                        display_cols.append("cycle")
                                    display_cols.extend(["Predicted_RUL", "Health_Status"])
                                    
                                    # If unit/cycle not present, create a Record ID column
                                    if not ("unit" in results_df.columns or "cycle" in results_df.columns):
                                        results_df.insert(0, "Record", [f"Record #{i+1}" for i in range(len(results_df))])
                                        display_cols = ["Record", "Predicted_RUL", "Health_Status"]
                                        
                                    st.dataframe(
                                        results_df[display_cols],
                                        use_container_width=True
                                    )
                                    
                                    # Download Results Button
                                    csv_output = results_df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="💾 Download Predictions CSV",
                                        data=csv_output,
                                        file_name="rul_prediction_results.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                    
            except Exception as e:
                st.error(f"Error parsing CSV file: {str(e)}")


# ==============================================================================
# PAGE 3: SENSOR ANALYSIS
# ==============================================================================
elif page == "Sensor Analysis":
    st.markdown("## 📊 Sensor Analysis")
    st.markdown(
        "Analyze the current engine sensor telemetry and inspect degradation trends across operational flight cycles."
    )
    
    # 1. Clean Sensor Data Table
    st.markdown("### Current Sensor Telemetry Profile")
    
    # Get current sensor inputs from session state or defaults
    current_sensors = {}
    for f in REQUIRED_FEATURES:
        meta = SENSOR_META.get(f, {})
        current_sensors[f] = {
            "Sensor Identifier": f,
            "Description": meta.get("name", f),
            "Engineering Unit": meta.get("unit", "N/A"),
            "Current Value": st.session_state.get(f, meta.get("default", 100.0))
        }
        
    sensor_df = pd.DataFrame(list(current_sensors.values()))
    
    col_tbl, col_dist = st.columns([1, 1.2])
    
    with col_tbl:
        st.dataframe(
            sensor_df[["Sensor Identifier", "Description", "Current Value", "Engineering Unit"]],
            use_container_width=True,
            height=460
        )
        
    with col_dist:
        st.markdown("#### Sensor Value Distribution")
        # Plot horizontal or bar distribution
        # Normalize sensor values to percentage of their typical range to make comparison meaningful
        norm_data = []
        for f in REQUIRED_FEATURES:
            meta = SENSOR_META.get(f, {})
            val = st.session_state.get(f, meta.get("default", 100.0))
            min_v = meta.get("min", val * 0.9)
            max_v = meta.get("max", val * 1.1)
            norm_val = ((val - min_v) / (max_v - min_v)) * 100 if max_v > min_v else 50
            norm_data.append({
                "Sensor": f,
                "Current Value": val,
                "Relative Position in Range (%)": np.clip(norm_val, 0, 100)
            })
            
        norm_df = pd.DataFrame(norm_data)
        
        fig_bar = px.bar(
            norm_df,
            x="Sensor",
            y="Current Value",
            title="Telemetry Readings Across 15 Active Sensors",
            color="Current Value",
            color_continuous_scale="Viridis"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#E2E8F0"),
            height=430,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1E293B")
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    # 2. Sensor Trend (Cycle vs Sensor Value)
    st.markdown("### 📈 Sensor Degradation Trend Analysis")
    st.caption("Track sensor telemetry variations across engine operational flight cycles to identify wear progression.")
    
    # Check if user has uploaded batch data with cycle column or load sample data
    trend_df = None
    if "batch_results" in st.session_state and "cycle" in st.session_state["batch_results"].columns:
        trend_df = st.session_state["batch_results"]
    else:
        sample_csv_path = os.path.join(BASE_DIR, "sample_sensor_data.csv")
        if os.path.exists(sample_csv_path):
            trend_df = pd.read_csv(sample_csv_path)
            
    if trend_df is not None and "cycle" in trend_df.columns:
        # Check if multiple engines are present
        if "unit" in trend_df.columns and trend_df["unit"].nunique() > 1:
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                selected_unit = st.selectbox(
                    "Select Engine Unit:",
                    options=trend_df["unit"].unique(),
                    index=0
                )
            with col_sel2:
                selected_sensor = st.selectbox(
                    "Select Telemetry Sensor Channel:",
                    options=REQUIRED_FEATURES,
                    index=0,
                    format_func=lambda x: f"{x} - {SENSOR_META.get(x, {}).get('name', x)}"
                )
            unit_trend_df = trend_df[trend_df["unit"] == selected_unit].sort_values("cycle")
        else:
            selected_sensor = st.selectbox(
                "Select Telemetry Sensor Channel:",
                options=REQUIRED_FEATURES,
                index=0,
                format_func=lambda x: f"{x} - {SENSOR_META.get(x, {}).get('name', x)}"
            )
            unit_trend_df = trend_df.sort_values("cycle")
            
        if len(unit_trend_df) > 1:
            fig_trend = px.line(
                unit_trend_df,
                x="cycle",
                y=selected_sensor,
                markers=True,
                title=f"Cycle vs {selected_sensor.upper()} ({SENSOR_META.get(selected_sensor, {}).get('name', '')})",
                labels={"cycle": "Operational Flight Cycle", selected_sensor: f"{selected_sensor} Value"}
            )
            fig_trend.update_traces(line_color="#00D2FF", marker=dict(size=6, color="#38BDF8"))
            fig_trend.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.6)",
                font=dict(color="#E2E8F0"),
                height=380,
                xaxis=dict(showgrid=True, gridcolor="#1E293B"),
                yaxis=dict(showgrid=True, gridcolor="#1E293B")
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown(
                f"""
                <div style="background:#131B29; border:1px solid #1E293B; border-radius:8px; padding:14px; font-size:0.88rem; color:#94A3B8;">
                    💡 <b>Degradation Signature:</b> As aircraft turbofans accumulate cycles, critical thermodynamic parameters exhibit characteristic drift. 
                    For example, compressor exit temperatures and static pressures increase due to fouling and clearance expansion, 
                    while fuel-flow efficiency ratios deteriorate steadily before engine failure.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("The available dataset contains only a single cycle observation. Upload a multi-cycle dataset to observe longitudinal trend progressions.")
    else:
        st.info("No multi-cycle dataset detected. Upload a CSV with a `cycle` column on the RUL Prediction page to view temporal degradation trajectories.")


# ==============================================================================
# PAGE 4: MODEL COMPARISON
# ==============================================================================
elif page == "Model Comparison":
    st.markdown("## 🔬 Machine Learning Model Comparison")
    st.markdown(
        "Compare the performance characteristics of individual base regressors versus the ensemble architecture on the NASA C-MAPSS FD001 dataset."
    )
    
    # Actual evaluation metrics from training notebook (Cells 29 & 36)
    # Validation set: Random Forest, Extra Trees, Gradient Boosting, Voting Ensemble
    # Official FD001 Test set: Voting Ensemble
    val_metrics = pd.DataFrame({
        "Model": [
            "Random Forest Regressor",
            "Extra Trees Regressor",
            "Gradient Boosting Regressor",
            "Voting Ensemble (Validation)"
        ],
        "MAE": [12.4816, 12.4261, 12.4583, 12.3520],
        "RMSE": [17.1007, 16.8752, 16.9987, 16.8422],
        "R²": [0.8320, 0.8364, 0.8340, 0.8370]
    })
    
    # Official test set metric from model_info.json / Notebook Cell 36
    test_metrics_data = {
        "Model": "Voting Ensemble (Official Test Set)",
        "MAE": MODEL_INFO["test_metrics"]["MAE"] if MODEL_INFO else 12.6120,
        "RMSE": MODEL_INFO["test_metrics"]["RMSE"] if MODEL_INFO else 17.7962,
        "R²": MODEL_INFO["test_metrics"]["R2"] if MODEL_INFO else 0.8166
    }
    
    # Display Clean Comparison Table
    st.markdown("### 📋 Evaluated Performance Metrics")
    
    col_tbl1, col_tbl2 = st.columns([1.3, 0.7])
    with col_tbl1:
        st.markdown("##### Validation Set Comparison (Cross-Engine Validation)")
        st.dataframe(
            val_metrics.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
            use_container_width=True
        )
    with col_tbl2:
        st.markdown("##### Official FD001 Test Fleet")
        test_df_disp = pd.DataFrame([test_metrics_data])
        st.dataframe(
            test_df_disp.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
            use_container_width=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Plotly Comparison Charts for MAE, RMSE, R²
    st.markdown("### 📊 Interactive Performance Benchmarks")
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    
    with chart_col1:
        fig_mae = px.bar(
            val_metrics,
            x="Model",
            y="MAE",
            title="Mean Absolute Error (Lower is Better)",
            color="MAE",
            color_continuous_scale="Blues_r"
        )
        fig_mae.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#E2E8F0"),
            height=320,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(tickangle=-25, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1E293B", range=[11, 13])
        )
        st.plotly_chart(fig_mae, use_container_width=True)
        
    with chart_col2:
        fig_rmse = px.bar(
            val_metrics,
            x="Model",
            y="RMSE",
            title="Root Mean Squared Error (Lower is Better)",
            color="RMSE",
            color_continuous_scale="Purples_r"
        )
        fig_rmse.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#E2E8F0"),
            height=320,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(tickangle=-25, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1E293B", range=[15, 18])
        )
        st.plotly_chart(fig_rmse, use_container_width=True)
        
    with chart_col3:
        fig_r2 = px.bar(
            val_metrics,
            x="Model",
            y="R²",
            title="R² Variance Explained (Higher is Better)",
            color="R²",
            color_continuous_scale="Teal"
        )
        fig_r2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#E2E8F0"),
            height=320,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(tickangle=-25, showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1E293B", range=[0.80, 0.85])
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        
    st.markdown("---")
    
    # --------------------------------------------------------------------------
    # FEATURE IMPORTANCE SECTION
    # --------------------------------------------------------------------------
    st.markdown("### 🔍 Sensor Feature Importance")
    st.caption("Relative importance ranking of sensor features as determined by the trained tree ensemble.")
    
    # Exact feature importance values from training notebook (Cell 32)
    feat_imp_data = pd.DataFrame({
        "Sensor": [
            "sensor_11", "sensor_4", "sensor_12", "sensor_9", "sensor_7",
            "sensor_14", "sensor_15", "sensor_21", "sensor_20", "sensor_2",
            "sensor_17", "sensor_3", "sensor_13", "sensor_8", "sensor_6"
        ],
        "Importance": [
            0.190707, 0.143845, 0.119773, 0.078002, 0.077902,
            0.063303, 0.060464, 0.059070, 0.048351, 0.040703,
            0.039269, 0.031050, 0.026454, 0.020531, 0.000577
        ]
    })
    
    # Add descriptive sensor names
    feat_imp_data["Description"] = feat_imp_data["Sensor"].apply(
        lambda x: f"{x} ({SENSOR_META.get(x, {}).get('name', x)})"
    )
    
    # Sort ascending for clean horizontal bar chart display
    feat_imp_sorted = feat_imp_data.sort_values(by="Importance", ascending=True)
    
    fig_imp = px.bar(
        feat_imp_sorted,
        x="Importance",
        y="Description",
        orientation="h",
        title="Relative Contribution of Input Sensors to RUL Prediction",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(color="#E2E8F0"),
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor="#1E293B", title="Relative Importance Weight"),
        yaxis=dict(title="")
    )
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.markdown(
        """
        <div style="background:#0F172A; border-left: 3px solid #38BDF8; padding: 12px 18px; border-radius: 4px; font-size: 0.88rem; color: #CBD5E1;">
            <b>Scientific Note:</b> Feature importance indicates the relative contribution of input variables to model predictions and does not by itself establish physical causation.
            Sensors such as <code>sensor_11</code> (HPC Outlet Static Pressure), <code>sensor_4</code> (LPT Outlet Temperature), and <code>sensor_12</code> (Fuel Flow Ratio) reflect strong physical coupling to turbine aerodynamic degradation.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 5: MODEL INFORMATION
# ==============================================================================
elif page == "Model Information":
    st.markdown("## ℹ️ Model Architecture & Metadata")
    st.markdown(
        "Technical configuration, hyperparameter specifications, and feature schema of the trained model."
    )
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">⚙️ Core Model Specifications</div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Problem Type</span>
                    <span class="sidebar-info-val">Regression</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Domain</span>
                    <span class="sidebar-info-val">Manufacturing / Aerospace</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Dataset</span>
                    <span class="sidebar-info-val">NASA C-MAPSS FD001</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Target Variable</span>
                    <span class="sidebar-info-val">Remaining Useful Life (RUL)</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">RUL Capping Threshold</span>
                    <span class="sidebar-info-val">125 Cycles (Piecewise Linear Degradation)</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Ensemble Method</span>
                    <span class="sidebar-info-val">Voting Regressor (Soft Ensemble)</span>
                </div>
                <div class="sidebar-info-row">
                    <span class="sidebar-info-label">Input Dimension</span>
                    <span class="sidebar-info-val">15 Selected Sensor Features</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_info2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">🤖 Base Estimators in Voting Ensemble</div>
                <div style="margin-bottom: 12px;">
                    <b style="color:#38BDF8;">1. Random Forest Regressor</b><br>
                    <span style="font-size:0.85rem; color:#94A3B8;">
                    • Estimators: 300 decision trees<br>
                    • Splitting Criterion: Squared error<br>
                    • Reduces variance via bootstrap bagging
                    </span>
                </div>
                <div style="margin-bottom: 12px;">
                    <b style="color:#38BDF8;">2. Extra Trees Regressor</b><br>
                    <span style="font-size:0.85rem; color:#94A3B8;">
                    • Estimators: 300 extremely randomized trees<br>
                    • Extreme randomization reduces correlation among base trees<br>
                    • Superior robustness to sensor measurement noise
                    </span>
                </div>
                <div style="margin-bottom: 12px;">
                    <b style="color:#38BDF8;">3. Gradient Boosting Regressor</b><br>
                    <span style="font-size:0.85rem; color:#94A3B8;">
                    • Estimators: 300 boosting stages, learning rate = 0.05, max_depth = 3<br>
                    • Sequential residual boosting minimizes subtle non-linear biases
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("### 📑 Selected Sensor Feature Schema")
    st.write(
        "During exploratory data analysis, sensors with near-zero variance across all operating cycles "
        "(`sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) were eliminated to prevent noise contamination. "
        "The remaining 15 active sensors are:"
    )
    
    schema_rows = []
    for f in REQUIRED_FEATURES:
        meta = SENSOR_META.get(f, {})
        schema_rows.append({
            "Feature Name": f,
            "Description": meta.get("name", "Sensor Channel"),
            "Physical Unit": meta.get("unit", "N/A"),
            "Sensor Definition": meta.get("desc", "Turbofan measurement channel")
        })
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)


# ==============================================================================
# PAGE 6: ABOUT PROJECT
# ==============================================================================
elif page == "About Project":
    st.markdown("## 📖 About The Project")
    
    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-title">✈️ Problem Statement</div>
            <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">
                Aircraft turbofan engines undergo continuous physical degradation during operation due to thermal cycling, 
                aerodynamic erosion, vibration, and mechanical fatigue. In legacy commercial aviation practices, engine maintenance 
                relies heavily on static scheduled flight hour thresholds or corrective repairs following component faults. 
                Predictive maintenance leverages continuous sensor streams to estimate <b>Remaining Useful Life (RUL)</b>, 
                allowing airline fleet operators to schedule overhauls just-in-time, minimize unscheduled ground delays, 
                and optimize maintenance lifecycle costs.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">🎯 Objective</div>
                <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    Estimate the remaining useful life of an aircraft engine using sensor data and machine-learning regression models.
                    The model outputs the forecasted number of remaining operational cycles until the engine reaches its functional failure boundary.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">📦 Dataset: NASA C-MAPSS</div>
                <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    The <b>Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)</b> dataset was generated by NASA Ames Research Center. 
                    Subset <b>FD001</b> simulates high-bypass commercial turbofans operating under standard sea-level flight conditions with a single failure mode: 
                    high-pressure compressor (HPC) degradation.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_a2:
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">🤖 Machine Learning Models</div>
                <ul style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    <li><b>Random Forest Regressor:</b> Ensemble of decision trees with bagging.</li>
                    <li><b>Extra Trees Regressor:</b> Randomized tree ensemble reducing variance.</li>
                    <li><b>Gradient Boosting Regressor:</b> Sequential gradient-based residual optimization.</li>
                    <li><b>Voting Regressor:</b> Combines predictions through equal-weight ensemble averaging.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div class="glass-card">
                <div class="glass-card-title">📐 Model Evaluation Metrics</div>
                <ul style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    <li><b>Mean Absolute Error (MAE):</b> Average magnitude of error in flight cycles.</li>
                    <li><b>Root Mean Squared Error (RMSE):</b> Quadratic metric penalizing severe outlier prediction errors.</li>
                    <li><b>R² (Coefficient of Determination):</b> Proportion of variance explained by the ensemble.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="glass-card">
            <div class="glass-card-title">💡 System Output & Operational Scope</div>
            <p style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                The primary output is the <b>Estimated Remaining Useful Life in operational flight cycles</b>. 
                Predictions are categorized into qualitative operational tiers (Healthy, Moderate, Warning, Critical) to guide engineering workflow decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.8rem; padding: 10px 0;">
        Aircraft Engine Health Monitor • NASA C-MAPSS Machine Learning Project • Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
