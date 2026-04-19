"""
AQI Prediction Dashboard
Visualizes and compares Manual FIS, GA-Tuned FIS, and ANFIS models
for Air Quality Index prediction using PM2.5 and NO2 inputs.
"""

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import json
import os
import itertools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e2e8f0;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.3);
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(139, 92, 246, 0.3);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #c4b5fd;
        margin: 24px 0 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(139, 92, 246, 0.4);
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(59, 130, 246, 0.2) 100%);
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #c4b5fd, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Prediction output */
    .prediction-box {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15), rgba(59, 130, 246, 0.15));
        border: 1px solid rgba(139, 92, 246, 0.5);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin: 8px 0;
    }
    .prediction-model {
        font-size: 0.85rem;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: 800;
        color: #f1f5f9;
        line-height: 1.1;
    }
    .prediction-category {
        font-size: 1rem;
        font-weight: 600;
        margin-top: 8px;
        padding: 4px 14px;
        border-radius: 20px;
        display: inline-block;
    }

    /* AQI category colors */
    .cat-good        { background: #059669; color: #d1fae5; }
    .cat-moderate    { background: #d97706; color: #fef3c7; }
    .cat-unhealthy-s { background: #ea580c; color: #ffedd5; }
    .cat-unhealthy   { background: #dc2626; color: #fee2e2; }
    .cat-very-unhealthy { background: #9333ea; color: #f3e8ff; }
    .cat-hazardous   { background: #7f1d1d; color: #fef2f2; }

    /* Info box */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.875rem;
        color: #93c5fd;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(139, 92, 246, 0.3) !important;
        color: #c4b5fd !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #8b5cf6, #3b82f6);
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Success/warning/error */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px;
    }

    /* Number input container */
    div[data-testid="stNumberInput"] {
        position: relative;
        padding-top: 2px;
    }

    /* Number input styling */
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        color: black !important;
        border-radius: 10px !important;
        text-align: center !important;
        font-weight: 600 !important;
        height: 50px !important;
        padding-top: 15px !important; /* Make room for the title in the box */
        /* Balance centering by adding padding to the left to offset the buttons on the right */
        padding-left: 45px !important; 
    }
    
    /* Custom label styling - we'll use this in the HTML */
    .input-box-title {
        position: absolute;
        width: 100%;
        text-align: center;
        top: 6px;
        left: 0;
        z-index: 10;
        font-size: 0.65rem;
        font-weight: 700;
        color: yellow;
        text-transform: uppercase;
        pointer-events: none;
        letter-spacing: 0.05em;
    }

    /* Selection box */
    .stSelectbox select {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(139, 92, 246, 0.4) !important;
        color: #f1f5f9 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        width: 100%;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
    }

    /* Hide streamlit branding and header white bar */
    header[data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0) !important;
        background: transparent !important;
    }
    
    header { 
        visibility: hidden !important;
    }

    /* Reduce top padding for a cleaner look */
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* Hide the deploy button specifically */
    [data-testid="stDeployButton"] {
        display: none !important;
    }

    /* Hide redundant menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ─── Model Helpers ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

# ── Membership Functions ────────────────────────────────────────────────────
def triangular_mf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a)
    if b < x < c:
        return (c - x) / (c - b)
    return 0.0


def left_shoulder_mf(x: float, a: float, b: float) -> float:
    if x <= a:
        return 1.0
    if a < x < b:
        return (b - x) / (b - a)
    return 0.0


def right_shoulder_mf(x: float, a: float, b: float) -> float:
    if x <= a:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    return 1.0


def trapezoidal_mf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a)
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


def fuzzify(value: float, mf_config: dict) -> dict:
    results = {}
    for set_name, (mf_type, params) in mf_config.items():
        if mf_type == "triangular":
            results[set_name] = triangular_mf(value, *params)
        elif mf_type == "trapezoidal":
            results[set_name] = trapezoidal_mf(value, *params)
        elif mf_type == "left_shoulder":
            results[set_name] = left_shoulder_mf(value, *params)
        elif mf_type == "right_shoulder":
            results[set_name] = right_shoulder_mf(value, *params)
    return results


def evaluate_rule(inputs_memberships: dict, rule_definition: dict, weight: float = 1.0):
    degrees = []
    for var_name, set_name in rule_definition["if"]:
        degree = inputs_memberships.get(var_name, {}).get(set_name, 0.0)
        degrees.append(degree)
    firing_strength = min(degrees) if degrees else 0.0
    return firing_strength * weight, rule_definition["then"]


def sugeno_defuzzification(firing_strengths: list, rule_outputs: list) -> float:
    total_weight = sum(firing_strengths)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(w * z for w, z in zip(firing_strengths, rule_outputs))
    return weighted_sum / total_weight


def predict_sugeno(input_values: dict, mf_configs: dict, rule_base: list, rule_weights=None) -> float:
    inputs_memberships = {}
    for var_name, value in input_values.items():
        inputs_memberships[var_name] = fuzzify(value, mf_configs[var_name])
    firing_strengths = []
    rule_outputs = []
    for i, rule in enumerate(rule_base):
        weight = rule_weights[i] if rule_weights is not None else 1.0
        w, z = evaluate_rule(inputs_memberships, rule, weight)
        firing_strengths.append(w)
        rule_outputs.append(z)
    return sugeno_defuzzification(firing_strengths, rule_outputs)


# ── ANFIS Model ─────────────────────────────────────────────────────────────
class ANFISLayer(nn.Module):
    def __init__(self, n_inputs=2, n_terms=3):
        super(ANFISLayer, self).__init__()
        self.n_inputs = n_inputs
        self.n_terms = n_terms
        self.rule_idx = list(itertools.product(range(n_terms), repeat=n_inputs))
        self.centers = nn.Parameter(torch.zeros(n_inputs, n_terms))
        self.sigmas = nn.Parameter(torch.ones(n_inputs, n_terms))
        self.consequents = nn.Parameter(torch.zeros(n_terms**n_inputs, n_inputs + 1))

    def forward(self, x):
        batch = x.size(0)
        x_exp = x.unsqueeze(2)
        c_exp = self.centers.unsqueeze(0)
        s_exp = self.sigmas.unsqueeze(0).abs() + 1e-6
        mu = torch.exp(-((x_exp - c_exp) ** 2) / (2 * s_exp**2))
        w_list = []
        for r, idx in enumerate(self.rule_idx):
            w_r = mu[:, 0, idx[0]]
            for i in range(1, len(idx)):
                w_r = w_r * mu[:, i, idx[i]]
            w_list.append(w_r)
        w = torch.stack(w_list, dim=1)
        w_sum = w.sum(dim=1, keepdim=True) + 1e-8
        w_norm = w / w_sum
        x_aug = torch.cat([x, torch.ones(batch, 1, device=x.device)], dim=1)
        f = x_aug @ self.consequents.T
        return (w_norm * f).sum(dim=1, keepdim=True)


# ── AQI Category ────────────────────────────────────────────────────────────
def get_aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "cat-good", "🟢"
    elif aqi <= 100:
        return "Moderate", "cat-moderate", "🟡"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "cat-unhealthy-s", "🟠"
    elif aqi <= 200:
        return "Unhealthy", "cat-unhealthy", "🔴"
    elif aqi <= 300:
        return "Very Unhealthy", "cat-very-unhealthy", "🟣"
    else:
        return "Hazardous", "cat-hazardous", "⚫"


def aqi_category_color(aqi: float) -> str:
    if aqi <= 50:
        return "#059669"
    elif aqi <= 100:
        return "#d97706"
    elif aqi <= 150:
        return "#ea580c"
    elif aqi <= 200:
        return "#dc2626"
    elif aqi <= 300:
        return "#9333ea"
    return "#7f1d1d"


# ═══════════════════════════════════════════════════════════════════════════
# ─── Data / Model Loading (Cached) ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "ai_train", "results")
DATASET_DIR = os.path.join(BASE_DIR, "..", "ai_train", "dataset")


@st.cache_resource
def load_models():
    """Load all model parameters from the ai_train/results directory."""
    errors = []

    # ── Manual FIS Config (hard-coded from stage4) ────────────────────────
    PM25_MANUAL = {
        "Low": ("left_shoulder", [30, 60]),
        "Medium": ("triangular", [45, 90, 135]),
        "High": ("right_shoulder", [110, 200]),
    }
    NO2_MANUAL = {
        "Low": ("left_shoulder", [40, 80]),
        "Medium": ("triangular", [60, 130, 200]),
        "High": ("right_shoulder", [160, 300]),
    }
    RULE_BASE = [
        {"if": [("pm2.5", "High"), ("no2", "High")],       "then": 500},
        {"if": [("pm2.5", "High"), ("no2", "Medium")],     "then": 350},
        {"if": [("pm2.5", "High"), ("no2", "Low")],        "then": 250},
        {"if": [("pm2.5", "Medium"), ("no2", "High")],     "then": 350},
        {"if": [("pm2.5", "Medium"), ("no2", "Medium")],   "then": 150},
        {"if": [("pm2.5", "Medium"), ("no2", "Low")],      "then": 100},
        {"if": [("pm2.5", "Low"), ("no2", "High")],        "then": 250},
        {"if": [("pm2.5", "Low"), ("no2", "Medium")],      "then": 100},
        {"if": [("pm2.5", "Low"), ("no2", "Low")],         "then": 50},
    ]

    # ── GA-Tuned FIS ─────────────────────────────────────────────────────
    ga_data = None
    try:
        ga_path = os.path.join(RESULTS_DIR, "ga_results.json")
        with open(ga_path, "r") as f:
            ga_data = json.load(f)
    except Exception as e:
        errors.append(f"GA results: {e}")

    # ── ANFIS ─────────────────────────────────────────────────────────────
    scalers = None
    anfis_model = None
    try:
        scaler_path = os.path.join(RESULTS_DIR, "scalers.pkl")
        with open(scaler_path, "rb") as f:
            scalers = pickle.load(f)
    except Exception as e:
        errors.append(f"ANFIS scalers: {e}")

    try:
        anfis_model = ANFISLayer(n_inputs=2, n_terms=3)
        model_path = os.path.join(RESULTS_DIR, "anfis_model.pth")
        anfis_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        anfis_model.eval()
    except Exception as e:
        errors.append(f"ANFIS weights: {e}")

    return {
        "pm25_manual": PM25_MANUAL,
        "no2_manual": NO2_MANUAL,
        "rule_base": RULE_BASE,
        "ga_data": ga_data,
        "scalers": scalers,
        "anfis": anfis_model,
        "errors": errors,
    }


@st.cache_data
def load_dataset():
    """Load and preprocess the Delhi AQI dataset."""
    try:
        csv_path = os.path.join(DATASET_DIR, "city_day.csv")
        raw_df = pd.read_csv(csv_path)
        delhi_df = (
            raw_df[raw_df["City"] == "Delhi"][["Date", "PM2.5", "NO2", "AQI"]]
            .dropna()
            .copy()
        )
        delhi_df["Date"] = pd.to_datetime(delhi_df["Date"])
        delhi_df = delhi_df.sort_values("Date").reset_index(drop=True)
        return delhi_df, None
    except Exception as e:
        return None, str(e)


def run_all_predictions(models, X_raw):
    """Run inference for all three models on the given input array."""
    results = {}

    # Manual FIS
    preds_manual = np.array([
        predict_sugeno(
            {"pm2.5": p, "no2": n},
            {"pm2.5": models["pm25_manual"], "no2": models["no2_manual"]},
            models["rule_base"],
        )
        for p, n in X_raw
    ])
    results["Manual FIS"] = preds_manual

    # GA-Tuned FIS
    if models["ga_data"]:
        gd = models["ga_data"]
        preds_ga = np.array([
            predict_sugeno(
                {"pm2.5": p, "no2": n},
                {"pm2.5": gd["pm25_mf_ga"], "no2": gd["no2_mf_ga"]},
                models["rule_base"],
                gd["rw_ga"],
            )
            for p, n in X_raw
        ])
        results["GA-Tuned FIS"] = preds_ga

    # ANFIS
    if models["scalers"] and models["anfis"] is not None:
        scaler_X = models["scalers"]["scaler_X"]
        scaler_y = models["scalers"]["scaler_y"]
        X_norm = scaler_X.transform(X_raw)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        with torch.no_grad():
            y_norm_pred = models["anfis"](X_tensor).numpy()
        preds_anfis = scaler_y.inverse_transform(y_norm_pred).flatten()
        results["ANFIS"] = preds_anfis

    return results


def single_prediction(models, pm25: float, no2: float) -> dict:
    """Single-point prediction for all three models."""
    out = {}

    # Manual FIS
    out["Manual FIS"] = predict_sugeno(
        {"pm2.5": pm25, "no2": no2},
        {"pm2.5": models["pm25_manual"], "no2": models["no2_manual"]},
        models["rule_base"],
    )

    # GA-Tuned FIS
    if models["ga_data"]:
        gd = models["ga_data"]
        out["GA-Tuned FIS"] = predict_sugeno(
            {"pm2.5": pm25, "no2": no2},
            {"pm2.5": gd["pm25_mf_ga"], "no2": gd["no2_mf_ga"]},
            models["rule_base"],
            gd["rw_ga"],
        )

    # ANFIS
    if models["scalers"] and models["anfis"] is not None:
        scaler_X = models["scalers"]["scaler_X"]
        scaler_y = models["scalers"]["scaler_y"]
        X_raw = np.array([[pm25, no2]])
        X_norm = scaler_X.transform(X_raw)
        X_tensor = torch.tensor(X_norm, dtype=torch.float32)
        with torch.no_grad():
            y_norm = models["anfis"](X_tensor).numpy()
        out["ANFIS"] = float(scaler_y.inverse_transform(y_norm).flatten()[0])

    return out


# ─── Membership Function Plot ────────────────────────────────────────────────
def plot_membership_functions(pm25_mf, no2_mf, title_prefix="Manual"):
    """Create plotly figure showing both PM2.5 and NO2 membership functions."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["PM2.5 Membership Functions", "NO2 Membership Functions"],
    )

    colors = {"Low": "#34d399", "Medium": "#60a5fa", "High": "#f87171"}

    for mf_config, col, x_range in [
        (pm25_mf, 1, np.linspace(0, 500, 500)),
        (no2_mf, 2, np.linspace(0, 400, 400)),
    ]:
        for set_name, (mf_type, params) in mf_config.items():
            if mf_type == "left_shoulder":
                y = [left_shoulder_mf(x, *params) for x in x_range]
            elif mf_type == "right_shoulder":
                y = [right_shoulder_mf(x, *params) for x in x_range]
            elif mf_type == "triangular":
                y = [triangular_mf(x, *params) for x in x_range]
            elif mf_type == "trapezoidal":
                y = [trapezoidal_mf(x, *params) for x in x_range]
            else:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y,
                    name=set_name,
                    line=dict(color=colors.get(set_name, "#a78bfa"), width=2.5),
                    fill="tozeroy",
                    fillcolor=colors.get(set_name, "#a78bfa").replace(
                        ")", ", 0.15)"
                    ).replace("rgb", "rgba").replace("#", "rgba(")
                    if colors.get(set_name, "#a78bfa").startswith("rgba")
                    else f"rgba(100,100,200,0.1)",
                    showlegend=(col == 1),
                    legendgroup=set_name,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(
        title=f"{title_prefix} — Fuzzy Membership Functions",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.04)",
        font=dict(family="Inter", color="#e2e8f0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(139,92,246,0.4)",
            borderwidth=1,
            font=dict(color="white")
        ),
        height=350,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False, range=[0, 1.1])
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ─── Main App ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Load models & data
    models = load_models()
    df, ds_err = load_dataset()

    # ──── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 8px;">
            <span style="font-size:3rem;">🌫️</span>
            <h2 style="color:#c4b5fd; margin:4px 0; font-size:1.2rem;">AQI Prediction</h2>
            <p style="color:#64748b; font-size:0.75rem;">Soft Computing Mid-Term</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<p style="color:#a78bfa; font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em;">Navigation</p>', unsafe_allow_html=True)

        page = st.radio(
            "",
            ["🏠 Dashboard", "🔮 Single Prediction", "📊 Batch Evaluation", "📈 Model Comparison", "🔬 Membership Functions"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown('<p style="color:#a78bfa; font-weight:600; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em;">Model Status</p>', unsafe_allow_html=True)

        status_items = [
            ("Manual FIS", True),
            ("GA-Tuned FIS", models["ga_data"] is not None),
            ("ANFIS", models["anfis"] is not None),
        ]
        for name, ok in status_items:
            icon = "✅" if ok else "❌"
            color = "#34d399" if ok else "#f87171"
            st.markdown(f'<p style="color:{color}; font-size:0.85rem;">{icon} {name}</p>', unsafe_allow_html=True)

        if models["errors"]:
            st.markdown("---")
            st.warning("Some models failed to load:\n" + "\n".join(models["errors"]))

        st.markdown("---")
        st.markdown('<p style="color:#64748b; font-size:0.72rem; text-align:center;">Delhi Air Quality Dataset<br>Inputs: PM2.5 · NO2<br>Output: AQI (Sugeno FIS)</p>', unsafe_allow_html=True)

    # ──── Page: Dashboard ──────────────────────────────────────────────────
    if page == "🏠 Dashboard":
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">AQI Prediction Dashboard</div>
            <div class="hero-subtitle">
                Comparing Manual FIS · GA-Tuned FIS · ANFIS<br>
                for Air Quality Index Prediction using PM2.5 and NO2
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Quick stats from dataset
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            stats = [
                (col1, "📅 Data Points", f"{len(df):,}", "City of Delhi"),
                (col2, "🌡️ Avg AQI", f"{df['AQI'].mean():.1f}", f"Max: {df['AQI'].max():.0f}"),
                (col3, "💨 Avg PM2.5", f"{df['PM2.5'].mean():.1f}", "µg/m³"),
                (col4, "🧪 Avg NO2", f"{df['NO2'].mean():.1f}", "µg/m³"),
            ]
            for col, label, value, sub in stats:
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📅 AQI Time Series — Delhi</div>', unsafe_allow_html=True)

        if df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df["AQI"],
                name="Actual AQI",
                line=dict(color="#60a5fa", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(96,165,250,0.08)",
            ))
            # AQI category threshold lines
            thresholds = [(50, "Good", "#34d399"), (100, "Moderate", "#fbbf24"),
                          (150, "USG", "#f97316"), (200, "Unhealthy", "#ef4444"),
                          (300, "V.Unhealthy", "#a855f7")]
            for thr, lbl, clr in thresholds:
                fig.add_hline(y=thr, line=dict(color=clr, width=1, dash="dot"),
                              annotation_text=lbl, annotation_font_color=clr,
                              annotation_position="top right")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(family="Inter", color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI"),
                height=380,
                legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # PM2.5 & NO2 trend
            st.markdown('<div class="section-header">💨 PM2.5 & NO2 Time Series</div>', unsafe_allow_html=True)

            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=["PM2.5 (µg/m³)", "NO2 (µg/m³)"])
            fig2.add_trace(go.Scatter(x=df["Date"], y=df["PM2.5"],
                                      line=dict(color="#a78bfa", width=1.5),
                                      name="PM2.5", fill="tozeroy",
                                      fillcolor="rgba(167,139,250,0.08)"), row=1, col=1)
            fig2.add_trace(go.Scatter(x=df["Date"], y=df["NO2"],
                                      line=dict(color="#34d399", width=1.5),
                                      name="NO2", fill="tozeroy",
                                      fillcolor="rgba(52,211,153,0.08)"), row=2, col=1)
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(family="Inter", color="#e2e8f0"),
                height=400,
                showlegend=False,
            )
            fig2.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig2.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error(f"Dataset not found: {ds_err}")

        # About the models
        st.markdown('<div class="section-header">🧠 About the Models</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        models_info = [
            (c1, "Manual FIS", "#8b5cf6",
             "A Sugeno-style Fuzzy Inference System with hand-crafted membership functions for PM2.5 and NO2, and 9 IF-THEN rules that map pollutant levels to AQI.",
             ["❌ Manual parameter tuning", "📐 9 rules", "🔵 Zero-order Sugeno"]),
            (c2, "GA-Tuned FIS", "#3b82f6",
             "Same FIS architecture, but membership function parameters and rule weights are optimized using a Genetic Algorithm to minimize prediction error.",
             ["✅ Optimal MF parameters", "🧬 Genetic Algorithm", "⚖️ Rule weights tuned"]),
            (c3, "ANFIS", "#10b981",
             "Adaptive Neuro-Fuzzy Inference System trained end-to-end with gradient descent. Uses Gaussian membership functions and linear consequents.",
             ["✅ Gradient descent training", "🔢 9 rules (3³ combos)", "🤖 Neural network backend"]),
        ]
        for col, name, color, desc, features in models_info:
            with col:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border:1px solid {color}55;
                    border-radius:16px; padding:20px; height:100%;">
                    <h3 style="color:{color}; margin:0 0 8px;">{name}</h3>
                    <p style="color:#94a3b8; font-size:0.85rem; margin-bottom:12px;">{desc}</p>
                    {''.join(f'<p style="color:#cbd5e1; font-size:0.8rem; margin:4px 0;">{f}</p>' for f in features)}
                </div>
                """, unsafe_allow_html=True)

    # ──── Page: Single Prediction ──────────────────────────────────────────
    elif page == "🔮 Single Prediction":
        st.markdown('<div class="hero-title" style="font-size:1.8rem; text-align:center; margin-bottom:8px;">🔮 Single Point Prediction</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; color:#94a3b8; margin-bottom:24px;">Enter PM2.5 and NO2 values to get AQI predictions from all three models.</p>', unsafe_allow_html=True)

        col_input, col_output = st.columns([1, 2], gap="large")

        with col_input:
            st.markdown('<h3 style="color:#c4b5fd; margin:0 0 20px;">Input Parameters</h3>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown('<div class="input-box-title">PM2.5 (µg/m³)</div>', unsafe_allow_html=True)
                pm25 = st.number_input("PM2.5", value=120.0, min_value=0.0, max_value=1000.0, step=0.1, key="pm25_num", help="Fine particulate matter concentration", label_visibility="collapsed")
            with col_b:
                st.markdown('<div class="input-box-title">NO2 (µg/m³)</div>', unsafe_allow_html=True)
                no2 = st.number_input("NO2", value=80.0, min_value=0.0, max_value=1000.0, step=0.1, key="no2_num", help="Nitrogen dioxide concentration", label_visibility="collapsed")

            predict_btn = st.button("🔮 Predict AQI", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_output:
            preds = single_prediction(models, pm25, no2)

            model_colors = {
                "Manual FIS": "#8b5cf6",
                "GA-Tuned FIS": "#3b82f6",
                "ANFIS": "#10b981",
            }

            for model_name, aqi_val in preds.items():
                cat, css_cls, icon = get_aqi_category(aqi_val)
                color = model_colors.get(model_name, "#a78bfa")
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04); border:1px solid {color}55;
                    border-radius:16px; padding:20px; margin-bottom:12px; display:flex;
                    align-items:center; gap:20px;">
                    <div style="min-width:140px;">
                        <div style="color:{color}; font-size:0.8rem; text-transform:uppercase;
                            letter-spacing:0.08em; margin-bottom:4px;">{model_name}</div>
                        <div style="font-size:2.5rem; font-weight:800; color:#f1f5f9;
                            line-height:1;">{aqi_val:.1f}</div>
                        <div style="font-size:0.7rem; color:#64748b; margin-top:2px;">AQI</div>
                    </div>
                    <div style="flex:1;">
                        <span class="prediction-category {css_cls}">{icon} {cat}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge charts
            if preds:
                st.markdown("---")
                cols = st.columns(len(preds))
                for i, (model_name, aqi_val) in enumerate(preds.items()):
                    color = model_colors.get(model_name, "#a78bfa")
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=aqi_val,
                        title=dict(text=model_name, font=dict(color="#c4b5fd", size=13)),
                        number=dict(font=dict(color="#f1f5f9", size=28)),
                        gauge=dict(
                            axis=dict(range=[0, 500], tickcolor="#64748b"),
                            bar=dict(color=color),
                            bgcolor="rgba(0,0,0,0)",
                            steps=[
                                dict(range=[0, 50], color="rgba(5,150,105,0.2)"),
                                dict(range=[50, 100], color="rgba(217,119,6,0.2)"),
                                dict(range=[100, 150], color="rgba(234,88,12,0.2)"),
                                dict(range=[150, 200], color="rgba(220,38,38,0.2)"),
                                dict(range=[200, 300], color="rgba(147,51,234,0.2)"),
                                dict(range=[300, 500], color="rgba(127,29,29,0.2)"),
                            ],
                            threshold=dict(
                                line=dict(color=color, width=3),
                                thickness=0.75,
                                value=aqi_val,
                            ),
                        ),
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Inter", color="#e2e8f0"),
                        height=220,
                        margin=dict(l=20, r=20, t=30, b=10),
                    )
                    with cols[i]:
                        st.plotly_chart(fig_gauge, use_container_width=True)

            # Radar comparison
            if len(preds) > 1:
                st.markdown('<div class="section-header">📡 Prediction Comparison Radar</div>', unsafe_allow_html=True)
                model_names = list(preds.keys())
                values = [preds[m] for m in model_names]
                max_val = max(values) if values else 1

                # Normalize to 0-100 for display
                fig_radar = go.Figure()
                for mn, val in zip(model_names, values):
                    color = model_colors.get(mn, "#a78bfa")
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[val, pm25 / 6, no2 / 4, val],
                        theta=["AQI Prediction", "PM2.5 (scaled)", "NO2 (scaled)", "AQI Prediction"],
                        name=mn,
                        line=dict(color=color, width=2),
                        fill="toself",
                        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if "rgb" in color else f"rgba(100,100,200,0.1)",
                    ))
                fig_radar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    polar=dict(
                        radialaxis=dict(visible=True, color="#64748b"),
                        bgcolor="rgba(255,255,255,0.03)",
                    ),
                    legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(color="white")),
                    height=350,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        # AQI categories reference
        st.markdown('<div class="section-header">🗂️ AQI Category Reference</div>', unsafe_allow_html=True)
        ref_data = [
            ("0 – 50", "Good", "#059669", "Air quality is satisfactory; little or no risk."),
            ("51 – 100", "Moderate", "#d97706", "Acceptable; some pollutants may be a moderate health concern."),
            ("101 – 150", "Unhealthy for Sensitive Groups", "#ea580c", "Sensitive groups may experience health effects."),
            ("151 – 200", "Unhealthy", "#dc2626", "General public may experience health effects."),
            ("201 – 300", "Very Unhealthy", "#9333ea", "Health alert; everyone may experience serious effects."),
            ("301+", "Hazardous", "#7f1d1d", "Health warning of emergency conditions. Everyone affected."),
        ]
        cols_ref = st.columns(3)
        for i, (rng, cat, clr, desc) in enumerate(ref_data):
            with cols_ref[i % 3]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border-left:4px solid {clr};
                    border-radius:0 10px 10px 0; padding:10px 14px; margin-bottom:10px;">
                    <div style="color:{clr}; font-weight:700; font-size:0.9rem;">{rng}</div>
                    <div style="color:#f1f5f9; font-weight:600; font-size:0.85rem;">{cat}</div>
                    <div style="color:#94a3b8; font-size:0.75rem; margin-top:3px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    # ──── Page: Batch Evaluation ────────────────────────────────────────────
    elif page == "📊 Batch Evaluation":
        st.markdown('<div class="hero-title" style="font-size:1.8rem; text-align:center; margin-bottom:8px;">📊 Batch Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; color:#94a3b8; margin-bottom:24px;">Run all three models over the full Delhi dataset and visualize predictions.</p>', unsafe_allow_html=True)

        if df is None:
            st.error(f"Dataset not available: {ds_err}")
            return

        # Subset selector
        col_a, col_b = st.columns([2, 1])
        with col_a:
            n_samples = st.slider(
                "Number of samples to evaluate",
                min_value=50, max_value=len(df),
                value=min(500, len(df)), step=50,
            )
        with col_b:
            sample_mode = st.selectbox("Sampling mode", ["First N", "Last N", "Random"])

        if sample_mode == "First N":
            eval_df = df.iloc[:n_samples]
        elif sample_mode == "Last N":
            eval_df = df.iloc[-n_samples:]
        else:
            eval_df = df.sample(n=n_samples, random_state=42).sort_values("Date")

        run_btn = st.button("▶️ Run Batch Prediction", use_container_width=False)

        if run_btn or "batch_results" not in st.session_state:
            with st.spinner("Running predictions on all models…"):
                X_raw = eval_df[["PM2.5", "NO2"]].values
                y_true = eval_df["AQI"].values
                batch_preds = run_all_predictions(models, X_raw)
                st.session_state["batch_results"] = {
                    "preds": batch_preds,
                    "y_true": y_true,
                    "dates": eval_df["Date"].values,
                }
            st.success(f"✅ Evaluated {n_samples} samples.")

        if "batch_results" in st.session_state:
            br = st.session_state["batch_results"]
            preds = br["preds"]
            y_true = br["y_true"]
            dates = br["dates"]

            model_colors = {
                "Manual FIS": "#8b5cf6",
                "GA-Tuned FIS": "#3b82f6",
                "ANFIS": "#10b981",
            }

            # Metrics
            st.markdown('<div class="section-header">📏 Performance Metrics</div>', unsafe_allow_html=True)
            metrics_cols = st.columns(len(preds) + 1)
            with metrics_cols[0]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 Samples</div>
                    <div class="metric-value">{len(y_true):,}</div>
                    <div class="metric-sub">Delhi · City Day</div>
                </div>
                """, unsafe_allow_html=True)

            for i, (model_name, y_pred) in enumerate(preds.items(), 1):
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                r2 = r2_score(y_true, y_pred)
                color = model_colors.get(model_name, "#a78bfa")
                with metrics_cols[i]:
                    st.markdown(f"""
                    <div style="background:rgba(255,255,255,0.04); border:1px solid {color}55;
                        border-radius:16px; padding:16px; text-align:center;">
                        <div style="color:{color}; font-size:0.8rem; text-transform:uppercase;
                            letter-spacing:0.06em; margin-bottom:8px;">{model_name}</div>
                        <div style="display:flex; justify-content:space-around; margin-top:8px;">
                            <div>
                                <div style="color:#64748b; font-size:0.7rem;">MAE</div>
                                <div style="color:#f1f5f9; font-size:1.1rem; font-weight:700;">{mae:.1f}</div>
                            </div>
                            <div>
                                <div style="color:#64748b; font-size:0.7rem;">RMSE</div>
                                <div style="color:#f1f5f9; font-size:1.1rem; font-weight:700;">{rmse:.1f}</div>
                            </div>
                            <div>
                                <div style="color:#64748b; font-size:0.7rem;">R²</div>
                                <div style="color:#f1f5f9; font-size:1.1rem; font-weight:700;">{r2:.3f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Time-series comparison
            st.markdown('<div class="section-header">📅 Prediction vs Actual (Time Series)</div>', unsafe_allow_html=True)
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=dates, y=y_true,
                name="Actual AQI",
                line=dict(color="#f1f5f9", width=1.5),
                opacity=0.7,
            ))
            for model_name, y_pred in preds.items():
                color = model_colors.get(model_name, "#a78bfa")
                fig_ts.add_trace(go.Scatter(
                    x=dates, y=y_pred,
                    name=model_name,
                    line=dict(color=color, width=1.5, dash="dash" if model_name == "Manual FIS" else "solid"),
                    opacity=0.85,
                ))
            fig_ts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(family="Inter", color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI"),
                legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1, font=dict(color="white")),
                height=400,
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            # Scatter actual vs predicted
            st.markdown('<div class="section-header">🎯 Actual vs Predicted Scatter</div>', unsafe_allow_html=True)
            n_cols = len(preds)
            scatter_cols = st.columns(n_cols)
            for i, (model_name, y_pred) in enumerate(preds.items()):
                color = model_colors.get(model_name, "#a78bfa")
                r2 = r2_score(y_true, y_pred)
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=y_true, y=y_pred,
                    mode="markers",
                    marker=dict(color=color, size=4, opacity=0.5),
                    name="Predictions",
                ))
                max_v = max(y_true.max(), y_pred.max())
                fig_sc.add_trace(go.Scatter(
                    x=[0, max_v], y=[0, max_v],
                    mode="lines",
                    line=dict(color="#64748b", dash="dot", width=1.5),
                    name="Perfect fit",
                ))
                fig_sc.update_layout(
                    title=dict(text=f"{model_name}  (R²={r2:.3f})", font=dict(color=color, size=13)),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(255,255,255,0.03)",
                    font=dict(family="Inter", color="#e2e8f0"),
                    xaxis=dict(title="Actual AQI", gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="Predicted AQI", gridcolor="rgba(255,255,255,0.06)"),
                    height=340,
                    showlegend=False,
                )
                with scatter_cols[i]:
                    st.plotly_chart(fig_sc, use_container_width=True)

    # ──── Page: Model Comparison ────────────────────────────────────────────
    elif page == "📈 Model Comparison":
        st.markdown('<div class="hero-title" style="font-size:1.8rem; text-align:center; margin-bottom:8px;">📈 Model Comparison</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; color:#94a3b8; margin-bottom:24px;">Side-by-side comparison of all three models over the full dataset.</p>', unsafe_allow_html=True)

        if df is None:
            st.error(f"Dataset not available: {ds_err}")
            return

        with st.spinner("Running full dataset evaluation…"):
            X_raw = df[["PM2.5", "NO2"]].values
            y_true = df["AQI"].values
            all_preds = run_all_predictions(models, X_raw)

        model_colors = {
            "Manual FIS": "#8b5cf6",
            "GA-Tuned FIS": "#3b82f6",
            "ANFIS": "#10b981",
        }

        # Metrics table
        rows = []
        for model_name, y_pred in all_preds.items():
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            rows.append({"Model": model_name, "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R² Score": round(r2, 6)})

        metrics_df = pd.DataFrame(rows)

        # Highlight winner
        best_mae = metrics_df["MAE"].min()
        best_rmse = metrics_df["RMSE"].min()
        best_r2 = metrics_df["R² Score"].max()

        st.markdown('<div class="section-header">🏆 Metrics Summary</div>', unsafe_allow_html=True)
        st.dataframe(
            metrics_df.style
                .highlight_min(subset=["MAE", "RMSE"], color="#1a472a")
                .highlight_max(subset=["R² Score"], color="#1a472a")
                .set_properties(**{"background-color": "rgba(240, 240, 240, 0.9)", "color": "black"}),
            use_container_width=True,
        )

        # Bar charts
        st.markdown('<div class="section-header">📊 Metric Bar Charts</div>', unsafe_allow_html=True)
        fig_bar = make_subplots(rows=1, cols=3, subplot_titles=["MAE (↓ better)", "RMSE (↓ better)", "R² Score (↑ better)"])

        for i, metric in enumerate(["MAE", "RMSE", "R² Score"], 1):
            for row in rows:
                color = model_colors.get(row["Model"], "#a78bfa")
                fig_bar.add_trace(go.Bar(
                    x=[row["Model"]], y=[row[metric]],
                    name=row["Model"],
                    marker_color=color,
                    showlegend=(i == 1),
                ), row=1, col=i)

        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(family="Inter", color="#e2e8f0"),
            bargap=0.3,
            height=360,
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(139,92,246,0.3)", borderwidth=1, font=dict(color="white")),
        )
        fig_bar.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
        fig_bar.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Heatmap of metrics
        st.markdown('<div class="section-header">🔥 Normalized Metric Heatmap</div>', unsafe_allow_html=True)
        heatmap_data = []
        for row in rows:
            # Normalize: lower is better for MAE/RMSE (invert), higher is better for R²
            max_mae = max(r["MAE"] for r in rows)
            max_rmse = max(r["RMSE"] for r in rows)
            min_r2 = min(r["R² Score"] for r in rows)
            max_r2 = max(r["R² Score"] for r in rows)
            norm_mae = 1 - row["MAE"] / max_mae if max_mae else 0
            norm_rmse = 1 - row["RMSE"] / max_rmse if max_rmse else 0
            norm_r2 = (row["R² Score"] - min_r2) / (max_r2 - min_r2) if max_r2 != min_r2 else 1
            heatmap_data.append([norm_mae, norm_rmse, norm_r2])

        fig_hm = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=["MAE Score", "RMSE Score", "R² Score"],
            y=[r["Model"] for r in rows],
            text=[[f"{v:.3f}" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
            colorscale="Viridis",
            showscale=True,
        ))
        fig_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(family="Inter", color="#e2e8f0"),
            height=280,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown('<div class="info-box">💡 Scores normalized 0–1 where 1 = best performance. MAE & RMSE are inverted (lower error → higher score).</div>', unsafe_allow_html=True)

        # Prediction surface (2D heatmap of AQI prediction over PM2.5/NO2 space)
        st.markdown('<div class="section-header">🌐 Prediction Surface (PM2.5 × NO2)</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Select model for surface plot", list(all_preds.keys()))

        pm25_grid = np.linspace(0, 400, 60)
        no2_grid = np.linspace(0, 300, 60)
        PM, NO = np.meshgrid(pm25_grid, no2_grid)

        Z = np.zeros_like(PM)
        for i in range(PM.shape[0]):
            for j in range(PM.shape[1]):
                pts = single_prediction(models, float(PM[i, j]), float(NO[i, j]))
                Z[i, j] = pts.get(model_choice, 0)

        fig_surf = go.Figure(go.Heatmap(
            x=pm25_grid, y=no2_grid, z=Z,
            colorscale="Plasma",
            colorbar=dict(title="AQI", tickfont=dict(color="#e2e8f0")),
        ))
        fig_surf.update_layout(
            title=dict(text=f"{model_choice} — AQI over PM2.5 × NO2 space", font=dict(color="#c4b5fd", size=14)),
            xaxis=dict(title="PM2.5 (µg/m³)", color="#94a3b8"),
            yaxis=dict(title="NO2 (µg/m³)", color="#94a3b8"),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e2e8f0"),
            height=420,
        )
        st.plotly_chart(fig_surf, use_container_width=True)

    # ──── Page: Membership Functions ────────────────────────────────────────
    elif page == "🔬 Membership Functions":
        st.markdown('<div class="hero-title" style="font-size:1.8rem; text-align:center; margin-bottom:8px;">🔬 Membership Functions</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; color:#94a3b8; margin-bottom:24px;">Visualize the fuzzy membership functions for each model.</p>', unsafe_allow_html=True)

        # Manual FIS MFs
        st.markdown('<div class="section-header">🔵 Manual FIS</div>', unsafe_allow_html=True)
        fig_manual = plot_membership_functions(
            models["pm25_manual"], models["no2_manual"], "Manual FIS"
        )
        st.plotly_chart(fig_manual, use_container_width=True)

        # GA-Tuned FIS MFs
        if models["ga_data"]:
            st.markdown('<div class="section-header">🟡 GA-Tuned FIS</div>', unsafe_allow_html=True)

            gd = models["ga_data"]
            # Note: mf stored as list [type, params] in JSON (not tuple)
            pm25_ga = {k: (v[0], v[1]) for k, v in gd["pm25_mf_ga"].items()}
            no2_ga  = {k: (v[0], v[1]) for k, v in gd["no2_mf_ga"].items()}

            fig_ga = plot_membership_functions(pm25_ga, no2_ga, "GA-Tuned FIS")
            st.plotly_chart(fig_ga, use_container_width=True)

            # Diff comparison
            st.markdown('<div class="section-header">📊 Manual vs GA — Parameter Comparison</div>', unsafe_allow_html=True)

            param_rows = []
            for var_label, manual_mf, ga_mf in [("PM2.5", models["pm25_manual"], pm25_ga),
                                                   ("NO2", models["no2_manual"], no2_ga)]:
                for set_name in ["Low", "Medium", "High"]:
                    m_type, m_params = manual_mf[set_name]
                    g_type, g_params = ga_mf[set_name]
                    param_rows.append({
                        "Variable": var_label,
                        "Set": set_name,
                        "Type": m_type,
                        "Manual Params": str([round(p, 2) for p in m_params]),
                        "GA Params": str([round(p, 2) for p in g_params]),
                    })

            st.dataframe(pd.DataFrame(param_rows), use_container_width=True)

        # ANFIS learned MFs (Gaussian)
        if models["anfis"] is not None:
            st.markdown('<div class="section-header">🟢 ANFIS — Learned Gaussian MFs</div>', unsafe_allow_html=True)
            anfis = models["anfis"]
            centers = anfis.centers.detach().numpy()  # (2, 3)
            sigmas  = anfis.sigmas.detach().numpy()   # (2, 3)

            fig_anfis = make_subplots(rows=1, cols=2,
                                      subplot_titles=["PM2.5 Gaussian MFs", "NO2 Gaussian MFs"])
            colors_mf = ["#8b5cf6", "#3b82f6", "#10b981"]
            for var_i, (var_label, x_max) in enumerate([("PM2.5", 500), ("NO2", 400)]):
                x = np.linspace(0, x_max, 400)
                for k in range(3):
                    c = centers[var_i, k]
                    s = abs(sigmas[var_i, k]) + 1e-6
                    y = np.exp(-((x - c) ** 2) / (2 * s**2))
                    fig_anfis.add_trace(
                        go.Scatter(x=x, y=y,
                                   name=f"MF {k+1}",
                                   line=dict(color=colors_mf[k], width=2.5),
                                   fill="tozeroy",
                                   fillcolor=f"rgba(100,100,200,0.08)",
                                   showlegend=(var_i == 0),
                                   legendgroup=f"mf{k}"),
                        row=1, col=var_i + 1,
                    )
            fig_anfis.update_layout(
                title="ANFIS — Learned Gaussian Membership Functions",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.04)",
                font=dict(family="Inter", color="#e2e8f0"),
                legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(139,92,246,0.4)", borderwidth=1),
                height=350,
            )
            fig_anfis.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
            fig_anfis.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False, range=[0, 1.1])
            st.plotly_chart(fig_anfis, use_container_width=True)

            # ANFIS Centers & Sigmas table
            mf_rows = []
            for var_i, var_label in enumerate(["PM2.5", "NO2"]):
                for k in range(3):
                    mf_rows.append({
                        "Variable": var_label,
                        "MF": f"MF {k+1}",
                        "Center (µ)": round(float(centers[var_i, k]), 4),
                        "Sigma (σ)": round(float(abs(sigmas[var_i, k])), 4),
                    })
            st.dataframe(pd.DataFrame(mf_rows), use_container_width=True)

            # Consequent parameters
            st.markdown('<div class="section-header">🔢 ANFIS Consequent Parameters</div>', unsafe_allow_html=True)
            cons = anfis.consequents.detach().numpy()  # (9, 3)
            rule_labels = [f"Rule {i+1}" for i in range(cons.shape[0])]
            cons_df = pd.DataFrame(cons, index=rule_labels, columns=["w_PM2.5", "w_NO2", "bias"])
            cons_df = cons_df.round(6)
            st.dataframe(cons_df, use_container_width=True)

        # Info about rule base
        st.markdown('<div class="section-header">📋 Shared Rule Base</div>', unsafe_allow_html=True)
        rule_rows = []
        for i, rule in enumerate(models["rule_base"]):
            conds = " AND ".join([f"{v.upper()} is {s}" for v, s in rule["if"]])
            rule_rows.append({"Rule #": i + 1, "IF": conds, "THEN AQI =": rule["then"]})
        st.dataframe(pd.DataFrame(rule_rows), use_container_width=True)


if __name__ == "__main__":
    main()
