# ============================================================
# Fraud Detection System
#  #  Author : Sanjana R 
#  Run    : streamlit run app.py
# ============================================================

import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be very first ST call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection — Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
MODEL_PATH       = "D:/Real-Time-Fraud-Detection/outputs/fraud_detection_model.pkl"
SCALER_PATH      = "D:/Real-Time-Fraud-Detection/outputs/scaler.pkl"
SCREENSHOTS_DIR  = "screenshots"

# ─────────────────────────────────────────────
#  FEATURE ORDER  (must match training exactly)
# ─────────────────────────────────────────────
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount_Scaled", "Time_Scaled"]


# ============================================================
#  CSS  — full visual identity
# ============================================================
def inject_css() -> None:
    st.markdown(
        """
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ────────────────────────────────────────── */
:root {
    --bg0:       #06080f;
    --bg1:       #0c1120;
    --bg2:       #111827;
    --bg3:       #162033;
    --border:    #1c2d45;
    --border-hi: #2a4060;
    --cyan:      #0df0d4;
    --cyan-dim:  #08a890;
    --blue:      #3b82f6;
    --green:     #22c55e;
    --red:       #ef4444;
    --amber:     #f59e0b;
    --text:      #e2eaf5;
    --muted:     #5a7090;
    --r:         12px;
    --r-sm:      7px;
    --shadow-cyan: 0 0 28px rgba(13,240,212,.18);
    --shadow-card: 0 4px 24px rgba(0,0,0,.45);
}

/* ── Base ─────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg0) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #08101e !important;
    border-right: 1px solid var(--border);

    min-width: 230px;
    max-width: 230px;
}
.block-container {
    max-width: 1350px !important;
    margin: auto !important;

    padding-top: 1rem !important;
    padding-bottom: 2rem !important;

    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }

/* ── Hero ─────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(130deg, #08111f 0%, #0c1a2e 60%, #0d1f30 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--cyan);
    border-radius: var(--r);
    padding: 2.6rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.hero::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(13,240,212,.10) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(13,240,212,.08);
    border: 1px solid rgba(13,240,212,.25);
    color: var(--cyan); border-radius: 20px;
    padding: 3px 14px; font-size: .72rem;
    font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: .9rem;
}
.hero-title {
    font-size: 2.5rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(90deg, var(--cyan), #5eead4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 .45rem;
}
.hero-sub { color: var(--muted); font-size: 1rem; font-weight: 300; margin: 0; }

/* ── KPI card ─────────────────────────────────────────────── */
.kpi {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.35rem 1.5rem;
    text-align: center;
    transition: border-color .25s, transform .25s, box-shadow .25s;
    box-shadow: var(--shadow-card);
}
.kpi:hover {
    border-color: rgba(13,240,212,.35);
    transform: translateY(-3px);
    box-shadow: var(--shadow-cyan);
}
.kpi-val {
    font-size: 2rem; font-weight: 700;
    color: var(--cyan); display: block; line-height: 1.1;
}
.kpi-lbl {
    font-size: .72rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: .1em; margin-top: 5px;
}
.kpi-icon { font-size: 1.5rem; display: block; margin-bottom: 6px; }

/* ── Card wrapper ─────────────────────────────────────────── */
.card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-card);
}
.card-title {
    font-size: 1rem; font-weight: 700;
    color: var(--text); margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 8px;
}
.card-title-bar {
    flex: 1; height: 1px; background: var(--border);
}

/* ── Section separator ────────────────────────────────────── */
.sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-hi), var(--cyan-dim), var(--border-hi), transparent);
    margin: 2rem 0; border: none;
}

/* ── Inputs ───────────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
    background: #0a1220 !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .9rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(13,240,212,.12) !important;
}
/* Slider track */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--cyan) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #7a95b8 !important;
    font-size: .78rem !important;
    font-weight: 600 !important;
    letter-spacing: .06em !important;
    text-transform: uppercase !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Primary button ───────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, var(--cyan), var(--cyan-dim)) !important;
    color: #03111e !important; border: none !important;
    border-radius: var(--r-sm) !important;
    padding: .72rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important; font-size: .92rem !important;
    letter-spacing: .06em !important; text-transform: uppercase !important;
    width: 100% !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(13,240,212,.38) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result boxes ─────────────────────────────────────────── */
.result-fraud {
    background: linear-gradient(135deg, rgba(239,68,68,.13), rgba(239,68,68,.04));
    border: 2px solid var(--red);
    border-radius: var(--r); padding: 2rem 2.5rem; text-align: center;
    box-shadow: 0 0 32px rgba(239,68,68,.20);
    animation: pulseRed 2.2s ease-in-out infinite;
}
@keyframes pulseRed {
    0%,100% { box-shadow: 0 0 20px rgba(239,68,68,.18); }
    50%     { box-shadow: 0 0 44px rgba(239,68,68,.40); }
}
.result-normal {
    background: linear-gradient(135deg, rgba(34,197,94,.1), rgba(34,197,94,.03));
    border: 2px solid var(--green);
    border-radius: var(--r); padding: 2rem 2.5rem; text-align: center;
    box-shadow: 0 0 28px rgba(34,197,94,.18);
}
.result-icon    { font-size: 3.2rem; display: block; margin-bottom: .5rem; }
.result-verdict {
    font-size: 2rem; font-weight: 800;
    font-family: 'Outfit', sans-serif; margin: .3rem 0;
}
.result-caption { color: var(--muted); font-size: .9rem; margin-top: .5rem; }

/* ── Risk badges ──────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 5px 18px; border-radius: 20px;
    font-weight: 700; font-size: .8rem;
    letter-spacing: .1em; text-transform: uppercase;
}
.badge-high   { background:rgba(239,68,68,.15);  border:1px solid var(--red);   color:var(--red);   }
.badge-med    { background:rgba(245,158,11,.12); border:1px solid var(--amber); color:var(--amber); }
.badge-low    { background:rgba(245,158,11,.08); border:1px solid #b07a00;      color:#f0b840;      }
.badge-vlow   { background:rgba(34,197,94,.1);   border:1px solid var(--green); color:var(--green); }

/* ── Probability strip ────────────────────────────────────── */
.prob-bar-bg {
    width: 100%; height: 10px; background: var(--bg3);
    border-radius: 5px; overflow: hidden; margin-top: 6px;
}
.prob-bar-fill { height: 100%; border-radius: 5px; transition: width .8s ease; }

/* ── Mini metric tile ─────────────────────────────────────── */
.mtile {
    background: #0a1220;
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 1rem 1.2rem; text-align: center;
}
.mtile-val { font-size: 1.5rem; font-weight: 700; font-family: 'Outfit', sans-serif; }
.mtile-lbl { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-top: 3px; }

/* ── Table ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: var(--r) !important; overflow: hidden; }
[data-testid="stDataFrame"] thead tr th {
    background: var(--bg2) !important; color: var(--cyan) !important;
    font-family: 'Outfit', sans-serif !important; font-size: .72rem !important;
    text-transform: uppercase !important; letter-spacing: .09em !important;
}
[data-testid="stDataFrame"] tbody tr:hover td { background: #131f32 !important; }

/* ── Sidebar ──────────────────────────────────────────────── */
.sb-logo {
    text-align: center;
    padding: 1.5rem 1rem 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.4rem;
}
.sb-logo-icon { font-size: 2.8rem; display: block; }
.sb-logo-name {
    font-size: 1.25rem; font-weight: 800;
    background: linear-gradient(90deg, var(--cyan), #5eead4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-logo-tag  { font-size: .68rem; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; margin-top: 2px; }

/* Status dot */
.dot {
    display: inline-block; width: 8px; height: 8px;
    border-radius: 50%; margin-right: 6px;
    animation: blink 1.6s ease-in-out infinite;
}
.dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot-amber { background: var(--amber); box-shadow: 0 0 6px var(--amber); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Info alert ───────────────────────────────────────────── */
.info-bar {
    background: rgba(13,240,212,.06);
    border-left: 3px solid var(--cyan);
    border-radius: 0 var(--r-sm) var(--r-sm) 0;
    padding: .8rem 1.2rem;
    font-size: .85rem; color: #9ab8cc;
    margin-bottom: 1.6rem;
}

/* ── Footer ───────────────────────────────────────────────── */
.footer {
    background: var(--bg2); border: 1px solid var(--border);
    border-top: 2px solid var(--cyan-dim);
    border-radius: var(--r); padding: 1.4rem 2rem;
    margin-top: 3rem;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: .6rem;
}
.footer-brand { font-weight: 700; color: var(--cyan); font-size: .95rem; }
.footer-txt   { font-size: .78rem; color: var(--muted); }

/* ── Pipeline step ────────────────────────────────────────── */
.pipeline-step {
    background: var(--bg3);
    border: 1px solid var(--border); border-radius: var(--r-sm);
    padding: .9rem 1rem; margin-bottom: .5rem;
    display: flex; align-items: flex-start; gap: 10px;
}
.pipeline-step-icon { font-size: 1.3rem; flex-shrink: 0; margin-top: 1px; }
.pipeline-step-title { font-weight: 700; font-size: .88rem; color: var(--text); }
.pipeline-step-desc  { font-size: .78rem; color: var(--muted); margin-top: 2px; line-height: 1.45; }

/* ── About tech row ───────────────────────────────────────── */
.tech-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid var(--border);
}
.tech-name  { font-weight: 600; font-size: .88rem; min-width: 150px; color: var(--text); }
.tech-desc  { font-size: .82rem; color: var(--muted); }

/* ── Plotly overrides ─────────────────────────────────────── */
.js-plotly-plot .plotly,
.js-plotly-plot .plotly .svg-container { background: transparent !important; }
</style>
""",
        unsafe_allow_html=True,
    )


# ============================================================
#  MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load trained RF model + StandardScaler.  Returns (model, scaler, loaded_ok)."""
    model, scaler, ok = None, None, False
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        ok = model is not None
    except Exception as exc:
        st.warning(f"⚠️ Could not load model artifacts: {exc}")
    return model, scaler, ok


# ============================================================
#  PREDICTION ENGINE
# ============================================================
def build_feature_vector(amount: float, time_val: float, v_vals: dict) -> pd.DataFrame:
    """
    Construct the 30-column feature DataFrame that matches training layout:
    V1..V28, Amount_Scaled, Time_Scaled
    """
    row = {col: 0.0 for col in FEATURE_COLS}
    # V-features supplied by user
    for k, v in v_vals.items():
        if k in row:
            row[k] = v
    # Amount_Scaled / Time_Scaled are pre-scaled by the caller
    row["Amount_Scaled"] = amount
    row["Time_Scaled"]   = time_val
    return pd.DataFrame([row], columns=FEATURE_COLS)


def scale_value(scaler, raw_value: float, feature: str) -> float:
    """Use the scaler to transform a single feature if scaler is available."""
    if scaler is None:
        return raw_value
    try:
        # Scaler was fit on [Amount, Time] in that order
        idx = {"Amount": 0, "Time": 1}.get(feature, None)
        if idx is None:
            return raw_value
        val_arr = np.zeros((1, 2))
        val_arr[0, idx] = raw_value
        # We only need the transformed value for the target column
        scaled = scaler.transform(val_arr)
        return float(scaled[0, idx])
    except Exception:
        return raw_value


def run_prediction(model, scaler, amount_raw: float, v_vals: dict) -> dict:
    """
    Run fraud prediction.
    Returns dict: prediction, fraud_prob, normal_prob, risk, risk_class
    """
    # Scale Amount & Time (use 0 for Time as it's less critical in user input)
    amount_scaled = scale_value(scaler, amount_raw, "Amount")
    time_scaled   = scale_value(scaler, 0.0, "Time")

    df_input = build_feature_vector(amount_scaled, time_scaled, v_vals)

    proba = model.predict_proba(df_input)[0]

    fraud_prob = round(float(proba[1]) * 100, 2)
    normal_prob = round(float(proba[0]) * 100, 2)

    # Custom fraud threshold
    pred = 1 if fraud_prob >= 20 else 0

    if fraud_prob >= 80:
        risk, rc = "HIGH", "high"
    elif fraud_prob >= 50:
        risk, rc = "MEDIUM", "med"
    elif fraud_prob >= 20:
        risk, rc = "LOW", "low"
    else:
        risk, rc = "VERY LOW", "vlow"

    return {
        "prediction": pred,
        "fraud_prob": fraud_prob,
        "normal_prob": normal_prob,
        "risk": risk,
        "risk_class": rc,
        "is_fraud": pred == 1,
    }


def demo_predict(amount_raw: float, v_vals: dict) -> dict:
    """
    Rule-based fallback when no model is loaded (demo / showcase mode).
    Mirrors the real prediction interface exactly.
    """
    score = 0
    score += max(0, -v_vals.get("V14", 0)) * 5
    score += max(0, -v_vals.get("V12", 0)) * 4
    score += max(0, -v_vals.get("V11", 0)) * 3
    score += max(0, -v_vals.get("V17", 0)) * 3
    score += max(0, -v_vals.get("V10", 0)) * 3
    score += max(0,  v_vals.get("V4",  0)) * 2
    if amount_raw > 1500:
        score += 10

    fraud_prob  = min(round(float(score), 2), 99.0)
    normal_prob = round(100.0 - fraud_prob, 2)
    pred        = 1 if fraud_prob >= 50 else 0

    if fraud_prob >= 80:   risk, rc = "HIGH",     "high"
    elif fraud_prob >= 50: risk, rc = "MEDIUM",   "med"
    elif fraud_prob >= 20: risk, rc = "LOW",       "low"
    else:                  risk, rc = "VERY LOW", "vlow"

    return {
        "prediction":  pred,
        "fraud_prob":  fraud_prob,
        "normal_prob": normal_prob,
        "risk":        risk,
        "risk_class":  rc,
        "is_fraud":    pred == 1,
    }


# ============================================================
#  SHARED UI HELPERS
# ============================================================
def render_hero(title: str, subtitle: str, tag: str = "🛡️ AI-Powered Fintech Security") -> None:
    st.markdown(
        f"""
<div class="hero">
  <span class="hero-tag">{tag}</span>
  <div class="hero-title">{title}</div>
  <p class="hero-sub">{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_kpi(value: str, label: str, icon: str) -> None:
    st.markdown(
        f"""
<div class="kpi">
  <span class="kpi-icon">{icon}</span>
  <span class="kpi-val">{value}</span>
  <div class="kpi-lbl">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sep() -> None:
    st.markdown('<div class="sep"></div>', unsafe_allow_html=True)


def render_card_open(title: str, icon: str = "") -> None:
    st.markdown(
        f"""
<div class="card">
<div class="card-title">{icon} {title} <div class="card-title-bar"></div></div>
""",
        unsafe_allow_html=True,
    )


def render_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def render_mtile(value: str, label: str, color: str = "var(--cyan)") -> None:
    st.markdown(
        f"""
<div class="mtile">
  <div class="mtile-val" style="color:{color};">{value}</div>
  <div class="mtile-lbl">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_prob_bar(pct: float, color: str) -> None:
    st.markdown(
        f"""
<div class="prob-bar-bg">
  <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    from datetime import datetime
    year = datetime.now().year
    st.markdown(
        f"""
<div class="footer">
  <div>
    <div class="footer-brand">🛡️ FraudDtection AI</div>
    <div class="footer-txt"> Credit Card Fraud Detection</div>
  </div>
  <div style="text-align:center;">
    <div class="footer-txt">Python · Streamlit · Scikit-learn · Random Forest</div>
    <div class="footer-txt">Sanjana R · </div>
  </div>
  <div style="text-align:right;">
    <div class="footer-txt"><span class="dot dot-green"></span>All systems operational</div>
    <div class="footer-txt">Dataset: Kaggle CC Fraud Detection</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
#  SIDEBAR
# ============================================================
def render_sidebar(model_loaded: bool) -> str:
    with st.sidebar:
        # Logo
        st.markdown(
            """
<div class="sb-logo">
  <span class="sb-logo-icon">🛡️</span>
  <div class="sb-logo-name">FD</div>
  <div class="sb-logo-tag">Detection Engine </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # Navigation
        st.markdown(
            "<p style='color:var(--muted);font-size:.68rem;text-transform:uppercase;"
            "letter-spacing:.12em;margin-bottom:6px;'>Navigation</p>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            label="nav",
            options=["🏠  Home", "🔍  Fraud Detection", "📊  Model Insights", "📖  About Project"],
            label_visibility="collapsed",
        )

        render_sep()

        # Model status box
        if model_loaded:
            dot_cls, status_txt, stat_color = "dot-green", "Model Online",  "var(--green)"
        else:
            dot_cls, status_txt, stat_color = "dot-amber", "Demo Mode",     "var(--amber)"

        if "scan_count" not in st.session_state:
            st.session_state.scan_count = 0

        st.markdown(
            f"""
<div style="background:#07101e;border:1px solid var(--border);
            border-radius:var(--r-sm);padding:1rem;margin-bottom:.9rem;">
  <div style="font-size:.68rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:.1em;margin-bottom:7px;">System Status</div>
  <div style="display:flex;align-items:center;gap:7px;">
    <span class="dot {dot_cls}"></span>
    <span style="color:{stat_color};font-size:.85rem;font-weight:600;">{status_txt}</span>
  </div>
  <div style="color:var(--muted);font-size:.74rem;margin-top:6px;line-height:1.7;">
    Algorithm: Random Forest<br>
    Features: V4 V10 V11 V12 V14 V17
  </div>
</div>

<div style="background:#07101e;border:1px solid var(--border);
            border-radius:var(--r-sm);padding:1rem;">
  <div style="font-size:.68rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:.1em;margin-bottom:5px;">Session Stats</div>
  <div style="color:var(--cyan);font-size:1.6rem;font-weight:700;line-height:1;">
    {st.session_state.scan_count}
  </div>
  <div style="color:var(--muted);font-size:.74rem;">Transactions scanned</div>
</div>
""",
            unsafe_allow_html=True,
        )

    return page


# ============================================================
#  PAGE: HOME
# ============================================================
def page_home() -> None:
    render_hero(
        "Fraud-Detection",
        "Credit Card Fraud Detection Engine · Random Forest",
        "Fraud Prevention"
            )

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("99.87%", "Accuracy", "🎯"),
        ("58.78%", "Precision", "📊"),
        ("81.05%", "Recall", "📡"),
        ("97.52%", "ROC-AUC", "📈"),
    ]
    for col, (val, lbl, icon) in zip([c1, c2, c3, c4], kpis):
        with col:
            render_kpi(val, lbl, icon)

    render_sep()

    # Dataset overview + why RF
    left, right = st.columns(2)

    with left:
        render_card_open("📦 Dataset Overview", "")
        st.markdown(
            """
<div style="color:#8aaccc;font-size:.9rem;line-height:1.9;">
  <b style="color:var(--cyan);">Kaggle Credit Card Fraud Detection</b><br><br>
  📊 <b style="color:var(--text);">284,807</b> total transactions<br>
  🚨 <b style="color:var(--red);">492</b> fraud cases (0.17 %)<br>
  🧬 <b style="color:var(--text);">30</b> features (V1–V28 + Amount + Time)<br>
  🔒 PCA-anonymised features for privacy<br>
  📅 Sept 2013 · European cardholders<br>
  ⚖️ Class imbalance handled via <b style="color:var(--cyan);">SMOTE</b>
</div>
""",
            unsafe_allow_html=True,
        )
        render_card_close()

    with right:
        render_card_open("🌲 Why Random Forest?", "")
        rows = [
            ("🌳", "Ensemble Method",   "Combines 100s of trees → higher accuracy"),
            ("⚖️", "Handles Imbalance", "class_weight='balanced' penalises missed fraud"),
            ("📊", "Feature Importance","Ranks which features signal fraud most"),
            ("🔒", "Robustness",        "Resistant to noise & outliers"),
            ("🧐", "Interpretable",     "Explainable to business stakeholders"),
        ]
        for icon, name, desc in rows:
            st.markdown(
                f"""
<div class="tech-row">
  <span style="font-size:1.1rem;width:26px;">{icon}</span>
  <span class="tech-name">{name}</span>
  <span class="tech-desc">{desc}</span>
</div>
""",
                unsafe_allow_html=True,
            )
        render_card_close()

    render_sep()

    # ML Pipeline steps
    render_card_open("🔧 ML Pipeline", "")
    steps = [
        ("📥", "Data Loading",        "284,807 credit card transactions from Kaggle"),
        ("🔍", "EDA",                 "5 visualisations — class distribution, amount, time, heatmap, correlation"),
        ("⚙️", "Feature Scaling",     "StandardScaler on Amount & Time to match V1–V28 range"),
        ("✂️", "Train-Test Split",    "80/20 stratified split — preserves 0.17 % fraud ratio"),
        ("⚖️", "SMOTE Balancing",     "Synthetic minority oversampling on training set only"),
        ("🌲", "Model Training",      "RandomForestClassifier · 100 estimators · class_weight='balanced'"),
        ("📊", "Evaluation",          "Accuracy · Precision · Recall · F1-Score · ROC-AUC"),
        ("🎯", "Prediction System",   "Real-time fraud detection with confidence scores & risk levels"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f"""
<div class="pipeline-step">
  <div class="pipeline-step-icon">{icon}</div>
  <div>
    <div class="pipeline-step-title">{title}</div>
    <div class="pipeline-step-desc">{desc}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
    render_card_close()


# ============================================================
#  PAGE: FRAUD DETECTION
# ============================================================
def page_fraud_detection(model, scaler, model_loaded: bool) -> None:
    render_hero(
        "🔍 Fraud Detection",
        "Enter transaction details for an instant AI-powered risk assessment.",
        tag="⚡ Real-Time Inference",
    )

    # Status bar
    stat_color = "var(--green)" if model_loaded else "var(--amber)"
    stat_txt   = "Model Online — Random Forest" if model_loaded else "Demo Mode — rule-based fallback (no model file found)"
    st.markdown(
        f"""
<div class="info-bar">
  <span class="dot {"dot-green" if model_loaded else "dot-amber"}"></span>
  <span style="color:{stat_color};font-weight:600;">{stat_txt}</span>
  &nbsp;·&nbsp; Model path: <code style="background:#0d1a2c;padding:1px 6px;
  border-radius:4px;font-size:.8rem;">{MODEL_PATH}</code>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Input form ────────────────────────────────────────────
    with st.form("fraud_form", clear_on_submit=False):
        render_card_open("💳 Transaction Input", "")

        # Row 1 — Amount + V14 + V10
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            amount = st.number_input(
                "Transaction Amount (€)",
                min_value=0.01,
                max_value=50_000.0,
                value=st.session_state.get("amount", 250.0),
                step=0.01,
                help="Raw transaction amount before scaling.",
            )
        with col_b:
            v14 = st.slider(
                "V14  (highest fraud signal)",
                min_value=-20.0, max_value=10.0,
                value=st.session_state.get("v14", 0.0),
                help="V14 has the strongest negative correlation with fraud."
            )
        with col_c:
            v10 = st.slider(
                "V10",
                min_value=-20.0, max_value=15.0,
                value=st.session_state.get("v10", 0.0)
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2 — V17 + V12 + V4 + V11
        col_d, col_e, col_f, col_g = st.columns(4)

        with col_d:
            v17 = st.number_input(
                "V17",
                value=st.session_state.get("v17", 0.0),
                step=0.0001,
                format="%.4f"
            )
        with col_e:
            v12 = st.number_input(
                "V12",
                value=st.session_state.get("v12", 0.0),
                step=0.0001,
                format="%.4f"
            )
        with col_f:
            v4  = st.number_input(
                "V4",
                value=st.session_state.get("v4", 0.0),
                step=0.0001,
                format="%.4f"
            )
        with col_g:
            v11 = st.number_input(
                "V11",
                value=st.session_state.get("v11", 0.0),
                step=0.0001,
                format="%.4f"
            )

        render_card_close()

        submitted = st.form_submit_button(
            "🔍 ANALYZE TRANSACTION",
            use_container_width=True
        )

        
    # ── Quick-fill examples (outside form) ───────────────────
    st.markdown(
        "<p style='color:var(--muted);font-size:.8rem;margin:.6rem 0 .4rem;'>"
        "Quick-fill examples →</p>",
        unsafe_allow_html=True,
    )

    qa, qb, qc = st.columns(3)

    # Fraud Sample
    if qa.button("🚨 Load Fraud Sample", use_container_width=True):
        st.session_state["amount"] = 25000.0
        st.session_state["v14"] = -18.0
        st.session_state["v10"] = -15.0
        st.session_state["v17"] = -17.0
        st.session_state["v12"] = -14.0
        st.session_state["v4"] = 10.0
        st.session_state["v11"] = -12.0
        st.rerun()

    # Normal Sample
    if qb.button("✅ Load Normal Sample", use_container_width=True):
        st.session_state["amount"] = 120.0
        st.session_state["v14"] = 0.2
        st.session_state["v10"] = 0.1
        st.session_state["v17"] = 0.0
        st.session_state["v12"] = 0.0
        st.session_state["v4"] = 1.0
        st.session_state["v11"] = -0.2
        st.rerun()

    # Random Sample
    if qc.button("🔀 Random Sample", use_container_width=True):
        import random

        if random.random() > 0.5:
            st.session_state["amount"] = 18000.0
            st.session_state["v14"] = -16.0
            st.session_state["v10"] = -14.0
            st.session_state["v17"] = -15.0
            st.session_state["v12"] = -13.0
            st.session_state["v4"] = 8.0
            st.session_state["v11"] = -10.0
        else:
            st.session_state["amount"] = 80.0
            st.session_state["v14"] = 0.4
            st.session_state["v10"] = 0.1
            st.session_state["v17"] = 0.0
            st.session_state["v12"] = 0.0
            st.session_state["v4"] = 1.0
            st.session_state["v11"] = -0.1

        st.rerun()

    # ── Prediction ────────────────────────────────────────────
    if submitted:
        v_vals = {"V4": v4, "V10": v10, "V11": v11, "V12": v12, "V14": v14, "V17": v17}

        with st.spinner("🔄 Analysing transaction…"):
            time.sleep(0.35)          # brief UX delay
            if model_loaded:
                result = run_prediction(model, scaler, amount, v_vals)
            else:
                result = demo_predict(amount, v_vals)

        st.session_state.scan_count = st.session_state.get("scan_count", 0) + 1
        render_sep()

        # ── Result + gauge row ────────────────────────────────
        col_res, col_gauge = st.columns([3, 2])

        with col_res:
            if result["fraud_prob"] >= 20:
                verdict_html = f"""
<div class="result-fraud">
  <span class="result-icon">🚨</span>
  <div class="result-verdict" style="color:var(--red);">FRAUDULENT TRANSACTION</div>
  <div class="result-caption">This transaction has been flagged as potentially fraudulent.</div>
  <br>
  <span class="badge badge-{result['risk_class']}">⚡ {result['risk']} RISK</span>
</div>"""
            else:
                verdict_html = f"""
<div class="result-normal">
  <span class="result-icon">✅</span>
  <div class="result-verdict" style="color:var(--green);">LEGITIMATE TRANSACTION</div>
  <div class="result-caption">This transaction appears to be normal and safe to process.</div>
  <br>
  <span class="badge badge-vlow">🟢 {result['risk']} RISK</span>
</div>"""
            st.markdown(verdict_html, unsafe_allow_html=True)

        with col_gauge:
            # Inline SVG gauge — no Plotly dependency
            pct   = result["fraud_prob"]
            ang   = -90 + (pct / 100) * 180   # −90 → +90
            rad   = ang * 3.14159 / 180
            gx    = round(100 + 80 * __import__("math").cos(rad), 2)
            gy    = round(100 - 80 * __import__("math").sin(rad) + 20, 2)

            if pct >= 80:   g_col = "#ef4444"
            elif pct >= 50: g_col = "#f59e0b"
            elif pct >= 20: g_col = "#f0b840"
            else:           g_col = "#22c55e"

            st.markdown(
                f"""
<div class="card" style="text-align:center;padding:1.5rem;">
  <div style="font-size:.72rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:.1em;margin-bottom:.5rem;">Fraud Risk Score</div>
  <svg viewBox="0 30 200 120" width="100%" style="max-width:220px;display:block;margin:0 auto;">
    <!-- background arc -->
    <path d="M 20 120 A 80 80 0 0 1 180 120" fill="none"
          stroke="#1c2d45" stroke-width="14" stroke-linecap="round"/>
    <!-- coloured zones -->
    <path d="M 20 120 A 80 80 0 0 1 57 57" fill="none"
          stroke="rgba(34,197,94,.35)" stroke-width="14" stroke-linecap="butt"/>
    <path d="M 57 57  A 80 80 0 0 1 100 40" fill="none"
          stroke="rgba(240,184,64,.35)" stroke-width="14" stroke-linecap="butt"/>
    <path d="M 100 40 A 80 80 0 0 1 143 57" fill="none"
          stroke="rgba(245,158,11,.45)" stroke-width="14" stroke-linecap="butt"/>
    <path d="M 143 57 A 80 80 0 0 1 180 120" fill="none"
          stroke="rgba(239,68,68,.45)" stroke-width="14" stroke-linecap="butt"/>
    <!-- needle -->
    <line x1="100" y1="120" x2="{gx}" y2="{gy}"
          stroke="{g_col}" stroke-width="3" stroke-linecap="round"
          style="filter:drop-shadow(0 0 4px {g_col});"/>
    <circle cx="100" cy="120" r="6" fill="{g_col}"/>
    <!-- label -->
    <text x="100" y="105" text-anchor="middle" fill="{g_col}"
          font-size="22" font-weight="700" font-family="Outfit,sans-serif">{pct:.1f}%</text>
    <text x="100" y="133" text-anchor="middle" fill="#5a7090"
          font-size="9" font-family="Outfit,sans-serif">FRAUD PROBABILITY</text>
  </svg>
</div>
""",
                unsafe_allow_html=True,
            )

        # ── Probability bars + metric tiles ───────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)

        fraud_c  = "var(--red)"   if result["is_fraud"] else "var(--green)"
        normal_c = "var(--green)" if not result["is_fraud"] else "var(--red)"
        risk_colors = {"HIGH": "var(--red)", "MEDIUM": "var(--amber)",
                       "LOW": "#f0b840",     "VERY LOW": "var(--green)"}

        with mc1:
            render_mtile(f"{result['fraud_prob']:.1f}%",  "Fraud Probability",  fraud_c)
            render_prob_bar(result["fraud_prob"], fraud_c)

        with mc2:
            render_mtile(f"{result['normal_prob']:.1f}%", "Normal Probability", normal_c)
            render_prob_bar(result["normal_prob"], normal_c)

        with mc3:
            render_mtile(result["risk"], "Risk Level", risk_colors[result["risk"]])

        with mc4:
            icon_v = "🚨 FRAUD" if result["is_fraud"] else "✅ NORMAL"
            vc     = "var(--red)" if result["is_fraud"] else "var(--green)"
            render_mtile(icon_v, "Verdict", vc)

        # ── Transaction summary table ─────────────────────────
        render_sep()
        st.markdown(
            "<div class='card-title'>📋 Transaction Summary "
            "<div class='card-title-bar'></div></div>",
            unsafe_allow_html=True,
        )
        from datetime import datetime
        summary_df = pd.DataFrame(
            {
                "Field":  ["Amount (€)", "V4",    "V10",   "V11",   "V12",   "V14",   "V17",   "Timestamp"],
                "Value":  [f"€{amount:.2f}", f"{v4:.4f}", f"{v10:.4f}",
                           f"{v11:.4f}", f"{v12:.4f}", f"{v14:.4f}", f"{v17:.4f}",
                           datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                "Note":   ["Raw input", "PCA feat", "PCA feat", "PCA feat",
                           "PCA feat",  "Strongest fraud signal", "PCA feat", "Auto"],
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ============================================================
#  PAGE: MODEL INSIGHTS
# ============================================================
def page_model_insights() -> None:
    render_hero(
        "📊 Model Insights",
        "Performance metrics, confusion matrix, ROC curve, and feature importance.",
        tag="🔬 Model Analysis",
    )

    # Top metric row
    metrics = [
        ("99.87%", "Accuracy",  "var(--cyan)"),
        ("58.78%",  "Precision", "var(--green)"),
        ("81.05%",  "Recall",    "var(--green)"),
        ("68.14%",  "F1-Score",  "var(--cyan)"),
        ("97.52%",  "ROC-AUC",   "var(--cyan)"),
        ("100",    "Estimators","var(--muted)"),
    ]
    for col, (val, lbl, color) in zip(st.columns(6), metrics):
        with col:
            render_mtile(val, lbl, color)

    render_sep()

    # Screenshot images
    def _img(filename: str, caption: str) -> None:
        path = os.path.join(SCREENSHOTS_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.markdown(
                f"""
<div style="background:var(--bg3);border:1px dashed var(--border-hi);
            border-radius:var(--r);padding:2.5rem;text-align:center;color:var(--muted);">
  📂 Place <code style="color:var(--cyan);">{filename}</code>
  inside the <code>screenshots/</code> folder to display this chart.
</div>
""",
                unsafe_allow_html=True,
            )

    col_l, col_r = st.columns(2)

    with col_l:
        render_card_open("🎯 Confusion Matrix & ROC Curve", "")
        _img("06_confusion_matrix_roc.png", "Confusion Matrix & ROC Curve")
        render_card_close()

    with col_r:
        render_card_open("🏆 Feature Importance", "")
        _img("07_feature_importance.png", "Top 20 Features — Random Forest")
        render_card_close()

    # Confusion matrix interpretation
    render_sep()
    render_card_open("🔍 Confusion Matrix Explained", "")
    ci1, ci2, ci3, ci4 = st.columns(4)
    cm_boxes = [
        ("var(--cyan)",  "56,846", "True Negatives",  "Normal correctly identified"),
        ("var(--amber)", "16",     "False Positives",  "Normal flagged as Fraud (false alarms)"),
        ("var(--red)",   "8",      "False Negatives",  "Fraud missed by model ← minimise this"),
        ("var(--green)", "90",     "True Positives",   "Fraud correctly caught ✔"),
    ]
    for col, (color, val, title, desc) in zip([ci1, ci2, ci3, ci4], cm_boxes):
        with col:
            st.markdown(
                f"""
<div class="mtile" style="border-top:2px solid {color};">
  <div class="mtile-val" style="color:{color};">{val}</div>
  <div style="font-weight:700;font-size:.83rem;color:var(--text);margin:4px 0;">{title}</div>
  <div class="mtile-lbl">{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )
    render_card_close()


# ============================================================
#  PAGE: ABOUT PROJECT
# ============================================================
def page_about() -> None:
    render_hero(
        "📖 About the Project",
        "Fraud Detection Engine ",
        tag="🎓  Project",
    )

    col_l, col_r = st.columns([3, 2])

    with col_l:
        render_card_open("🎯 Project Objective", "")
        st.markdown(
            """
<p style="color:#8aaccc;line-height:1.85;font-size:.93rem;">
  This project builds an end-to-end machine learning pipeline to detect
  fraudulent credit card transactions in <b style="color:var(--cyan);">real time</b>.
  Credit card fraud costs businesses billions annually. The system identifies fraud
  with high <b>Recall</b> (catching as many real frauds as possible) while controlling
  false positives, making it suitable for production deployment at financial institutions.
</p>
""",
            unsafe_allow_html=True,
        )
        render_card_close()

        render_card_open("⚙️ ML Workflow", "")
        pipeline = [
            ("📥", "Data Ingestion",     "Load 284,807 labelled transactions from Kaggle."),
            ("🔍", "EDA",                "Class distribution, amount/time analysis, correlation heatmaps."),
            ("⚙️", "Preprocessing",      "StandardScaler on Amount & Time; V1–V28 already PCA-scaled."),
            ("✂️", "Stratified Split",   "80/20 split — stratify=y keeps fraud ratio intact in both sets."),
            ("⚖️", "SMOTE",              "Applied on training set ONLY — generates synthetic fraud rows."),
            ("🌲", "Random Forest",      "100 estimators, max_depth=10, class_weight='balanced', n_jobs=-1."),
            ("📊", "Evaluation",         "Accuracy · Precision · Recall · F1 · ROC-AUC · Confusion Matrix."),
            ("💾", "Persistence",        "joblib.dump() → fraud_detection_model.pkl + scaler.pkl."),
        ]
        for icon, title, desc in pipeline:
            st.markdown(
                f"""
<div class="pipeline-step">
  <div class="pipeline-step-icon">{icon}</div>
  <div>
    <div class="pipeline-step-title">{title}</div>
    <div class="pipeline-step-desc">{desc}</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        render_card_close()

        render_card_open("💡 SMOTE Explained", "")
        st.markdown(
            """
<p style="color:#8aaccc;line-height:1.85;font-size:.9rem;">
  <b style="color:var(--cyan);">SMOTE</b> (Synthetic Minority Over-sampling Technique)
  addresses the extreme class imbalance (99.83 % normal vs 0.17 % fraud) by creating
  <i>synthetic</i> fraud samples rather than simply duplicating existing ones.<br><br>
  For each minority sample it finds its k-nearest neighbours and interpolates new points
  along the connecting lines — giving the model more fraud examples to learn from without
  the risk of overfitting to repeated identical rows.<br><br>
  <b style="color:var(--amber);">Critical rule:</b> SMOTE is applied
  <b>only on the training set</b> — never on validation or test data, which must
  reflect the real-world distribution.
</p>
""",
            unsafe_allow_html=True,
        )
        render_card_close()

    with col_r:
        render_card_open("🛠️ Tech Stack", "")
        tech = [
            ("🐍", "Python 3.10+",       "Core language"),
            ("🧠", "Scikit-Learn",        "RF · Scaler · Metrics"),
            ("⚖️", "Imbalanced-Learn",   "SMOTE oversampling"),
            ("📊", "Pandas / NumPy",     "Data wrangling"),
            ("📉", "Matplotlib/Seaborn", "Static charts"),
            ("🌐", "Streamlit",          "Web dashboard"),
            ("💾", "Joblib",             "Model persistence"),
        ]
        for icon, name, desc in tech:
            st.markdown(
                f"""
<div class="tech-row">
  <span style="font-size:1.1rem;width:26px;">{icon}</span>
  <span class="tech-name">{name}</span>
  <span class="tech-desc">{desc}</span>
</div>
""",
                unsafe_allow_html=True,
            )
        render_card_close()

        render_card_open("🏆 Model Performance", "")
        st.markdown(
            """
<div style="color:#8aaccc;font-size:.9rem;line-height:2.1;">
  ✅ Accuracy  &nbsp;&nbsp; <b style="color:var(--cyan);">99.87 %</b><br>
  🎯 Precision &nbsp; <b style="color:var(--green);">58.58 %</b><br>
  📡 Recall    &nbsp;&nbsp;&nbsp; <b style="color:var(--green);">81.05 %</b><br>
  📊 F1-Score  &nbsp; <b style="color:var(--green);">68.14 %</b><br>
  📈 ROC-AUC   &nbsp; <b style="color:var(--cyan);">97.52 %</b><br>
  🌳 Estimators &nbsp;<b style="color:var(--text);">100 trees</b>
</div>
""",
            unsafe_allow_html=True,
        )
        render_card_close()

        render_card_open("🔮 Future Enhancements", "")
        ideas = [
            "XGBoost / LightGBM comparison",
            "GridSearchCV hyperparameter tuning",
            "SHAP explainability (per-prediction)",
            "Threshold optimisation (precision/recall trade-off)",
            "Flask REST API for microservice deployment",
            "MLflow experiment tracking",
            "Docker containerisation",
        ]
        for idea in ideas:
            st.markdown(
                f"<div style='color:var(--muted);font-size:.85rem;padding:4px 0;"
                f"border-bottom:1px solid var(--border);'>▸ {idea}</div>",
                unsafe_allow_html=True,
            )
        render_card_close()


# ============================================================
#  MAIN
# ============================================================
def main() -> None:
    inject_css()

    model, scaler, model_loaded = load_artifacts()

    page = render_sidebar(model_loaded)

    if page == "🏠  Home":
        page_home()
    elif page == "🔍  Fraud Detection":
        page_fraud_detection(model, scaler, model_loaded)
    elif page == "📊  Model Insights":
        page_model_insights()
    elif page == "📖  About Project":
        page_about()

    render_footer()


if __name__ == "__main__":
    main()
