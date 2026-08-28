# Hospital Readmission AI - Ultra Enterprise Healthcare Edition (RESTRUCTURED, CLEAN, FIXED)
# Save as: app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import hashlib
import warnings
import random
from datetime import datetime

warnings.filterwarnings("ignore")

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Viz
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Hospital Readmission AI", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# REALISTIC HOSPITAL DASHBOARD UI
# ---------------------------
st.markdown("""
<style>
:root {
    --bg: #f3f6fb;
    --card: #ffffff;
    --line: #dbe4f0;
    --text: #0f172a;
    --muted: #64748b;
    --primary: #2563eb;
    --success: #059669;
    --warning: #d97706;
    --danger: #dc2626;
    --navy: #0f172a;
}
.stApp {
    background: linear-gradient(200deg, #f8fbff 0%, #eef4fb 100%);
    color: var(--text);
}
.block-container {
    padding-top: 5rem !important;
    padding-bottom: 3rem !important;
    max-width: 200% !important;
}
.main-header {
    background: linear-gradient(135deg, #0b1b35, #13294b 55%, #1e3a5f);
    padding: 26px 30px;
    border-radius: 26px;
    margin-bottom: 18px;
    box-shadow: 0 18px 40px rgba(15,23,42,0.18);
    width: 100%;
}
.big-title {
    font-size: clamp(1.8rem, 2.5vw, 2.9rem);
    font-weight: 800;
    color: white;
    line-height: 1.15;
}
.sub-title {
    font-size: 1rem;
    color: #dbe7ff;
    margin-top: 8px;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text);
    margin: 10px 0 14px;
}
.card, .metric-card, .info-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 18px;
    box-shadow: 0 10px 28px rgba(15,23,42,0.06);
}
.metric-card h3 {
    margin: 0;
    font-size: 2rem;
    color: var(--text);
}
.metric-card p {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 0.95rem;
}
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}
.badge {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background:#e0ecff;
    color:#1d4ed8;
    font-weight:700;
    font-size:0.8rem;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
    min-width: 320px !important;
    max-width: 320px !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
.stButton > button, .stDownloadButton > button {
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.68rem 1rem !important;
    font-weight: 700 !important;
}
.stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8) !important; }
.stDownloadButton > button { background: linear-gradient(135deg, #059669, #047857) !important; }
input, textarea, [data-baseweb="select"] div { color: var(--text) !important; }
label, .stSelectbox label, .stSlider label, .stTextInput label { color: var(--text) !important; font-weight: 600 !important; }
@media (max-width: 1100px) { .kpi-strip { grid-template-columns: repeat(2, 1fr);} }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# SESSION STATE
# ---------------------------
def init_state():
    defaults = {
        "logged_in": False,
        "username": "",
        "page": "Home",
        "df": None,
        "clean_df": None,
        "encoders": {},
        "scaler": None,
        "feature_columns": [],
        "trained_models": {},
        "best_model": None,
        "best_model_name": None,
        "model_results": None,
        "prediction_log": pd.DataFrame(),
        "simulation_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
USERS_FILE = "users.json"

# ---------------------------
# AUTH
# ---------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def register_user(username, password):
    users = load_users()
    if not username or not password:
        return False, "Username and password required"
    if username in users:
        return False, "Username already exists"
    users[username] = hash_password(password)
    save_users(users)
    return True, "Registration successful"

def login_user(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

# ---------------------------
# HELPERS
# ---------------------------
def risk_label(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.45:
        return "Medium Risk"
    return "Low Risk"

def create_age_group(age):
    if age < 30:
        return "18-29"
    elif age < 45:
        return "30-44"
    elif age < 60:
        return "45-59"
    elif age < 75:
        return "60-74"
    return "75+"

# ---------------------------
# DATA GENERATION
# ---------------------------
@st.cache_data(show_spinner=False)
def generate_hospital_dataset(n=2500, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    genders = ["Male", "Female"]
    admission_types = ["Emergency", "Urgent", "Elective", "Trauma", "Referral"]
    diagnosis_categories = ["Diabetes", "Cardiac", "Respiratory", "Renal", "Infection", "Orthopedic", "Neurology", "Cancer"]
    discharge_dispositions = ["Home", "Transferred", "Rehab", "Nursing Facility", "Expired"]
    insurance_types = ["Private", "Government", "Self-pay"]
    marital_statuses = ["Single", "Married", "Divorced", "Widowed"]

    rows = []
    for i in range(n):
        age = np.random.randint(18, 91)
        admission_type = np.random.choice(admission_types, p=[0.35, 0.2, 0.2, 0.1, 0.15])
        diagnosis = np.random.choice(diagnosis_categories)
        discharge = np.random.choice(discharge_dispositions, p=[0.65, 0.12, 0.08, 0.12, 0.03])

        previous_admissions = np.random.poisson(2)
        num_medications = np.random.randint(1, 30)
        time_in_hospital = np.random.randint(1, 15)
        number_emergency = np.random.poisson(1)
        number_inpatient = np.random.poisson(1)
        chronic_conditions = np.random.randint(0, 6)
        missed_followups = np.random.randint(0, 5)

        risk_score = (
            age * 0.012 + previous_admissions * 0.9 + num_medications * 0.08 +
            time_in_hospital * 0.18 + number_inpatient * 1.1 + number_emergency * 0.8 +
            chronic_conditions * 0.9 + missed_followups * 1.0
        )
        if admission_type == "Emergency": risk_score += 2.2
        if diagnosis in ["Cardiac", "Renal", "Respiratory", "Diabetes"]: risk_score += 2.4
        if discharge in ["Transferred", "Nursing Facility", "Rehab"]: risk_score += 1.8

        prob = 1 / (1 + np.exp(-(risk_score - 11.5) / 3.5))
        readmitted = np.random.binomial(1, prob)

        rows.append({
            "patient_id": f"PT{100000+i}",
            "age": age,
            "gender": np.random.choice(genders),
            "marital_status": np.random.choice(marital_statuses),
            "insurance_type": np.random.choice(insurance_types, p=[0.4, 0.45, 0.15]),
            "admission_type": admission_type,
            "diagnosis_category": diagnosis,
            "discharge_disposition": discharge,
            "num_lab_procedures": np.random.randint(10, 100),
            "num_medications": num_medications,
            "num_procedures": np.random.randint(0, 8),
            "time_in_hospital": time_in_hospital,
            "number_outpatient": np.random.poisson(1),
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "previous_admissions": previous_admissions,
            "a1c_result": np.random.choice(["Normal", "Abnormal", "Unknown"], p=[0.4, 0.45, 0.15]),
            "glucose_result": np.random.choice(["Normal", "High", "Very High"], p=[0.45, 0.4, 0.15]),
            "chronic_conditions": chronic_conditions,
            "missed_followups": missed_followups,
            "readmitted_30_days": readmitted
        })

    return pd.DataFrame(rows)

# ---------------------------
# FIXED CLEANING (BUG FIX HERE)
# ---------------------------
def clean_data(df):
    df = df.copy()
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    missing_before = int(df.isnull().sum().sum())

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(str)
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].replace(["nan", "None", "NaN"], np.nan)
            df[col] = df[col].fillna(fill_val)

    if "age" in df.columns:
        df["age_group"] = df["age"].apply(create_age_group)
    if "time_in_hospital" in df.columns:
        df["high_los_risk"] = (df["time_in_hospital"] >= 8).astype(int)
    if "previous_admissions" in df.columns:
        df["frequent_admission_flag"] = (df["previous_admissions"] >= 3).astype(int)
    if "num_medications" in df.columns:
        df["medication_burden"] = pd.cut(df["num_medications"], bins=[-1, 5, 10, 20, 100], labels=["Low", "Moderate", "High", "Very High"]).astype(str)
    if set(["chronic_conditions", "number_inpatient", "number_emergency", "missed_followups"]).issubset(df.columns):
        df["clinical_complexity_score"] = df["chronic_conditions"] + df["number_inpatient"] + df["number_emergency"] + df["missed_followups"]
    if set(["a1c_result", "glucose_result"]).issubset(df.columns):
        df["lab_risk_score"] = ((df["a1c_result"] == "Abnormal").astype(int) + (df["glucose_result"] == "Very High").astype(int))

    missing_after = int(df.isnull().sum().sum())
    return df, duplicate_count, missing_before, missing_after

# ---------------------------
# PREPARE / TRAIN
# ---------------------------
def prepare_training_data(df, target="readmitted_30_days"):
    data = df.copy()
    if "patient_id" in data.columns:
        data = data.drop(columns=["patient_id"])
    X = data.drop(columns=[target])
    y = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)

    encoders = {}
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    feature_columns = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=feature_columns), y, encoders, scaler, feature_columns

@st.cache_resource(show_spinner=False)
def train_models_cached(X_df, y_series):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42, stratify=y_series)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=42)
    }
    results, trained = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "F1 Score": round(f1_score(y_test, preds, zero_division=0), 4),
            "ROC-AUC": round(roc_auc_score(y_test, probs), 4)
        })
        trained[name] = {"model": model, "X_test": X_test, "y_test": y_test, "preds": preds, "probs": probs}

    results_df = pd.DataFrame(results).sort_values(by=["Recall", "ROC-AUC"], ascending=False).reset_index(drop=True)
    best_model_name = results_df.iloc[0]["Model"]
    return results_df, trained, best_model_name

# ---------------------------
# TRANSFORM INPUT
# ---------------------------
def transform_input_for_prediction(input_df):
    df = input_df.copy()
    df["readmitted_30_days"] = 0
    df, _, _, _ = clean_data(df)
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id", "readmitted_30_days"], errors="ignore")

    for col in df.columns:
        if col in st.session_state.encoders:
            le = st.session_state.encoders[col]
            safe_vals = [v if v in le.classes_ else le.classes_[0] for v in df[col].astype(str)]
            df[col] = le.transform(safe_vals)

    for col in st.session_state.feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[st.session_state.feature_columns]
    return st.session_state.scaler.transform(df)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="main-header">
    <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap;">
        <div style="font-size:2.2rem;line-height:1;">🏥</div>
        <div style="flex:1;min-width:0;">
            <div class="big-title">Hospital Readmission AI</div>
            <div class="sub-title">Ultra Enterprise Healthcare Edition — clean, professional, stable, and simple to use.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR NAV
# ---------------------------
PAGES = ["Home", "Login / Register", "Dataset Center", "Data Cleaning", "EDA Dashboard", "Train Models", "Model Evaluation", "Patient Prediction", "Simulation", "Reports"]

with st.sidebar:
    st.markdown("<h2 style='margin-top:0;margin-bottom:10px;color:#0f172a;'>🏥 Menu</h2>", unsafe_allow_html=True)
    selected = st.radio("Open Module", PAGES, index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0)
    st.session_state.page = selected
    if st.session_state.logged_in:
        st.success(f"Logged in: {st.session_state.username}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.page = "Home"
            st.rerun()

page = st.session_state.page

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    st.markdown("""
    <style>
    .hero-wrap{display:grid;grid-template-columns:1.35fr 0.95fr;gap:18px;align-items:stretch;margin-top:6px;}
    .hero-panel{background:linear-gradient(135deg,#ffffff,#f8fbff);border:1px solid #dbe4f0;border-radius:28px;padding:28px;box-shadow:0 14px 34px rgba(15,23,42,.07);min-height:280px;display:flex;flex-direction:column;justify-content:space-between;overflow:hidden;}
    .hero-title{font-size:clamp(2rem,3.4vw,3.25rem);line-height:1.08;font-weight:900;color:#0f172a;margin:14px 0 10px;word-break:break-word;white-space:normal;}
    .hero-text{font-size:1.04rem;line-height:1.85;color:#475569;max-width:95%;}
    .chip{display:inline-flex;align-items:center;gap:8px;background:#eaf1ff;color:#1d4ed8;padding:10px 16px;border-radius:999px;font-size:.84rem;font-weight:800;width:fit-content;}
    .status-box{background:linear-gradient(180deg,#0f172a,#162d50);color:#fff;border-radius:28px;padding:26px;min-height:280px;box-shadow:0 14px 34px rgba(15,23,42,.16);display:flex;flex-direction:column;justify-content:space-between;}
    .status-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;}
    .status-mini{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:16px;min-height:96px;display:flex;flex-direction:column;justify-content:space-between;}
    .status-mini h3{margin:0;font-size:1.55rem;color:#fff;line-height:1.1;}
    .status-mini p{margin:0;color:#dbe7ff;font-size:.92rem;}
    .section-title{font-size:1.5rem;font-weight:900;color:#0f172a;margin:28px 0 14px;}
    .uniform-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
    .uniform-card{background:#fff;border:1px solid #dbe4f0;border-radius:24px;padding:22px;box-shadow:0 10px 28px rgba(15,23,42,.06);min-height:170px;display:flex;flex-direction:column;justify-content:space-between;}
    .uniform-card h3{margin:0;font-size:2rem;color:#0f172a;line-height:1.1;word-break:break-word;}
    .uniform-card p{margin:10px 0 0;color:#64748b;font-size:.96rem;line-height:1.6;}
    .split-grid{display:grid;grid-template-columns:1.2fr 1fr;gap:18px;margin-top:18px;}
    .panel-card{background:#fff;border:1px solid #dbe4f0;border-radius:26px;padding:24px;box-shadow:0 10px 28px rgba(15,23,42,.06);height:100%;}
    .goal-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px;}
    .goal-item{background:#f8fbff;border:1px solid #e2e8f0;border-radius:18px;padding:16px;min-height:110px;}
    .goal-item b{display:block;color:#0f172a;margin-bottom:6px;}
    .goal-item span{color:#64748b;line-height:1.65;}
    .use-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:10px;}
    .use-card{background:#fff;border:1px solid #dbe4f0;border-radius:24px;padding:22px;box-shadow:0 10px 28px rgba(15,23,42,.06);min-height:165px;}
    .use-card h4{margin:0 0 10px;color:#0f172a;font-size:1.1rem;}
    .use-card p{margin:0;color:#64748b;line-height:1.75;}
    @media (max-width:1100px){.hero-wrap,.split-grid,.uniform-grid,.use-grid,.goal-grid,.status-grid{grid-template-columns:1fr !important;}.hero-panel,.status-box,.uniform-card,.use-card,.goal-item{min-height:auto;}.hero-text{max-width:100%;}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='hero-wrap'>
        <div class='hero-panel'>
            <div>
                <div class='chip'>🏥 Clinical Decision Support System</div>
                <div class='hero-title'>Hospital Executive Overview</div>
                <div class='hero-text'>Predict 30-day readmission risk, surface patient-level warning indicators, and support discharge planning with a realistic healthcare analytics workflow designed for clinical and operational teams.</div>
            </div>
            <div style='display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;'>
                <span class='chip' style='background:#ecfdf5;color:#047857;'>✓ Readmission Risk Engine</span>
                <span class='chip'>✓ Model Evaluation Suite</span>
                <span class='chip' style='background:#fff7ed;color:#c2410c;'>✓ Patient Simulation Queue</span>
            </div>
        </div>
        <div class='status-box'>
            <div>
                <div style='font-size:.92rem;font-weight:800;color:#bfdbfe;letter-spacing:.03em;'>CARE OPERATIONS STATUS</div>
                <div style='font-size:1.8rem;font-weight:900;line-height:1.2;margin-top:8px;'>Readmission Intelligence Center</div>
                <div style='color:#dbe7ff;line-height:1.8;margin-top:10px;'>A cleaner executive panel for patient risk surveillance, model validation, and hospital decision support.</div>
            </div>
            <div class='status-grid'>
                <div class='status-mini'><h3>5</h3><p>ML Models</p></div>
                <div class='status-mini'><h3>30</h3><p>Day Window</p></div>
                <div class='status-mini'><h3>Recall</h3><p>Priority Metric</p></div>
                <div class='status-mini'><h3>Live</h3><p>Simulation Ready</p></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Operational Snapshot</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='uniform-grid'>
        <div class='uniform-card'><h3>5</h3><p>Validated machine learning models available for benchmarking and hospital risk scoring.</p></div>
        <div class='uniform-card'><h3>30-Day</h3><p>Clinical readmission prediction window aligned to common healthcare quality measures.</p></div>
        <div class='uniform-card'><h3>Recall-First</h3><p>Detection strategy designed to reduce the chance of missing high-risk patients.</p></div>
        <div class='uniform-card'><h3>Real-Time</h3><p>Simulation pipeline for testing patient records and live-style readmission scenarios.</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='split-grid'>
        <div class='panel-card'>
            <h3 style='margin-top:0;color:#0f172a;'>Care Management Goals</h3>
            <div class='goal-grid'>
                <div class='goal-item'><b>High-Risk Identification</b><span>Flag patients likely to return within 30 days after discharge.</span></div>
                <div class='goal-item'><b>Discharge Planning</b><span>Support follow-up care, counseling, and medication readiness.</span></div>
                <div class='goal-item'><b>Clinical Monitoring</b><span>Track chronic complexity, prior visits, and treatment burden.</span></div>
                <div class='goal-item'><b>Quality Analytics</b><span>Support hospital operations, reporting, and care improvement efforts.</span></div>
            </div>
        </div>
        <div class='panel-card'>
            <h3 style='margin-top:0;color:#0f172a;'>Recommended Workflow</h3>
            <ol style='color:#334155;line-height:2.05;padding-left:18px;margin-top:12px;'>
                <li>Login / Register</li>
                <li>Generate or Upload Dataset</li>
                <li>Run Cleaning & Feature Engineering</li>
                <li>Train and Compare Models</li>
                <li>Evaluate Clinical Performance</li>
                <li>Predict and Simulate Patient Risk</li>
            </ol>
            <div style='margin-top:18px;padding:16px;border-radius:18px;background:#eff6ff;border:1px solid #bfdbfe;color:#1e3a8a;line-height:1.8;'>
                <b>Executive Note:</b> In readmission prediction, <b>recall</b> is often prioritized because missing a high-risk patient can be more costly than a false positive alert.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Hospital Use Cases</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='use-grid'>
        <div class='use-card'><h4>Nurse Case Review</h4><p>Quickly identify patients who may require follow-up calls, medication checks, or discharge counseling.</p></div>
        <div class='use-card'><h4>Doctor Discharge Support</h4><p>Use model-backed risk insights before discharge to improve planning and continuity of care.</p></div>
        <div class='use-card'><h4>Quality Improvement Team</h4><p>Track readmission patterns and risk drivers across patients, admissions, and treatment history.</p></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# LOGIN
# ---------------------------
elif page == "Login / Register":
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.session_state.page = "Home"
                st.success("Login successful. Redirecting to Home...")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        ru = st.text_input("Create Username")
        rp = st.text_input("Create Password", type="password")
        if st.button("Register"):
            ok, msg = register_user(ru, rp)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

# ---------------------------
# DATASET
# ---------------------------
elif page == "Dataset Center":
    st.subheader("Dataset Center")
    c1, c2 = st.columns(2)
    with c1:
        rows = st.slider("Rows", 500, 6000, 2500, 500)
        if st.button("Generate Dataset"):
            st.session_state.df = generate_hospital_dataset(rows, seed=random.randint(1, 9999))
            st.session_state.clean_df = None
            st.success("Dataset generated successfully")
    with c2:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            st.session_state.df = pd.read_csv(uploaded)
            st.session_state.clean_df = None
            st.success("Dataset uploaded successfully")

    if st.session_state.df is not None:
        df = st.session_state.df
        a, b, c, d = st.columns(4)
        a.metric("Rows", df.shape[0])
        b.metric("Columns", df.shape[1])
        c.metric("Duplicates", int(df.duplicated().sum()))
        d.metric("Missing", int(df.isnull().sum().sum()))
        st.dataframe(df.head(15), use_container_width=True)

# ---------------------------
# CLEANING
# ---------------------------
elif page == "Data Cleaning":
    st.subheader("Data Cleaning & Feature Engineering")
    if st.session_state.df is None:
        st.warning("Generate or upload a dataset first")
    else:
        if st.button("Run Cleaning"):
            clean_df, dup_count, missing_before, missing_after = clean_data(st.session_state.df)
            st.session_state.clean_df = clean_df
            a, b, c = st.columns(3)
            a.metric("Duplicates Removed", dup_count)
            b.metric("Missing Before", missing_before)
            c.metric("Missing After", missing_after)
            st.success("Cleaning completed successfully")
        if st.session_state.clean_df is not None:
            st.dataframe(st.session_state.clean_df.head(15), use_container_width=True)

# ---------------------------
# EDA
# ---------------------------
elif page == "EDA Dashboard":
    st.subheader("EDA Dashboard")
    if st.session_state.clean_df is None:
        st.warning("Run data cleaning first")
    else:
        df = st.session_state.clean_df
        st.plotly_chart(px.pie(df, names=df["readmitted_30_days"].map({0: "No", 1: "Yes"}), title="Readmission Distribution"), use_container_width=True)
        st.plotly_chart(px.histogram(df, x="diagnosis_category", color=df["readmitted_30_days"].map({0: "No", 1: "Yes"}), barmode="group", title="Diagnosis vs Readmission"), use_container_width=True)

# ---------------------------
# TRAIN
# ---------------------------
elif page == "Train Models":
    st.subheader("Train Models")
    if st.session_state.clean_df is None:
        st.warning("Run data cleaning first")
    else:
        if st.button("Train All Models"):
            X, y, encoders, scaler, feature_cols = prepare_training_data(st.session_state.clean_df)
            results_df, trained, best_model_name = train_models_cached(X, y)
            st.session_state.encoders = encoders
            st.session_state.scaler = scaler
            st.session_state.feature_columns = feature_cols
            st.session_state.trained_models = trained
            st.session_state.model_results = results_df
            st.session_state.best_model_name = best_model_name
            st.session_state.best_model = trained[best_model_name]["model"]
            st.success(f"Training complete. Best model: {best_model_name}")
        if st.session_state.model_results is not None:
            st.dataframe(st.session_state.model_results, use_container_width=True)

# ---------------------------
# EVALUATION
# ---------------------------
elif page == "Model Evaluation":
    st.subheader("Model Evaluation")
    if not st.session_state.trained_models:
        st.warning("Train models first")
    else:
        model_name = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
        md = st.session_state.trained_models[model_name]
        cm = confusion_matrix(md["y_test"], md["preds"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

# ---------------------------
# PREDICTION
# ---------------------------
elif page == "Patient Prediction":
    st.subheader("Patient Risk Prediction")
    if st.session_state.best_model is None:
        st.warning("Train models first")
    else:
        with st.form("predict_form"):
            c1, c2 = st.columns(2)
            with c1:
                patient_id = st.text_input("Patient ID", value=f"NEW{random.randint(1000,9999)}")
                age = st.slider("Age", 18, 90, 55)
                gender = st.selectbox("Gender", ["Male", "Female"])
                diagnosis_category = st.selectbox("Diagnosis Category", ["Diabetes", "Cardiac", "Respiratory", "Renal", "Infection", "Orthopedic", "Neurology", "Cancer"])
                admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective", "Trauma", "Referral"])
                discharge_disposition = st.selectbox("Discharge Disposition", ["Home", "Transferred", "Rehab", "Nursing Facility", "Expired"])
            with c2:
                num_medications = st.slider("Medications", 0, 40, 12)
                time_in_hospital = st.slider("Hospital Stay", 1, 20, 5)
                previous_admissions = st.slider("Previous Admissions", 0, 10, 2)
                number_emergency = st.slider("Emergency Visits", 0, 10, 1)
                number_inpatient = st.slider("Inpatient Visits", 0, 10, 1)
                chronic_conditions = st.slider("Chronic Conditions", 0, 8, 2)
            submit = st.form_submit_button("Predict Risk")

        if submit:
            input_df = pd.DataFrame([{
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "marital_status": "Single",
                "insurance_type": "Private",
                "admission_type": admission_type,
                "diagnosis_category": diagnosis_category,
                "discharge_disposition": discharge_disposition,
                "num_lab_procedures": 45,
                "num_medications": num_medications,
                "num_procedures": 2,
                "time_in_hospital": time_in_hospital,
                "number_outpatient": 1,
                "number_emergency": number_emergency,
                "number_inpatient": number_inpatient,
                "previous_admissions": previous_admissions,
                "a1c_result": "Normal",
                "glucose_result": "High",
                "chronic_conditions": chronic_conditions,
                "missed_followups": 1
            }])
            X_scaled = transform_input_for_prediction(input_df)
            pred = st.session_state.best_model.predict(X_scaled)[0]
            prob = float(st.session_state.best_model.predict_proba(X_scaled)[0][1])
            label = risk_label(prob)
            a, b, c = st.columns(3)
            a.metric("Prediction", "Readmitted" if pred == 1 else "Not Readmitted")
            b.metric("Risk Probability", f"{prob:.2%}")
            c.metric("Risk Category", label)

# ---------------------------
# SIMULATION
# ---------------------------
elif page == "Simulation":
    st.subheader("Real-Time Simulation")
    if st.session_state.best_model is None:
        st.warning("Train models first")
    else:
        n = st.slider("Patients to Simulate", 5, 25, 10)
        if st.button("Run Simulation"):
            sim_df = generate_hospital_dataset(n, seed=random.randint(1, 9999)).head(n).copy()
            X_scaled = transform_input_for_prediction(sim_df)
            probs = st.session_state.best_model.predict_proba(X_scaled)[:, 1]
            preds = st.session_state.best_model.predict(X_scaled)
            sim_df["risk_probability"] = probs
            sim_df["risk_category"] = sim_df["risk_probability"].apply(risk_label)
            sim_df["prediction"] = np.where(preds == 1, "Readmitted", "Not Readmitted")
            st.session_state.simulation_df = sim_df
        if st.session_state.simulation_df is not None:
            st.dataframe(st.session_state.simulation_df[["patient_id", "age", "diagnosis_category", "risk_probability", "risk_category", "prediction"]], use_container_width=True)

# ---------------------------
# REPORTS
# ---------------------------
elif page == "Reports":
    st.subheader("Reports & Downloads")
    if st.session_state.df is not None:
        st.download_button("Download Original Dataset", st.session_state.df.to_csv(index=False).encode("utf-8"), "hospital_dataset.csv", "text/csv")
    if st.session_state.clean_df is not None:
        st.download_button("Download Cleaned Dataset", st.session_state.clean_df.to_csv(index=False).encode("utf-8"), "cleaned_hospital_dataset.csv", "text/csv")
    if st.session_state.model_results is not None:
        st.download_button("Download Model Results", st.session_state.model_results.to_csv(index=False).encode("utf-8"), "model_results.csv", "text/csv")
