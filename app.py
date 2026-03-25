import os
os.chdir(r"C:\Users\Nishant\Downloads\NDxGenius\fraud_detection")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ---------- Page Config ----------
st.set_page_config(
    page_title="Fraud Detection | SecureBank",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------- Custom CSS for Banking Vibe ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main background – subtle gradient */
    .stApp {
        background: linear-gradient(145deg, #f0f4fa 0%, #e6ecf3 100%);
    }

    /* Card styling for the main container */
    .main-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 28px;
        padding: 2rem;
        margin: 2rem auto;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.05), 0 8px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.7);
    }

    /* Header area */
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        color: #1e3a8a;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .header p {
        color: #4b5563;
        font-size: 1rem;
    }

    /* Input labels */
    .input-label {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
        display: block;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
        border-left: 8px solid #3b82f6;
    }
    .prediction-card.fraud {
        border-left-color: #ef4444;
    }
    .prediction-card.legitimate {
        border-left-color: #10b981;
    }

    /* Risk meter (progress bar) */
    .risk-meter {
        width: 100%;
        background-color: #e5e7eb;
        border-radius: 12px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .risk-fill {
        height: 12px;
        border-radius: 12px;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
        width: 0%;
        transition: width 0.3s ease;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a, #2563eb);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
    }

    /* Input fields */
    .stNumberInput input {
        border-radius: 20px;
        border: 1px solid #d1d5db;
        padding: 0.6rem 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper: Heuristic Risk Factor ----------
def heuristic_risk(amount, time):
    """
    Simple heuristic to simulate sensitivity to amount and time.
    Returns a risk factor between 0 and 1.
    """
    # Amount risk: high amounts increase risk, capped at 1
    amount_risk = min(amount / 5000, 1.0)
    # Time risk: very early or very late transactions might be riskier (demo only)
    # Here we assume time is seconds; risk increases slightly for large time values
    time_risk = min(time / 86400, 0.2)   # up to 20% extra risk after 1 day
    risk = amount_risk + time_risk
    return min(risk, 1.0)

# ---------- Load Model and Scaler ----------
@st.cache_resource
def load_models():
    model = joblib.load('models/fraud_detector.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'fraud_detector.pkl' and 'scaler.pkl' are in the 'models/' folder.")
    st.stop()

# ---------- Main UI ----------
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="header">
        <h1>🏦 SecureBank • Fraud Monitor</h1>
        <p>Real‑time transaction risk analysis with AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="input-label">💰 Transaction Amount</span>', unsafe_allow_html=True)
        amount = st.number_input("", min_value=0.0, value=100.0, step=10.0, key="amount", label_visibility="collapsed")
    with col2:
        st.markdown('<span class="input-label">⏱️ Time (seconds from first transaction)</span>', unsafe_allow_html=True)
        time_val = st.number_input("", min_value=0, value=0, step=1, key="time", label_visibility="collapsed")

    # Build feature dictionary (all PCA features set to 0)
    features = {
        'Time': time_val,
        'V1': 0.0, 'V2': 0.0, 'V3': 0.0, 'V4': 0.0, 'V5': 0.0,
        'V6': 0.0, 'V7': 0.0, 'V8': 0.0, 'V9': 0.0, 'V10': 0.0,
        'V11': 0.0, 'V12': 0.0, 'V13': 0.0, 'V14': 0.0, 'V15': 0.0,
        'V16': 0.0, 'V17': 0.0, 'V18': 0.0, 'V19': 0.0, 'V20': 0.0,
        'V21': 0.0, 'V22': 0.0, 'V23': 0.0, 'V24': 0.0, 'V25': 0.0,
        'V26': 0.0, 'V27': 0.0, 'V28': 0.0,
        'Amount': amount
    }

    # Predict button
    if st.button("🔍 Analyze Transaction", use_container_width=True):
        # 1. Model prediction (raw)
        X_new = pd.DataFrame([features])
        X_new[['Amount', 'Time']] = scaler.transform(X_new[['Amount', 'Time']])
        prob_model = model.predict_proba(X_new)[0, 1]

        # 2. Heuristic risk
        heuristic = heuristic_risk(amount, time_val)

        # 3. Combine: give more weight to heuristic for demonstration (70% heuristic, 30% model)
        #    This makes the final probability clearly respond to amount/time.
        final_prob = 0.7 * heuristic + 0.3 * prob_model

        # 4. Determine verdict
        pred = 1 if final_prob > 0.5 else 0

        # 5. Display results
        if pred == 1:
            card_class = "fraud"
            verdict = "⚠️ FRAUD ALERT"
            description = "This transaction has been flagged as high risk. Please review immediately."
            icon = "🚨"
        else:
            card_class = "legitimate"
            verdict = "✅ LEGITIMATE"
            description = "This transaction appears normal. No immediate action required."
            icon = "🔒"

        st.markdown(f"""
        <div class="prediction-card {card_class}">
            <h3>{icon} {verdict}</h3>
            <p>{description}</p>
            <div class="risk-meter">
                <div class="risk-fill" style="width: {final_prob*100}%;"></div>
            </div>
            <p style="margin-top: 0.5rem;"><strong>Fraud Probability:</strong> {final_prob:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

        # Optional: show model raw probability and heuristic for transparency
        with st.expander("Detailed analysis"):
            st.write(f"**Model raw probability:** {prob_model:.2%}")
            st.write(f"**Amount/Time risk factor:** {heuristic:.2%}")
            st.write("Final probability is a combination (70% risk factor + 30% model).")

        if final_prob > 0.7:
            st.warning("High‑risk transaction! Additional verification recommended.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Powered by AI • SecureBank fraud detection system | Model trained on credit card transaction data</p>
</div>
""", unsafe_allow_html=True)