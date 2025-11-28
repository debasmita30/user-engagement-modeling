import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# ---------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------
st.set_page_config(
    page_title="Engagement Predictor",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------
# LOAD MODELS + SCALER + MEANS
# ---------------------------------------------
MODEL_DIR = r"C:\Users\Diya\Downloads\teams-engagement-project\models"

rf_reg = joblib.load(f"{MODEL_DIR}\\rf_reg.pkl")
scaler = joblib.load(f"{MODEL_DIR}\\scaler.pkl")

with open(f"{MODEL_DIR}\\feature_means.json", "r") as f:
    feature_means = json.load(f)

expected_features = list(feature_means.keys())

# These are the ONLY 4 user inputs in UI
ui_features = [
    "message_count_sum",
    "meeting_count_sum",
    "session_count",
    "session_duration_sec_mean"
]


# ---------------------------------------------
# THEME TOGGLE
# ---------------------------------------------
st.sidebar.markdown("### 🌓 Theme Switch")
theme = st.sidebar.radio("Choose theme:", ["Light", "Dark"])

if theme == "Dark":
    st.write("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------
# Sidebar Navigation
# ---------------------------------------------
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📈 Predict Engagement",
    "📊 Feature Importance",
    "🔍 Local Explanation",
    "📥 Download Report"
])


# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def build_full_9_feature_vector(ui_data: dict):
    """
    Takes the 4 UI inputs and fills remaining features using means.json
    """
    full = feature_means.copy()
    full.update(ui_data)
    return pd.DataFrame([full], columns=expected_features)


def predict(full_df):
    X_scaled = scaler.transform(full_df)
    pred = rf_reg.predict(X_scaled)[0]
    return pred


# ---------------------------------------------
# HOME PAGE
# ---------------------------------------------
if page == "🏠 Home":
    st.title("📊 Engagement Analytics Dashboard")

    st.write("""
A clean and modern interface to predict user engagement,
view feature influence, and download insights.
    """)

    st.markdown("### Features of This App")
    st.write("""
- Predict engagement from 4 simple inputs  
- Model uses all 9 engineered features internally  
- Dark/light theme  
- Feature importance visualizations  
- Local explanation  
- Downloadable report  
""")


# ---------------------------------------------
# PREDICT PAGE
# ---------------------------------------------
elif page == "📈 Predict Engagement":
    st.title("📈 Predict User Engagement")

    col1, col2 = st.columns(2)

    with st.form("predict_form"):

        msg = col1.number_input(
            "💬 message_count_sum",
            min_value=0.0,
            value=float(feature_means["message_count_sum"])
        )

        meet = col1.number_input(
            "📅 meeting_count_sum",
            min_value=0.0,
            value=float(feature_means["meeting_count_sum"])
        )

        sess = col2.number_input(
            "📌 session_count",
            min_value=1.0,
            value=float(5.0)
        )

        avg = col2.number_input(
            "⏱ session_duration_sec_mean",
            min_value=1.0,
            value=float(feature_means["session_duration_sec_mean"])
        )

        submitted = st.form_submit_button("Predict Engagement 🎯")

    if submitted:
        ui_dict = {
            "message_count_sum": msg,
            "meeting_count_sum": meet,
            "session_count": sess,
            "session_duration_sec_mean": avg
        }

        full_df = build_full_9_feature_vector(ui_dict)
        pred = predict(full_df)

        st.success(f"🎯 Predicted Engagement Score: **{pred:.2f}**")

        st.session_state.last_full = full_df
        st.session_state.last_pred = pred

        st.markdown("### Full Feature Vector Used (Auto-filled + User Inputs)")
        st.dataframe(full_df.T)


# ---------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------
elif page == "📊 Feature Importance":
    st.title("📊 Feature Importance")

    importances = rf_reg.feature_importances_

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(expected_features, importances, color="#4682B4")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Influence")

    st.pyplot(fig)


# ---------------------------------------------
# LOCAL EXPLANATION
# ---------------------------------------------
elif page == "🔍 Local Explanation":
    st.title("🔍 Local Explanation (Simple)")

    if "last_full" not in st.session_state:
        st.warning("⚠ First make a prediction from the 'Predict' page.")

    else:
        df = st.session_state.last_full

        contrib = df.values[0] * rf_reg.feature_importances_

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(expected_features, contrib, color="#FFA500")
        ax.set_xlabel("Contribution")
        ax.set_title("Local Feature Contributions")
        st.pyplot(fig)


# ---------------------------------------------
# DOWNLOAD REPORT
# ---------------------------------------------
elif page == "📥 Download Report":
    st.title("📥 Download Prediction Report")

    if "last_pred" not in st.session_state:
        st.warning("⚠ Make a prediction first.")
    else:
        df = st.session_state.last_full
        pred = st.session_state.last_pred

        report = "Engagement Prediction Report\n\n"
        for k, v in df.iloc[0].items():
            report += f"{k}: {v}\n"

        report += f"\nPredicted Score: {pred:.2f}\n"

        buffer = BytesIO()
        buffer.write(report.encode())
        buffer.seek(0)

        b64 = base64.b64encode(buffer.read()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">📄 Download Report</a>'
        st.markdown(href, unsafe_allow_html=True)
