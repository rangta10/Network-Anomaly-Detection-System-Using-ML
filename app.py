import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="Network Security Dashboard", layout="wide")

st.title("Network Security Intelligence Dashboard")

# Load models safely
nsl_model = None
cic_model = None
scaler_cic = None

try:
    nsl_model = joblib.load("anomaly_model.pkl")
except Exception:
    nsl_model = None

try:
    cic_model = joblib.load("models/cic_model.pkl")
    scaler_cic = joblib.load("models/scaler_cic.pkl")
except Exception:
    cic_model = None
    scaler_cic = None

mode = st.sidebar.selectbox("Select Mode", ["Beginner", "Advanced"])

uploaded_test = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_test is not None:

    test_data = pd.read_csv(uploaded_test)
    test_data.columns = test_data.columns.str.strip()

    st.success("Dataset loaded successfully.")
    st.write("Rows:", len(test_data))

    # Detect dataset type
    if "protocol_type" in test_data.columns:
        dataset_type = "NSL"
    elif "Flow Duration" in test_data.columns:
        dataset_type = "CIC"
    else:
        st.error("Unsupported dataset format.")
        st.stop()

    st.info(f"Detected Dataset Type: {dataset_type}")

    # ============================
    # NSL DATASET PROCESSING
    # ============================
    if dataset_type == "NSL":

        if nsl_model is None:
            st.error("NSL model file not found.")
            st.stop()

        if "label" in test_data.columns:
            X = test_data.drop("label", axis=1)
        else:
            X = test_data.copy()

        predictions = nsl_model.predict(X)

    # ============================
    # CIC DATASET PROCESSING
    # ============================
    elif dataset_type == "CIC":

        if cic_model is None or scaler_cic is None:
            st.error("CIC model or scaler file not found.")
            st.stop()

        if "Label" in test_data.columns:
            y_true = test_data["Label"]
            y_true = y_true.apply(lambda x: 0 if x == "BENIGN" else 1)
            X = test_data.drop("Label", axis=1)
        else:
            y_true = None
            X = test_data.copy()

        X = X.select_dtypes(include=["int64", "float64"])
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)

        X_scaled = scaler_cic.transform(X)
        predictions = cic_model.predict(X_scaled)

    else:
        st.stop()

    # Ensure predictions are numeric 0/1
    predictions = np.array(predictions).astype(int)

    test_data["Prediction"] = predictions

    total = len(predictions)
    anomalies = int(np.sum(predictions))
    normal = total - anomalies
    anomaly_percent = (anomalies / total) * 100 if total > 0 else 0

    # ============================
    # BEGINNER MODE
    # ============================
    if mode == "Beginner":

        st.subheader("Security Status Overview")

        if anomaly_percent < 5:
            status = "SAFE"
            color = "green"
        elif anomaly_percent < 20:
            status = "LOW RISK"
            color = "orange"
        else:
            status = "HIGH RISK"
            color = "red"

        st.markdown(f"### System Status: :{color}[{status}]")
        st.write(f"Anomaly Percentage: {anomaly_percent:.2f}%")

        safety_score = 100 - anomaly_percent

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=safety_score,
            title={'text': "Network Safety Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        pie = go.Figure(data=[go.Pie(
            labels=["Normal Traffic", "Suspicious Traffic"],
            values=[normal, anomalies],
            hole=0.5
        )])

        st.plotly_chart(pie, use_container_width=True)

    # ============================
    # ADVANCED MODE
    # ============================
    if mode == "Advanced":

        st.subheader("Detailed Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", total)
            st.metric("Anomalies Detected", anomalies)

        with col2:
            st.metric("Normal Traffic", normal)
            st.metric("Anomaly Percentage", f"{anomaly_percent:.2f}%")

        if dataset_type == "CIC" and "y_true" in locals() and y_true is not None:
            acc = accuracy_score(y_true, predictions)
            st.write("Model Accuracy:", round(acc, 4))

            cm = confusion_matrix(y_true, predictions)

            cm_fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Normal", "Attack"],
                y=["Normal", "Attack"],
                colorscale="Blues"
            ))

            st.plotly_chart(cm_fig, use_container_width=True)

        st.subheader("Dataset Preview")
        st.dataframe(test_data.head(100))

        csv = test_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Analysis Report",
            data=csv,
            file_name="security_report.csv",
            mime="text/csv",
        )
