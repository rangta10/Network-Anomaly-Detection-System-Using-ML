import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit_authenticator as stauth

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Network Security Intelligence Dashboard",
    layout="wide"
)

# ---------------------------------------------------------
# AUTHENTICATION
# ---------------------------------------------------------
credentials = {
    "usernames": {
        "admin": {
            "name": "Administrator",
            "password": "$2b$12$mMtILkxbK.LeO1KGPs/n2O6K/CyaOZLDHVEShbQYbUsWB.NXA6rqa"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "secure_cookie",
    "secure_key",
    1
)

authenticator.login(location="main")
auth_status = st.session_state.get("authentication_status")

if auth_status is None:
    st.warning("Please login to continue.")
    st.stop()

if auth_status is False:
    st.error("Invalid login credentials.")
    st.stop()

authenticator.logout(location="sidebar")
st.sidebar.success(f"Welcome {st.session_state.get('name')}")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Select Mode",
    ["Beginner Mode", "Advanced Mode"]
)

train_file = st.sidebar.file_uploader("Upload Training Dataset (CSV)", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Testing Dataset (CSV)", type=["csv"])

st.title("Network Security Intelligence Dashboard")

# ---------------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------------
if train_file and test_file:

    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
    if 'class' in categorical_cols:
        categorical_cols.remove('class')

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    encoded_train = encoder.fit_transform(train_data[categorical_cols])
    encoded_train_df = pd.DataFrame(
        encoded_train,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    train_encoded = pd.concat(
        [train_data.drop(columns=categorical_cols).reset_index(drop=True),
         encoded_train_df.reset_index(drop=True)],
        axis=1
    )

    X_train = train_encoded.drop("class", axis=1)
    y_train = train_encoded["class"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    encoded_test = encoder.transform(test_data[categorical_cols])
    encoded_test_df = pd.DataFrame(
        encoded_test,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    test_encoded = pd.concat(
        [test_data.drop(columns=categorical_cols).reset_index(drop=True),
         encoded_test_df.reset_index(drop=True)],
        axis=1
    )

    if "class" in test_data.columns:
        y_test = test_data["class"]
        X_test = test_encoded.drop("class", axis=1)
    else:
        y_test = None
        X_test = test_encoded

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    test_data["Predicted_Class"] = predictions
    test_data["Confidence"] = probabilities.max(axis=1)

    total = len(test_data)
    threats = (test_data["Predicted_Class"] != "normal").sum()
    threat_percent = (threats / total) * 100
    safety_score = 100 - threat_percent
    avg_confidence = test_data["Confidence"].mean()

    # =====================================================
    # BEGINNER MODE
    # =====================================================
    if mode == "Beginner Mode":

        st.header("Network Safety Overview")

        if threat_percent == 0:
            st.success("System Status: SAFE")
            explanation = "No suspicious activity detected."
        elif threat_percent < 5:
            st.warning("System Status: LOW RISK")
            explanation = "Minor unusual activity detected."
        elif threat_percent < 20:
            st.error("System Status: MEDIUM RISK")
            explanation = "Suspicious activity detected. Review recommended."
        else:
            st.error("System Status: HIGH RISK")
            explanation = "Significant malicious patterns detected."

        st.info(explanation)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records Checked", total)
        col2.metric("Suspicious Records", threats)
        col3.metric("Threat Percentage", f"{threat_percent:.2f}%")

        # Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=safety_score,
            title={'text': "Network Safety Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 60], 'color': "orange"},
                    {'range': [60, 85], 'color': "yellow"},
                    {'range': [85, 100], 'color': "green"},
                ],
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        
        
        # ---- INSIGHT SECTION (fills the gap) ----
        st.markdown("---")
        st.subheader("Security Insight Summary")

        colA, colB, colC = st.columns(3)
        colA.metric("Average Detection Confidence", f"{avg_confidence:.2f}")
        colB.metric("Normal Traffic Records", total - threats)
        colC.metric("Suspicious Traffic Records", threats)

        severity = go.Figure(go.Indicator(
            mode="number+gauge",
            value=threat_percent,
            title={'text': "Threat Level %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 5], 'color': "green"},
                    {'range': [5, 20], 'color': "yellow"},
                    {'range': [20, 50], 'color': "orange"},
                    {'range': [50, 100], 'color': "red"},
                ],
            }
        ))
        st.plotly_chart(severity, use_container_width=True)  
        
        # ---- IMPROVED DONUT CHART ----
        st.subheader("Traffic Distribution Overview")

        normal_count = total - threats

        fig_pie = go.Figure(data=[go.Pie(
            labels=["Normal Traffic", "Suspicious Traffic"],
            values=[normal_count, threats],
            hole=0.55,
            marker=dict(
                colors=["#2ECC71", "#E74C3C"],
                line=dict(color="#111111", width=2)
            ),
            pull=[0, 0.08],
            textinfo="percent+label",
            textfont=dict(size=14),
            hovertemplate="<b>%{label}</b><br>" +
                          "Count: %{value}<br>" +
                          "Percentage: %{percent}<extra></extra>"
        )])

        fig_pie.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=40, b=40, l=0, r=0),
            annotations=[
                dict(
                    text=f"Total<br>{total}",
                    x=0.5, y=0.5,
                    font_size=18,
                    showarrow=False
                )
            ]
        )

        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Recommended Action")

        if threat_percent == 0:
            st.write("No action required. Continue periodic monitoring.")
        elif threat_percent < 5:
            st.write("Monitor network behavior and re-check logs.")
        elif threat_percent < 20:
            st.write("Investigate abnormal IPs and firewall alerts.")
        else:
            st.write("Immediate security review required.")

    # =====================================================
    # ADVANCED MODE
    # =====================================================
    if mode == "Advanced Mode":

        st.header("Advanced Threat Analytics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Threats Detected", threats)
        col3.metric("Threat %", f"{threat_percent:.2f}%")

        # Feature Importance
        st.subheader("Top Influential Features")
        importances = model.feature_importances_
        feature_names = X_train.columns

        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)

        fig_feat = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation='h'
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        # Confusion Matrix
        if y_test is not None:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            fig_cm = px.imshow(cm, text_auto=True)
            st.plotly_chart(fig_cm, use_container_width=True)

            accuracy = accuracy_score(y_test, predictions)
            st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

        # Confidence Histogram
        st.subheader("Prediction Confidence Distribution")
        hist = px.histogram(test_data, x="Confidence", nbins=30)
        st.plotly_chart(hist, use_container_width=True)

        # Filter by Attack Type
        st.subheader("Filter by Predicted Attack Type")
        attack_types = test_data["Predicted_Class"].unique()
        selected_attack = st.selectbox("Select Attack Type", attack_types)
        filtered = test_data[test_data["Predicted_Class"] == selected_attack]
        st.dataframe(filtered.head(50))

        # High Risk Records
        st.subheader("Top High-Risk Records")
        high_risk = test_data.sort_values(by="Confidence", ascending=False).head(10)
        st.dataframe(high_risk)

        # Dataset Explorer
        st.subheader("Dataset Explorer")
        st.dataframe(test_data.head(100))

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------
    csv = test_data.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Security Report",
        data=csv,
        file_name="security_analysis_report.csv",
        mime="text/csv"
    )
