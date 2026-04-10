import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import joblib
import numpy as np
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# Load same model as API
model = joblib.load(r"C:\Users\Mad_15\Dropbox\PC\Desktop\fraud-system\best_model.pkl")
FEATURE_ORDER = joblib.load(r"C:\Users\Mad_15\Dropbox\PC\Desktop\fraud-system\feature_order.pkl")


train_df = pd.read_csv(r"C:\Users\Mad_15\Dropbox\PC\Desktop\fraud-system\train_transactions.csv")
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

/* Metric Cards */
.metric-box {
    background: #ffffff;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}

/* Sidebar Analyze Button ALWAYS BLUE */
section[data-testid="stSidebar"] .stButton > button {
    background-color:#2563eb;
    color:white;
    border:none;
    font-weight:600;
    border-radius:8px;
}

/* Toggle Button Base */
div.stButton > button {
    height: 45px;
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #d1d5db;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("## 🛡 Fraud Detection System")
st.caption("Real-time transaction analysis powered by AI")

# ---------- SESSION STATE ----------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Prediction"

if "result" not in st.session_state:
    st.session_state.result = None

# ---------- TOGGLE HANDLERS ----------
def set_prediction():
    st.session_state.active_tab = "Prediction"

def set_insights():
    st.session_state.active_tab = "Insights"

# ---------- TOGGLE UI ----------
col1, col2 = st.columns(2, gap="small")

with col1:
    st.button("Prediction", on_click=set_prediction, use_container_width=True)

with col2:
    st.button("Insights", on_click=set_insights, use_container_width=True)


# Dynamic styling for toggle
if st.session_state.active_tab == "Prediction":
    pred_style = "background-color:#2563eb;color:white;border:none;"
    ins_style = "background-color:#f3f4f6;color:black;"
    exp_style = "background-color:#f3f4f6;color:black;"

elif st.session_state.active_tab == "Insights":
    pred_style = "background-color:#f3f4f6;color:black;"
    ins_style = "background-color:#2563eb;color:white;border:none;"
    exp_style = "background-color:#f3f4f6;color:black;"


st.markdown(f"""
<style>
div[data-testid="column"]:nth-of-type(1) div.stButton > button {{
    {pred_style}
}}
div[data-testid="column"]:nth-of-type(2) div.stButton > button {{
    {ins_style}
}}
</style>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------- SIDEBAR ----------
# ---------- SIDEBAR ----------
st.sidebar.header("Transaction Analysis")

mode = st.sidebar.radio(
    "Select Mode",
    ["Manual Input", "Live Transactions"]
)

# ============================
# 🔵 MANUAL MODE
# ============================
if mode == "Manual Input":

    amount = st.sidebar.number_input("Transaction Amount (INR)", 100, 100000, 25000)
    velocity = st.sidebar.number_input("Velocity (last 1 hour)", 0, 20, 7)
    distance = st.sidebar.number_input("Distance from Home (km)", 0, 2000, 450)

    if st.sidebar.button("Analyze Transaction"):

        data = {
            "transaction_amount": amount,
            "velocity_last_1h": velocity,
            "distance_from_home_km": distance
        }

        response = requests.post("http://localhost:8000/predict", json=data)

        st.session_state.result = response.json()
        st.session_state.latest_input = data


# ============================
# 🔴 LIVE MODE
# ============================
if mode == "Live Transactions":

    try:
        response = requests.get("http://localhost:8000/random-transaction")

        if response.status_code != 200:
            st.error(f"API Error: {response.text}")
        else:
            data = response.json()

            if isinstance(data, dict) and "analysis" in data:
                st.session_state.result = data["analysis"]
                st.session_state.latest_input = data["input"]

            else:
                st.error(f"Invalid API response: {data}")

    except Exception as e:
        st.error(f"API Error: {e}")

# ---------- DISPLAY ----------
if st.session_state.result:

    result = st.session_state.result
    prob = result['fraud_probability']
    risk = result['risk_level']
    action = result['recommended_action']

    percent = int(prob * 100)

    # ==============================
    # 🔴 PREDICTION TAB
    # ==============================
    if st.session_state.active_tab == "Prediction":

        # GET AI NOTE
        # GET AI NOTE
        ai_note = result.get("ai_investigation_note", "No explanation available")

        # FORMAT MESSAGE
        message = ai_note.replace(", ", ". ")

        # COLOR LOGIC
        if risk == "HIGH":
            st.error(f"🚨 HIGH Risk Alert - CRITICAL")

        elif risk == "MEDIUM":
            st.warning(f"⚠️ MEDIUM Risk Alert")

        else:
            st.success(f"✅ LOW Risk Alert")

        # METRICS
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"<div class='metric-box'><h4>Fraud Probability</h4><h1>{percent}%</h1></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Risk Level</h4><h1>{risk}</h1></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Action</h4><h2>{action.split()[0]}</h2></div>", unsafe_allow_html=True)

        # AI EXPLANATION BOX
        st.markdown("### AI Investigation Insight")

        st.info(message)
        st.markdown("---")

        # GAUGE
        st.subheader("Fraud Risk Meter")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percent,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2c3e50"},
                'steps': [
                    {'range': [0, 30], 'color': "#1abc9c"},
                    {'range': [30, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#e74c3c"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # SUMMARY
        st.markdown("### Transaction Summary")

        colA, colB, colC = st.columns(3)
        input_data = st.session_state.get("latest_input", {})

        colA.metric("Amount", f"₹{input_data.get('transaction_amount', 0)}")
        colB.metric("Velocity", f"{input_data.get('velocity_last_1h', 0)} txns/hr")
        colC.metric("Distance", f"{input_data.get('distance_from_home_km', 0)} km")

    # ==============================
    # 🔵 INSIGHTS TAB
    # ==============================
    elif st.session_state.active_tab == "Insights":

        st.subheader("📊 Fraud Analytics Dashboard")

        import plotly.express as px

        # -------------------------------
        # LOAD DATA (IMPORTANT)
        # -------------------------------
        try:
            df = pd.read_csv(r"C:\Users\Mad_15\Dropbox\PC\Desktop\fraud-system\train_transactions.csv")
        except:
            st.warning("Using demo data (dataset not found)")
            import numpy as np
            df = pd.DataFrame({
                "transaction_day_of_week": np.random.choice(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 200),
                "transaction_amount": np.random.randint(100, 50000, 200),
                "is_fraud": np.random.choice([0,1], 200),
                "is_international": np.random.choice([0,1], 200)
            })

        # -------------------------------
        # CREATE GRID LAYOUT
        # -------------------------------
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # -------------------------------
        # 1️⃣ WEEKLY FRAUD TRENDS
        # -------------------------------
        with col1:
            st.markdown("### 📅 Weekly Fraud Trends")

            weekly = df.groupby("transaction_day_of_week")["is_fraud"].sum().reset_index()

            fig1 = px.line(
                weekly,
                x="transaction_day_of_week",
                y="is_fraud",
                markers=True
            )

            st.plotly_chart(fig1, use_container_width=True)


        # -------------------------------
        # 2️⃣ AMOUNT vs FRAUD
        # -------------------------------
        with col2:
            st.markdown("### 💰 Amount vs Fraud")

            fig2 = px.box(
                df,
                x="is_fraud",
                y="transaction_amount_inr",
                color="is_fraud"
            )

            st.plotly_chart(fig2, use_container_width=True)


        # -------------------------------
        # 3️⃣ DOMESTIC vs INTERNATIONAL
        # -------------------------------
        with col3:
            st.markdown("### 🌍 Domestic vs International")

            intl = df.groupby("is_international")["is_fraud"].mean().reset_index()

            fig3 = px.bar(
                intl,
                x="is_international",
                y="is_fraud",
                color="is_international"
            )

            fig3.update_xaxes(
                tickvals=[0, 1],
                ticktext=["Domestic", "International"]
            )

            st.plotly_chart(fig3, use_container_width=True)


        # -------------------------------
        # 4️⃣ VELOCITY vs AMOUNT
        # -------------------------------
        with col4:
            st.markdown("### ⚡ Velocity vs Amount")

            fig4 = px.scatter(
                df,
                x="velocity_last_1h",
                y="transaction_amount_inr",
                color="is_fraud",
                opacity=0.7
            )

            st.plotly_chart(fig4, use_container_width=True)

