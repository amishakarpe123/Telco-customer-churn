import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Customer Prediction Dashboard",
    page_icon="🔮",
    layout="wide"
)

# FIXED: Changed unsafe_allowed_html=True to unsafe_allow_html=True
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except:
        return None

lottie_anim = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_qpwb7asv.json")

# 2. LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    try:
        with open("model (7).pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model pickle file: {e}")
    st.stop()

st.markdown("<h1 class='main-title'>🔮 Customer Predictive Analytics</h1>", unsafe_allow_html=True)

with st.sidebar:
    if lottie_anim:
        st_lottie(lottie_anim, height=200, key="sidebar_anim")
    else:
        st.write("🤖 **AI Model Service**")
    st.markdown("---")
    st.info("Adjust attributes and click Predict.")

# 3. USER INPUT FIELDS
st.markdown("### 📝 Customer Profile Details")
tab1, tab2, tab3 = st.tabs(["👤 Demographics", "📱 Services", "💳 Contract & Charges"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Is Senior Citizen?", ["No", "Yes"])
    with col2:
        Partner = st.selectbox("Has a Partner?", ["No", "Yes"])
        Dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with col2:
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with col3:
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    with col2:
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)

# 4. MAP DATA
input_data = {
    "gender": gender, "SeniorCitizen": 1 if SeniorCitizen == "Yes" else 0, "Partner": Partner,
    "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines,
    "InternetService": InternetService, "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection, "TechSupport": TechSupport, "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges
}

feature_order = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

input_df = pd.DataFrame([input_data])[feature_order]

st.markdown("---")
if st.button("🚀 Run Prediction Analysis", use_container_width=True):
    with st.spinner("Analyzing..."):
        try:
            prediction = model.predict(input_df)[0]
            st.success("Analysis Complete!")
            st.markdown(f"<div class='metric-card'><h3>Target Outcome</h3><h1 style='color: #2563EB;'>{prediction}</h1></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction Error: {e}")


