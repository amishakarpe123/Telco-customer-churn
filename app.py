import streamlit as pd
import streamlit as st
import pickle
import numpy as np
import requests
from streamlit_lottie import st_lottie

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for an attractive UI
st.markdown("""
    <style>
    .main { background-color: #f9fbfd; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(118, 75, 162, 0.4);
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allowed_html=True)

# 2. Load Animated Assets (Lottie)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Loading a sleek tech/analytics animation
lottie_analytics = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qpwb75s6.json")

# 3. Load the Saved Model
@st.cache_resource
def load_model():
    with open("model (7).pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")
    st.stop()

# 4. App Header Animation & Title
col1, col2 = st.columns([1, 2])
with col1:
    if lottie_analytics:
        st_lottie(lottie_analytics, height=150, key="header_anim")
with col2:
    st.title("Customer Insights & Prediction")
    st.caption("Enter customer demographics and service details to predict behavior.")

st.markdown("---")

# 5. Organized Input Form
st.markdown("### 📋 Customer Profile Data")

# Grouping inputs cleanly using columns
with st.container():
    col_a, col_b = st.columns(2)
    
    with col_a:
        gender = st.selectbox("Gender", options=["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen?", options=["No", "Yes"])
        partner = st.selectbox("Has Partner?", options=["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", options=["No", "Yes"])
        phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Type", options=["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security Service", options=["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup Service", options=["Yes", "No", "No internet service"])

    with col_b:
        device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support Plan", options=["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
        payment_method = st.selectbox("Payment Method", options=[
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0, step=1.0)

# 6. Feature Encoding Mapping
# Ensure order perfectly matches feature_names_in_: 
# [gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges]

# Simple structural encoding helpers (Adjust mapping values to match how your model was trained)
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
multi_map = {
    "No phone service": 0, "No internet service": 0, 
    "DSL": 1, "Fiber optic": 2, "Month-to-month": 0, "One year": 1, "Two year": 2,
    "Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}

def encode_val(val):
    if val in binary_map: return binary_map[val]
    if val in multi_map: return multi_map[val]
    return 1 # Fallback default for 'Yes' strings like choices in services

# Package up the features in the exact original array structure
features = [
    encode_val(gender),
    encode_val(senior_citizen),
    encode_val(partner),
    encode_val(dependents),
    float(tenure),
    encode_val(phone_service),
    encode_val(multiple_lines),
    encode_val(internet_service),
    encode_val(online_security),
    encode_val(online_backup),
    encode_val(device_protection),
    encode_val(tech_support),
    encode_val(streaming_tv),
    encode_val(streaming_movies),
    encode_val(contract),
    encode_val(paperless_billing),
    encode_val(payment_method),
    float(monthly_charges),
    float(total_charges)
]

# Convert to 2D numpy array for scikit-learn
input_data = np.array([features])

st.markdown("<br>", unsafe_allowed_html=True)

# 7. Prediction Trigger
if st.button("🔮 Analyze & Predict"):
    with st.spinner("Processing structural patterns..."):
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
        
    st.markdown("---")
    
    # Custom colored metric output based on outcome
    if prediction[0] == 1:
        st.error(f"⚠️ **High Churn Risk Predicted!** Probability: {probabilities[0][1]:.2%}")
    else:
        st.success(f"🎉 **Low Risk / Loyal Customer Profile.** Probability of retention: {probabilities[0][0]:.2%}")
