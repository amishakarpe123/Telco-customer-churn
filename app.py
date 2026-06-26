import streamlit as st
import pickle
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

# Custom Elegant CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 16px;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
        color: white;
    }
    .category-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allowed_html=True)

# 2. HTML/JS Embedded Lottie Animation (Ultra-reliable, no extra libraries needed)
def render_lottie_animation():
    lottie_html = """
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <div style="display: flex; justify-content: center; align-items: center;">
        <lottie-player src="https://assets5.lottiefiles.com/packages/lf20_qpwb75s6.json" 
            background="transparent" speed="1" style="width: 160px; height: 160px;" loop autoplay>
        </lottie-player>
    </div>
    """
    st.components.v1.html(lottie_html, height=170)

# 3. Load Safe Model Cache
@st.cache_resource
def load_model():
    # Make sure your file is exactly named "model (7).pkl" in the same directory
    with open("model (7).pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error loading 'model (7).pkl': {e}")
    st.info("Ensure the model file is uploaded to your repository or directory exactly as named.")
    st.stop()

# 4. Header Section
render_lottie_animation()
st.markdown("<h1 style='text-align: center; color: #1F2937;'>Customer Churn Analytics</h1>", unsafe_allowed_html=True)
st.markdown("<p style='text-align: center; color: #4B5563;'>Predict subscription retention risks using your trained Logistic Regression model.</p>", unsafe_allowed_html=True)
st.markdown("---")

# 5. Form Input Fields
st.markdown("### 📋 Input Customer Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='category-box'>", unsafe_allowed_html=True)
    st.markdown("**Demographics & Status**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    st.markdown("</div>", unsafe_allowed_html=True)
    
    st.markdown("<div class='category-box'>", unsafe_allowed_html=True)
    st.markdown("**Connectivity Details**")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    st.markdown("</div>", unsafe_allowed_html=True)

with col2:
    st.markdown("<div class='category-box'>", unsafe_allowed_html=True)
    st.markdown("**Security & Add-ons**")
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    st.markdown("</div>", unsafe_allowed_html=True)

    st.markdown("<div class='category-box'>", unsafe_allowed_html=True)
    st.markdown("**Billing & Contract**")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0, step=1.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=150.0, step=5.0)
    st.markdown("</div>", unsafe_allowed_html=True)

# 6. Strict Map Feature Conversion 
# (NOTE: If your model expects numeric encoded values instead of strings, 
# this block cleanly converts selections to standard numerical mapping).
binary_map = {"No": 0, "Yes": 1, "Male": 1, "Female": 0}
multi_map = {
    "No phone service": 0, "No internet service": 0,
    "DSL": 1, "Fiber optic": 2, "Month-to-month": 0, "One year": 1, "Two year": 2,
    "Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
}

def transform_input(val):
    if val in binary_map: return binary_map[val]
    if val in multi_map: return multi_map[val]
    return 0 

# Explicit ordered list matching model feature order
features_array = [
    transform_input(gender),
    transform_input(senior_citizen),
    transform_input(partner),
    transform_input(dependents),
    float(tenure),
    transform_input(phone_service),
    transform_input(multiple_lines),
    transform_input(internet_service),
    transform_input(online_security),
    transform_input(online_backup),
    transform_input(device_protection),
    transform_input(tech_support),
    transform_input(streaming_tv),
    transform_input(streaming_movies),
    transform_input(contract),
    transform_input(paperless_billing),
    transform_input(payment_method),
    float(monthly_charges),
    float(total_charges)
]

# Convert structure to 2D numpy array
final_features = np.array([features_array])

# 7. Prediction Block
if st.button("🔮 Run Assessment"):
    with st.spinner("Calculating probability vectors..."):
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)
    
    st.markdown("<br>", unsafe_allowed_html=True)
    if prediction[0] == 1:
        st.error(f"### ⚠️ Churn Alert! \n The model predicts this customer is highly likely to cancel. (Risk Probability: **{probability[0][1]:.1%}**)")
    else:
        st.success(f"### 🎉 High Retention! \n This profile shows stable customer patterns. (Retention Confidence: **{probability[0][0]:.1%}**)")

   
