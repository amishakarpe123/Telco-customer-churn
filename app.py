import streamlit as st
import pickle

model = pickle.load(
open(
'churn_model.pkl',
'rb'
)
)

st.title(
"Customer Churn Prediction"
)

tenure = st.number_input(
"Tenure"
)

monthly = st.number_input(
"Monthly Charges"
)

if st.button("Predict"):
    
    result = model.predict(
        [[tenure, monthly]]
    )
    
    if result[0] == 1:
        st.error(
        "Customer Will Churn"
        )
    else:
        st.success(
        "Customer Will Stay"
        )
